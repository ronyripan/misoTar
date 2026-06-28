import pandas as pd
import numpy as np
import re


def reverse_seq(seq):
    """
    Reverse the mRNA sequence so that it is in the opposite
    orientation relative to the miRNA before alignment.
    """
    return seq[::-1]


def overlap_len(site_start, site_end, region_start, region_end):
    """
    Calculate overlap length between a target site and transcript region.
    """
    return max(0, min(site_end, region_end) - max(site_start, region_start) + 1)


def get_region_with_max_overlap(mrna_id, start, end):
    """
    Determine whether a target site overlaps CDS, 3'UTR, or 5'UTR.
    """
    regions = {}

    for region_name in ["CDS", "UTR3", "UTR5"]:
        match = re.search(fr"{region_name}:(\d+)-(\d+)", str(mrna_id))

        if match:
            region_start, region_end = map(int, match.groups())
            regions[region_name] = overlap_len(
                start,
                end,
                region_start,
                region_end
            )

    if not regions:
        return pd.Series({
            "region": "Unknown",
            "CDS_overlap": 0,
            "UTR3_overlap": 0,
            "UTR5_overlap": 0
        })

    max_region = max(regions, key=regions.get)

    if regions[max_region] == 0:
        max_region = "Unknown"

    return pd.Series({
        "region": max_region,
        "CDS_overlap": regions.get("CDS", 0),
        "UTR3_overlap": regions.get("UTR3", 0),
        "UTR5_overlap": regions.get("UTR5", 0)
    })


def Smith_Waterman(seq1, seq2):
    """
    Perform modified Smith-Waterman local alignment.

    Parameters
    ----------
    seq1 : str
        miRNA/isomiR sequence.
    seq2 : str
        Reversed mRNA target-site sequence.
    """

    gap = -1

    wc4 = ["AT", "TA", "GC", "CG"]
    wobble = ["GT", "TG"]

    match_score = {
        "AT": 1, "TA": 1, "CG": 1, "GC": 1,
        "GT": 0, "TG": 0,
        "AC": -1, "CA": -1,
        "AG": -1, "GA": -1,
        "TC": -1, "CT": -1,
        "AA": -1, "CC": -1, "GG": -1, "TT": -1,
    }

    position = {
        "stop": 0,
        "left": 1,
        "up": 2,
        "left_up": 3
    }

    m = len(seq1)
    n = len(seq2)

    score_matrix = np.zeros((m + 1, n + 1))
    tracing_matrix = np.zeros((m + 1, n + 1))

    max_score = 0
    max_index = (0, 0)

    # Fill dynamic programming matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            pair = seq1[i - 1] + seq2[j - 1]
            match_value = match_score.get(pair, -1)

            left_up = score_matrix[i - 1, j - 1] + match_value
            up = score_matrix[i - 1, j] + gap
            left = score_matrix[i, j - 1] + gap

            score_matrix[i, j] = max(left_up, left, up, 0)

            if score_matrix[i, j] == 0:
                tracing_matrix[i, j] = position["stop"]
            elif score_matrix[i, j] == left_up:
                tracing_matrix[i, j] = position["left_up"]
            elif score_matrix[i, j] == left:
                tracing_matrix[i, j] = position["left"]
            else:
                tracing_matrix[i, j] = position["up"]

            if score_matrix[i, j] > max_score:
                max_index = (i, j)
                max_score = score_matrix[i, j]

    # If no positive local alignment exists
    if max_score == 0:
        return 0, "No positive local alignment"

    align_seq1 = ""
    align_seq2 = ""

    max_i, max_j = max_index

    # Trace back from the highest-scoring cell to recover local alignment
    while tracing_matrix[max_i, max_j] != position["stop"]:
        if tracing_matrix[max_i, max_j] == position["left_up"]:
            align_seq1 = seq1[max_i - 1] + align_seq1
            align_seq2 = seq2[max_j - 1] + align_seq2
            max_i -= 1
            max_j -= 1

        elif tracing_matrix[max_i, max_j] == position["left"]:
            align_seq1 = "-" + align_seq1
            align_seq2 = seq2[max_j - 1] + align_seq2
            max_j -= 1

        elif tracing_matrix[max_i, max_j] == position["up"]:
            align_seq1 = seq1[max_i - 1] + align_seq1
            align_seq2 = "-" + align_seq2
            max_i -= 1

    match_line = ""

    # Create human-readable alignment scheme
    for a, b in zip(align_seq1, align_seq2):

        if a != "-" and b != "-":
            pair = a + b

            if pair in wc4:
                match_line += "|"
            elif pair in wobble:
                match_line += ":"
            else:
                match_line += "."
        else:
            match_line += " "

    alignment_scheme = f"{align_seq1}\n{match_line}\n{align_seq2}"

    return max_score, alignment_scheme


def region_annotate(df):
    """
    Annotate each predicted miRNA/isomiR-mRNA interaction.

    For each row, this function:
    1. Finds the predicted target sequence in the full transcript.
    2. Converts the location to 1-based transcript coordinates.
    3. Assigns the site to CDS, 3'UTR, 5'UTR, or Unknown.
    4. Performs modified Smith-Waterman alignment.
    """

    for i, row in df.iterrows():

        # Convert U to T so RNA sequences can be handled consistently
        mrna_seq = str(row["mRNA"]).upper().replace("U", "T")
        mirna_seq = str(row["miRNA/isomiR"]).upper().replace("U", "T")
        mrna_id = row["saccver_mRNA"]
        ori_seq = str(row["Full_mRNA_Seq"]).upper().replace("U", "T")

        # Locate the target-site sequence in the full transcript
        start0 = ori_seq.find(mrna_seq)

        if start0 == -1:
            df.loc[i, "region"] = "Not_found"
            df.loc[i, "site_start"] = -1
            df.loc[i, "site_end"] = -1
            df.loc[i, "alignment_score"] = -1
            df.loc[i, "alignment_scheme"] = "Not_found"
            continue

        # Convert Python 0-based index to 1-based transcript coordinates
        start = start0 + 1
        end = start + len(mrna_seq) - 1

        # Identify target-site region based on maximum nucleotide overlap
        result = get_region_with_max_overlap(mrna_id, start, end)

        # Reverse mRNA target-site sequence before alignment
        mrna_rev = reverse_seq(mrna_seq)

        # Reconstruct miRNA/isomiR-mRNA pairing scheme
        score, alignment_scheme = Smith_Waterman(
            mirna_seq,
            mrna_rev
        )

        # Store transcript-region annotation results
        df.loc[i, "site_start"] = start
        df.loc[i, "site_end"] = end
        df.loc[i, "region"] = result["region"]
        df.loc[i, "CDS_overlap"] = result["CDS_overlap"]
        df.loc[i, "UTR3_overlap"] = result["UTR3_overlap"]
        df.loc[i, "UTR5_overlap"] = result["UTR5_overlap"]

        # Store alignment results
        df.loc[i, "alignment_score"] = score
        df.loc[i, "alignment_scheme"] = alignment_scheme

    return df


if __name__ == "__main__":

    input_file = r"C:\Users\16892\OneDrive - University of Central Florida\BIOINFORMATICS LAB\isomir_mrna_interaction_prediction\notebooks\post_process_data_sample.csv"

    output_file = r"C:\Users\16892\OneDrive - University of Central Florida\BIOINFORMATICS LAB\isomir_mrna_interaction_prediction\notebooks\output_data.csv"

    df = pd.read_csv(input_file)

    output_df = region_annotate(df)

    output_df.to_csv(output_file, index=False)

    print("Output saved to:", output_file)
