import pandas as pd
import re

def overlap_len(site_start, site_end, region_start, region_end):
    """
    All coordinates are 1-based inclusive.
    """
    return max(0, min(site_end, region_end) - max(site_start, region_start) + 1)


def get_region_with_max_overlap(mrna_id, start, end):
    regions = {}

    for region_name in ["CDS", "UTR3", "UTR5"]:
        match = re.search(fr'{region_name}:(\d+)-(\d+)', mrna_id)

        if match:
            region_start, region_end = map(int, match.groups())
            regions[region_name] = overlap_len(start, end, region_start, region_end)

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


def region_annotate(df):
    for i, row in df.iterrows():

        mrna_seq = row["mRNA"]
        mrna_id = row["saccver_mRNA"]
        ori_seq = row["Full_mRNA_Seq"]

        start0 = ori_seq.find(mrna_seq)

        if start0 == -1:
            df.loc[i, "region"] = "Not_found"
            df.loc[i, "site_start"] = -1
            df.loc[i, "site_end"] = -1
            continue

        start = start0 + 1
        end = start + len(mrna_seq) - 1

        result = get_region_with_max_overlap(mrna_id, start, end)

        df.loc[i, "site_start"] = start
        df.loc[i, "site_end"] = end
        df.loc[i, "region"] = result["region"]
        df.loc[i, "CDS_overlap"] = result["CDS_overlap"]
        df.loc[i, "UTR3_overlap"] = result["UTR3_overlap"]
        df.loc[i, "UTR5_overlap"] = result["UTR5_overlap"]

    return df


if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\16892\OneDrive - University of Central Florida\BIOINFORMATICS LAB\isomir_mrna_interaction_prediction\notebooks\post_process_data_sample.csv")

    output_df = region_annotate(df)

    print(output_df["region"].value_counts())

    #output_df.to_csv(r"C:\Users\16892\OneDrive - University of Central Florida\BIOINFORMATICS LAB\isomir_mrna_interaction_prediction\notebooks\output_data.csv", index=False)
