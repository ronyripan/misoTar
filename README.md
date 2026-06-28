# misoTar

misoTar is a deep learning framework for predicting miRNA/isomiR–mRNA interactions.

## Repository Contents

This repository contains the following files:

| File | Description |
|------|-------------|
| `misoTar.py` | Main script for training and evaluating the misoTar model. |
| `sample_train_df.csv` | Example training dataset. |
| `sample_test_df.csv` | Example test dataset. |
| `requirements.txt` | List of required Python packages and their versions. |
| `post_process_data_sample.csv` | Example input dataset to get interaction scheme and transcript region annotation. |
| `interaction_postprocess.py` | Post-processing script that shows interaction between miRNA/isomiR and mRNA and annotates each site as **5′ UTR**, **CDS**, or **3′ UTR** based on the region with the greatest nucleotide overlap. |

Once the training and testing datasets are correctly specified in `misoTar.py`, the model can be trained and evaluated directly.

---

## Dataset Format

Both training and testing datasets must contain the following three columns:

| Column         | Description                                                             |
| -------------- | ----------------------------------------------------------------------- |
| `miRNA/isomiR` | miRNA or isomiR sequence.                                               |
| `mRNA`         | Target mRNA sequence.                                                   |
| `label`        | Interaction label (`1` = interacting pair, `0` = non-interacting pair). |

Example:

| miRNA/isomiR           | mRNA     | label |
| ---------------------- | -------- | ----- |
| UGAGGUAGUAGGUUGUAUAGUU | AUGCU... | 1     |
| UGAGGUAGUAGGUUGUAUAGUU | CCGUA... | 0     |

---

## Installation

### 1. Create and Activate a Conda Environment

```bash
conda create -n misoTar python=3.11
conda activate misoTar
```

If `pip` is not available in the environment:

```bash
conda install pip -y
```

### 2. Install Dependencies

The recommended approach is to install all required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Alternatively, the major dependencies can be installed manually:

```bash
pip install torch torchvision
pip install transformers[torch]
pip install datasets
pip install scikit-learn
```

> **Note:** For GPU support, install the appropriate PyTorch version for your system following the instructions at https://pytorch.org/get-started/locally/.

---

## Running misoTar

1. Update the training and testing dataset paths in `misoTar.py`.
2. Activate the `misoTar` environment.
3. Run the script:

```bash
python misoTar.py
```

The model will train and evaluate using the specified datasets.

---

## Requirements

All package versions used in our experiments are listed in `requirements.txt`.
