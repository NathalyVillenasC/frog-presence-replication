# Frog Presence Replication

Replication study of a frog species presence-absence detection pipeline using acoustic recordings.

This project is based on the paper:

"Presence-absence estimation in audio recordings of tropical frog communities"

([paper](https://arxiv.org/pdf/1901.02495))

The goal of this repository is to reproduce and understand the methodology proposed in the paper as closely as the available data allow, while building a reproducible and well-structured machine learning pipeline for acoustic species detection. Where the delivered material differs from or is incomplete relative to the data described in the paper, those differences are documented explicitly and treated as limitations of the replication.

## Project Structure

```text
frog-presence-replication/

├── data/          # External datasets (not included in the repository)
├── models/        # Trained models and checkpoints (not included)
├── notebooks/     # Exploratory analysis and experiments
├── src/           # Reproducible source code
├── tests/         # Unit and integration tests
├── .env.example   # Template for the local dataset locations
├── README.md
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/NathalyVillenasC/frog-presence-replication.git
cd frog-presence-replication

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

Then tell the project where the audio lives:

```bash
copy .env.example .env          # Windows
cp .env.example .env            # macOS / Linux
```

Edit `.env` and set the three paths to your own copies of the collections. The folder names
are up to you — only the structure inside each one matters. `.env` is personal and is never
committed; `.env.example` is the template that is.

## Data

The audio is **not** stored in this repository: the three collections together weigh about
35 GB. They are obtained separately.

| Dataset | Contents | How to obtain it |
|---|---|---|
| 1 | Labelled frog calls, one folder per species, plus cue-point annotations and MATLAB segment files | Published by the authors on [Mendeley Data](https://data.mendeley.com/datasets/5j852hzfjs/1) and downloadable by anyone |
| 2 | Pond recordings (P1 and P2) digitised from cassettes, one folder per cassette with an A and a B side | **Not publicly available.** Obtained directly from the researchers |
| 3 | Clean species calls, two files per species | **Not publicly available.** Obtained directly from the researchers |

Datasets 2 and 3 were provided by the researchers behind the original study and are not
redistributed here. Anyone wishing to reproduce the full analysis has to request them.

## Notebooks

| Notebook | What it does |
|---|---|
| [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | Audits the three collections: what is on disk, which part matches the material the paper describes, what labels the delivered material carries and where they are written, and what cannot be reconstructed from it. This notebook is strictly read-only.  |

## Reproducibility

The repository is organized to separate exploratory work from reusable code components. Large
files such as audio datasets and trained models are not stored in GitHub and should be obtained
separately.

Claims in the notebooks are tagged by the kind of evidence behind them: `[PAPER]` for statements
made in the article, `[SOURCE]` for statements made by its authors in direct communication,
`[OBSERVED]` for measurements visible in the output of the cell above, and `[INFERENCE]` for
readings that are plausible but not provable from the data.
