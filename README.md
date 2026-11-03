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
├── notebooks/     # One notebook per stage of the article, in order (superseded work: branch archive/previous-notebooks)
├── src/           # Reproducible source code shared by the notebooks (src/frog_presence)
├── tests/         # Unit tests, and checks that the documentation matches the notebooks
├── .env.example   # Template for the local dataset locations
├── pyproject.toml # Package metadata, so src/ is importable
├── CLAUDE.md      # Working rules for the repository
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
pip install -e .                # makes src/frog_presence importable
```

Optionally, scoring can run on an NVIDIA GPU. Install the CuPy build that matches your CUDA version
(for example `pip install cupy-cuda12x`). Nothing requires it: `frog_presence.device.array_module()`
tries the GPU and falls back to NumPy on the CPU, with the same results, and no notebook installs anything.
Only the scoring of candidate segments uses it; the signal processing runs on the CPU.

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
| [`02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb) | Turns every annotated call into 20 modified cepstral coefficients per frame (pre-emphasis, framing, Hamming window, DFT, 40-filter bank, logarithm, DCT) and writes the feature matrix, the labels and the call table that the modelling notebooks read. |
| [`03_gmm_parameter_investigation.ipynb`](notebooks/03_gmm_parameter_investigation.ipynb) | Trains one diagonal GMM per species on 6, 12 and 18 seconds of calls with 2 to 64 Gaussians, over ten random selections, and measures the weighted error rate (Section 4.1 and Table 2 of the article). Says where the replication agrees with the article and where it does not. The training and scoring code lives in `src/frog_presence/gmm.py`. |
| [`04_feature_comparison.ipynb`](notebooks/04_feature_comparison.ipynb) | Compares 12 MFCC, 20 MFCC, 20 PLP-RASTA and the 20 m-FCC of notebook 02 on identical splits and random starts (Sections 4.2 and Figure 9), and reproduces Figure 8 with 12 MFCC. Quantifies the article's claims, examines how the RASTA filter's start affects PLP-RASTA, and shows the effect of holding whole recordings out. The extractors live in `src/frog_presence/features.py`. |
| [`05_model_selection.ipynb`](notebooks/05_model_selection.ipynb) | Chooses the feature set and the number of Gaussians with whole recordings kept apart between training and validation, the situation of the pond recordings. Compares the article's call-level protocol with the recording-level one, ranks six feature sets (including both RASTA starts), optimises duration and model order, and writes the configuration that the detector reads. The split search lives in `src/frog_presence/gmm.py`. |
| [`06_detector_calibration.ipynb`](notebooks/06_detector_calibration.ipynb) | Builds the detector of Sections 3.5 to 3.8 from the selected model: a fixed recording-level TRAIN / CALIBRATION split, one frozen mixture per species, likelihood-ratio verification against the median of the alternatives, and one threshold per species. Compares the article's one-vs-all threshold rule with the rule that matches the deployed decision, measures how much margin the thresholds have and how much they change with the split, and writes the frozen detector to `models/`. The code lives in `src/frog_presence/detector.py`. |
| [`07_candidate_segmentation.ipynb`](notebooks/07_candidate_segmentation.ipynb) | Builds and tests the front end of Sections 3.2 and 3.3 on the labelled recordings: 30 s blocks, species band-pass filter, 10 ms energy, moving average of 12, and the 3/3 endpoint rule. Measures the coverage of the annotated calls over a grid of the unreported constant `C`, selects one `C` per species, checks the endpoint padding through the frozen detector, and writes the frozen configuration. The code lives in `src/frog_presence/segmentation.py`. |
| [`08_field_inference.ipynb`](notebooks/08_field_inference.ipynb) | Applies the frozen segmentation and detector to the article's 141 ten-minute pond samples (22.38 hours), selected by filename, with pond, date and the species named in the sidecars recovered for each sample. Writes every candidate with its decision and the presence vectors; `f09` is `NA`. Scoring runs on the GPU when one is usable. The code lives in `src/frog_presence/field.py` and `inference.py`. |
| [`09_field_results_analysis.ipynb`](notebooks/09_field_results_analysis.ipynb) | Examines what the field output supports: presence by species, pond and date, what the detector accepts, how much each result depends on the thresholds and on the shortest candidate counted (varied, not tuned), and whether the output separates the samples whose sidecars name a species. No accuracy is computed, because the article's 18 labelled vectors were not delivered. |

Run the notebooks in order, `01` to `09`: each one reads what the previous ones wrote under `data/processed/` and `models/`, and says in its
last sections what it hands over. A change to a stage makes the notebooks after it stale, so they are re-run from that point.
`pytest` runs the unit tests of `src/` and `tests/test_docs_consistency.py`, which checks that the notebooks and this README agree
(every notebook described, every dependency declared, no personal path in an output, every cell executed in order).

## Reproducibility

The repository is organized to separate exploratory work from reusable code components. Large
files such as audio datasets and trained models are not stored in GitHub and should be obtained
separately.

Claims in the notebooks are tagged by the kind of evidence behind them: `[PAPER]` for statements
made in the article, `[SOURCE]` for statements made by its authors in direct communication,
`[OBSERVED]` for measurements visible in the output of the cell above, and `[INFERENCE]` for
readings that are plausible but not provable from the data.
