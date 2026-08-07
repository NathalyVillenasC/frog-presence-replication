# Frog Presence Replication

Replication study of a frog species presence-absence detection pipeline using acoustic recordings.

This project is based on the paper:

"Presence-absence estimation in audio recordings of tropical frog communities"

([paper](https://arxiv.org/pdf/1901.02495))

The goal of this repository is to reproduce and understand the methodology proposed in the paper, while building a reproducible and well-structured machine learning pipeline for acoustic species detection.

## Project Structure

```text
frog-presence-replication/

├── data/          # External datasets (not included in the repository)
├── models/        # Trained models and checkpoints (not included)
├── notebooks/     # Exploratory analysis and experiments
├── src/           # Reproducible source code
├── tests/         # Unit and integration tests
├── README.md
└── requirements.txt
```

## Reproducibility

The repository is organized to separate exploratory work from reusable code components. Large files such as audio datasets and trained models are not stored in GitHub and should be obtained separately.

More information about installation, data preparation, and execution will be added as the project develops.