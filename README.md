# Fed-MANTIS

**Communication-Efficient Federated Adaptation of Time-Series Foundation Models for Wearable Human Activity Recognition**

This repository contains the implementation and supporting materials for **Fed-MANTIS**, a cloud-edge federated learning framework for wearable Human Activity Recognition (HAR). Fed-MANTIS combines frozen **MANTIS time-series foundation-model embeddings** with compact trainable task-specific modules, allowing federated adaptation without communicating the foundation-model backbone.

The framework is evaluated on **PAMAP2** and **MHEALTH** under strict **Leave-One-Subject-Out (LOSO)** evaluation. In the federated setting, each training subject is treated as one client, while the held-out subject is used only for testing.

## Overview

Fed-MANTIS is designed to improve cross-subject wearable HAR while preserving data locality and reducing the amount of communicated model parameters.

The main idea is:

- raw wearable sensor windows remain local to each subject/client,
- a pretrained MANTIS backbone is used as a frozen feature extractor,
- only compact task-specific modules are trained,
- the server aggregates task-module parameters using weighted FedAvg,
- the foundation-model backbone is never federated or updated.

The evaluated MANTIS-based task modules include:

- linear probe,
- low-rank bottleneck adapter,
- supervised contrastive low-rank adapter,
- confusion-aware low-rank adapter.

The repository also includes conventional raw-window baselines for comparison.

## Repository Structure

```text
Fed-MANTIS/
│
├── models/                         # Model definitions
│   ├── cnn.py
│   ├── lstm.py
│   ├── deepConvLstmForFed.py
│   ├── inceptiontime.py
│   ├── linear_head.py
│   └── adapted_mantis_head.py
│
├── utils/                          # Dataset, LOSO split, training, and metrics utilities
│   ├── dataset.py
│   ├── dataset_embeddings.py
│   ├── loso_split.py
│   ├── train.py
│   └── metrics.py
│
├── experiments/                    # Embedding extraction, diagnostics, and visualisation scripts
│   ├── extract_mantis_embeddings.py
│   ├── extract_mantis_embeddings_per_subject.py
│   ├── activityEmbeddingSeparability.py
│   ├── confusionMatrix.py
│   └── ...
│
├── src-mhealth/                    # MHEALTH preprocessing and embedding scripts
│   ├── preprocess_mhealth.py
│   ├── create_mhealth_subject_cache.py
│   └── extract_mhealth_mantis_embeddings.py
│
├── run.py                          # Centralized raw-window baselines
├── runMantisembeddings.py           # Centralized MANTIS-based experiments
├── run_fedavg_baseline.py           # Federated raw-window baselines
├── run_fedavg_mantis_adapter.py     # Fed-MANTIS experiments
│
├── requirements.txt
├── requirements_mantis.txt
├── LICENSE
└── README.md
```

## Datasets

This repository does **not** redistribute the original datasets.

Please download the datasets from the official UCI Machine Learning Repository pages:

- PAMAP2 Physical Activity Monitoring:  
  https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

- MHEALTH Dataset:  
  https://archive.ics.uci.edu/dataset/319/mhealth+dataset

After downloading, place the datasets in local data folders as required by the preprocessing scripts.

Recommended local structure:

```text
Fed-MANTIS/
│
├── PAMAP2_Dataset/
│   └── Protocol/
│
├── data/
│
└── data_mhealth/
```

Dataset folders, cached embeddings, checkpoints, and logs are ignored by `.gitignore` and should not be committed.

## Environment Setup

Python 3.11 is recommended.

### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements_mantis.txt --no-cache-dir
```

To verify the installation:

```bash
python -c "import torch, numpy; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('NumPy:', numpy.__version__)"
```

## Preprocessing

### PAMAP2

The PAMAP2 preprocessing pipeline is implemented in:

```text
dataloader.py
```

It performs the following steps:

- removes activity label `0` (`other`),
- excludes subject 109 because it contains too few valid windows from a single activity class,
- uses wrist, chest, and ankle IMU streams,
- creates 51-channel multivariate sensor windows,
- applies forward-fill imputation within each subject,
- applies subject-specific normalisation,
- segments data into 512-sample windows with 50% overlap,
- remaps activity labels to contiguous class indices.

Run:

```bash
python dataloader.py
```

This generates files such as:

```text
data/X.npy
data/y.npy
data/subjects.npy
data/class_mapping.json
```

### MHEALTH

The MHEALTH preprocessing scripts are located in:

```text
src-mhealth/
```

Run:

```bash
python src-mhealth/preprocess_mhealth.py
```

This prepares the MHEALTH window-level data used for centralized and federated experiments.

## Extracting Frozen MANTIS Embeddings

### PAMAP2 centralized embeddings

```bash
python experiments/extract_mantis_embeddings.py
```

Output:

```text
data/X_mantis.npy
```

### PAMAP2 per-subject embeddings

```bash
python experiments/extract_mantis_embeddings_per_subject.py
```

Output:

```text
data/mantis_subject_cache/subject_101.npz
data/mantis_subject_cache/subject_102.npz
...
```

### MHEALTH embeddings

```bash
python src-mhealth/extract_mhealth_mantis_embeddings.py
python src-mhealth/create_mhealth_subject_cache.py
```

These scripts create MHEALTH MANTIS embedding files and per-subject caches for LOSO-FL experiments.

## Running Experiments

### Centralized raw-window baselines

Run CNN, LSTM, DeepConvLSTM, and InceptionTime on raw sensor windows:

```bash
python run.py
```

The script supports LOSO and random-split settings through the `SPLIT_MODE` configuration inside the file.

For the main paper experiments, use:

```python
SPLIT_MODE = "loso"
```

### Centralized MANTIS-based experiments

Run centralized experiments on frozen MANTIS embeddings:

```bash
python runMantisembeddings.py
```

This evaluates MANTIS-based variants such as:

```text
linear_probe
adapter_lowrank_128_r32
adapter_lowrank_256_r64
supcon_lowrank_128_r32
supcon_lowrank_256_r64
confusion_lowrank_128_r32_m05
confusion_lowrank_128_r32_m10
```

### Federated raw-window baselines

Run subject-level FedAvg on raw windows:

```bash
python run_fedavg_baseline.py
```

This script:

- treats each training subject as one client,
- holds out one subject for testing,
- repeats the process across LOSO folds,
- reports accuracy, macro-F1, parameter count, and communication cost.

### Fed-MANTIS

Run subject-level FedAvg on frozen MANTIS embeddings:

```bash
python run_fedavg_mantis_adapter.py
```

This is the main Fed-MANTIS script. It evaluates federated MANTIS task-module variants including:

```text
baseline_linear
adapter_lowrank_128_r32
adapter_lowrank_256_r64
supcon_lowrank_128_r32
supcon_lowrank_256_r64
confusion_lowrank_128_r32_m05
confusion_lowrank_128_r32_m10
```

In Fed-MANTIS, only the classifier or adapter-classifier parameters are communicated. The frozen MANTIS backbone remains local and is not updated.

## Evaluation Protocols

### Leave-One-Subject-Out

LOSO is the main evaluation protocol.

For each fold:

- one subject is held out completely for testing,
- the remaining subjects are used for training,
- no windows from the held-out subject appear during training.

This evaluates generalisation to unseen users.

### LOSO-FL

In the federated setting:

- each training subject corresponds to one client,
- the held-out subject is used only for testing,
- the server aggregates trainable task-module parameters using weighted FedAvg.

### Random Split

Random-split experiments are included only for protocol-gap analysis. They are not the main evaluation protocol because windows from the same subject may appear in both training and testing subsets.

## Hyperparameter Search

The main search space includes:

```text
Centralized epochs: 50
Federated communication rounds: 30
Batch size: 64
Optimizers: SGD, Adam
Learning rates:
  - SGD: 1e-2, 1e-3
  - Adam: 1e-3, 1e-4
Local epochs: 1, 3
Adapter configurations:
  - adapter dimension 128, rank 32
  - adapter dimension 256, rank 64
SupCon loss weight: 0.01, 0.05, 0.1
Confusion margin: 0.5, 1.0
Confusion loss weight: 0.05
```

The best configuration for each model family is selected based on mean LOSO accuracy across held-out subjects.

## Reproducibility Notes

Fixed random seeds are used where possible for model initialisation, data splitting, and client simulation. Small differences may occur due to hardware, CUDA versions, PyTorch versions, and nondeterministic GPU operations.

For exact reproduction, use the same dataset versions, preprocessing settings, cached MANTIS embeddings, and script configurations described above.

## Results and Supplementary Materials

Processed result tables and supplementary materials should be placed in:

```text
results/
supplementary/
```

The original datasets are not redistributed. Instead, this repository provides preprocessing and experiment scripts for reproducing the reported results from the official dataset sources.

## MANTIS Citation and License

This project uses the MANTIS time-series foundation model as a frozen embedding extractor. MANTIS is licensed under the Apache License 2.0. Please refer to the original MANTIS repository and license for full details.

If you use MANTIS, please cite:

```bibtex
@article{feofanov2025mantis,
  title={Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification},
  author={Vasilii Feofanov and Songkang Wen and Marius Alonso and Romain Ilbert and Hongbo Guo and Malik Tiomoko and Lujia Pan and Jianfeng Zhang and Ievgen Redko},
  journal={arXiv preprint arXiv:2502.15637},
  year={2025}
}
```



## License

This repository is released under the license included in `LICENSE`.
