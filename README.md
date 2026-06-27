# Cloud-Edge Collaborative Federated Learning with On-Device Foundation Models for Efficient Human Activity Recognition

This repository contains the implementation for **Fed-MANTIS**, a cloud-edge federated learning framework for Human Activity Recognition (HAR) on the **PAMAP2** dataset. The framework combines frozen **MANTIS time-series foundation-model embeddings** with compact task-specific adaptation.

The final proposed model, **Fed-MANTIS-SupCon**, uses:

- a frozen MANTIS backbone for embedding extraction,
- a compact low-rank bottleneck adapter,
- a linear classifier,
- supervised contrastive regularisation,
- subject-level Federated Averaging.

The goal is to improve cross-subject HAR performance while keeping raw sensor data local and reducing the amount of communicated model parameters.

---

## 1. Repository Overview

The codebase supports two representation families:

1. **Raw-window models** trained directly on PAMAP2 sensor windows.
2. **Frozen MANTIS embedding models** trained on cached foundation-model embeddings.

The project supports:

- centralised raw-window baselines,
- centralised MANTIS adapter/head variants,
- federated raw-window baselines using FedAvg,
- federated MANTIS adapter variants using FedAvg,
- Leave-One-Subject-Out (LOSO) evaluation,
- random-split evaluation for protocol-gap analysis,
- representation-space diagnostics and visualisation.

---

## 2. Model Families

### Raw-window models

The raw-window baselines are trained directly on PAMAP2 windows:

- CNN
- LSTM
- DeepConvLSTM
- InceptionTime
- optional Transformer if enabled in the code

### MANTIS-based models

The MANTIS-based models use frozen MANTIS embeddings as input:

- Linear Probe
- Low-Rank Adapter, 128/r32
- Low-Rank Adapter, 256/r64
- SupCon Low-Rank Adapter, 128/r32
- SupCon Low-Rank Adapter, 256/r64
- Confusion-Aware Low-Rank Adapter, 128/r32, margin 0.5
- Confusion-Aware Low-Rank Adapter, 128/r32, margin 1.0

The main proposed model reported in the thesis is:

```text
supcon_lowrank_128_r32
````

This corresponds to:

```text
Adapter dimension: 128
Low-rank bottleneck rank: 32
Trainable parameters: 423,564
Single update size: 1.616 MB
```

---

## 3. Expected Project Structure

```text
PampActivity/
│
├── PAMAP2_Dataset/                    # Raw PAMAP2 data
│
├── data/
│   ├── X.npy                          # Raw window data
│   ├── y.npy                          # Labels
│   ├── subjects.npy                   # Subject IDs
│   ├── X_mantis.npy                   # Cached MANTIS embeddings
│   ├── class_mapping.json
│   ├── random/                        # Optional non-overlapping random-split data
│   │   ├── X_random.npy
│   │   ├── y_random.npy
│   │   ├── subjects_random.npy
│   │   └── X_mantis_random.npy
│   └── mantis_subject_cache/
│       ├── subject_101.npz
│       ├── subject_102.npz
│       └── ...
│
├── models/
│   ├── cnn.py
│   ├── lstm.py
│   ├── deepconvlstm.py
│   ├── inceptiontime.py
│   ├── linear_head.py
│   ├── adapted_mantis_head.py
│   └── transformer.py                 # optional
│
├── utils/
│   ├── dataset.py
│   ├── dataset_embeddings.py
│   ├── loso_split.py
│   ├── train.py
│   └── metrics.py
│
├── experiments/
│   ├── extract_mantis_embeddings.py
│   ├── extract_mantis_embeddings_per_subject.py
│   ├── rawSensorSilhouetteRQ3.py
│   └── ...
│
├── dataloader.py
├── run.py
├── runMantisembeddings.py
├── run_fedavg_baseline.py
├── run_fedavg_mantis_adapter.py
└── README.md
```

---

## 4. Environment Setup

This project uses Python and PyTorch. Python 3.11 is recommended.

### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

or in Command Prompt:

```cmd
.venv\Scripts\activate
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

### Verify installation

```bash
python -c "import torch, numpy; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('NumPy:', numpy.__version__)"
```

---

## 5. Dataset Preparation

This project uses the **PAMAP2 Physical Activity Monitoring Dataset**, available from the UCI Machine Learning Repository:

```text
https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
```

Download and extract the dataset so that the protocol files are available at:

```text
PAMAP2_Dataset/Protocol/
```

### Preprocessing

The preprocessing pipeline is implemented in `dataloader.py`.

It performs the following steps:

* excludes Subject 109 because it has too few valid multi-class windows,
* removes activity label 0 (`other`),
* combines selected wrist, chest, and ankle IMU streams,
* produces 51-channel multivariate windows,
* applies forward-fill imputation per subject,
* applies per-subject normalisation,
* segments data into 512-sample windows,
* uses 50% overlap for main LOSO experiments,
* remaps activity labels to contiguous class indices.

Run preprocessing with:

```bash
python dataloader.py
```

This generates:

```text
data/X.npy
data/y.npy
data/subjects.npy
data/class_mapping.json
```

---

## 6. Extracting Frozen MANTIS Embeddings

### 6.1 Centralised embedding extraction

This creates one embedding matrix for all windows.

```bash
python experiments/extract_mantis_embeddings.py
```

Output:

```text
data/X_mantis.npy
```

Use this for centralised MANTIS adapter/head experiments.

### 6.2 Per-subject embedding extraction

This extracts and stores embeddings separately for each subject.

```bash
python experiments/extract_mantis_embeddings_per_subject.py
```

Output:

```text
data/mantis_subject_cache/subject_101.npz
data/mantis_subject_cache/subject_102.npz
...
```

Each cached file contains:

```text
X      MANTIS embeddings
y      Labels
sid    Subject ID
```

Use this for federated MANTIS adapter experiments.

---

## 7. Running Experiments

### 7.1 Centralised raw-window baselines

Run CNN, LSTM, DeepConvLSTM, and InceptionTime on raw windows:

```bash
python run.py
```

Important configuration inside `run.py`:

```python
SPLIT_MODE = "loso"      # strict unseen-subject evaluation
# or
SPLIT_MODE = "random"    # random split for protocol-gap comparison
```

---

### 7.2 Centralised MANTIS adapter/head experiments

Run centralised experiments on frozen MANTIS embeddings:

```bash
python runMantisembeddings.py
```

This script supports:

```python
SPLIT_MODE = "loso"
# or
SPLIT_MODE = "random"
```

It evaluates MANTIS-based variants such as:

```text
linear_probe
adapter_lowrank_128_r32
adapter_lowrank_256_r64
supcon_lowrank_128_r32
supcon_lowrank_256_r64
confusion_lowrank_128_r32_m05
confusion_lowrank_128_r32_m10
```

The main centralised model reported in the thesis is:

```text
supcon_lowrank_128_r32
```

---

### 7.3 Federated raw-window baselines

Run subject-wise FedAvg on raw PAMAP2 windows:

```bash
python run_fedavg_baseline.py
```

This script:

* treats each subject as a client,
* trains on all but one subject,
* evaluates on the held-out subject,
* repeats this in LOSO-FL form,
* reports accuracy, macro-F1, parameter count, and communication cost.

---

### 7.4 Federated MANTIS adapter experiments

Run subject-wise FedAvg on frozen MANTIS embeddings:

```bash
python run_fedavg_mantis_adapter.py
```

This is the main federated MANTIS script used for the final thesis experiments.

It evaluates adapter variants such as:

```text
baseline_linear
adapter_lowrank_128_r32
adapter_lowrank_256_r64
supcon_lowrank_128_r32
supcon_lowrank_256_r64
confusion_lowrank_128_r32_m05
confusion_lowrank_128_r32_m10
```

The main proposed federated model is:

```text
supcon_lowrank_128_r32
```

This corresponds to **Fed-MANTIS-SupCon**.

---

## 8. Evaluation Protocols

### Leave-One-Subject-Out (LOSO)

LOSO is the main evaluation protocol.

For each fold:

* one subject is held out entirely for testing,
* all remaining subjects are used for training,
* no windows from the test subject appear in training.

This measures generalisation to unseen users.

### LOSO-FL

In the federated setting:

* each training subject is treated as one client,
* the held-out subject is used only for testing,
* the server aggregates trainable task-module parameters using FedAvg.

For Fed-MANTIS, only the adapter-classifier parameters are communicated. The MANTIS backbone remains frozen.

### Random Split

A supplementary random-split experiment is used for protocol-gap analysis. It helps quantify how much performance changes when windows from the same subject may appear in both training and testing subsets.

The random-split comparison uses non-overlapping windows where available to reduce leakage from adjacent overlapping windows.

---

## 9. Quick Reproduction Guide

The main thesis experiments can be reproduced with the following workflow.

### Raw-window baselines

```bash
python dataloader.py
python run.py
python run_fedavg_baseline.py
````

Set `SPLIT_MODE = "loso"` in `run.py` for the main centralised LOSO baselines. Use `SPLIT_MODE = "random"` only for the protocol-gap comparison.

### MANTIS-based experiments

```bash
python experiments/extract_mantis_embeddings.py
python runMantisembeddings.py
python experiments/extract_mantis_embeddings_per_subject.py
python run_fedavg_mantis_adapter.py
```

Set `SPLIT_MODE = "loso"` in `runMantisembeddings.py` for centralised LOSO evaluation. The main reported MANTIS variant is:

```text
supcon_lowrank_128_r32
```

For the federated experiments, `run_fedavg_mantis_adapter.py` evaluates the Fed-MANTIS adapter variants under LOSO-FL. The main proposed federated model is also:

```text
supcon_lowrank_128_r32
```


## 10. Reproducibility Notes

Fixed random seeds are used where possible for model initialisation, data splitting, and client simulation. Small differences may still occur due to hardware, CUDA versions, PyTorch versions, and nondeterministic GPU operations.

For exact thesis reproduction, use the same processed PAMAP2 arrays, cached MANTIS embeddings, and script configurations described above.



## 11. MANTIS Citation and License

This project uses the MANTIS time-series foundation model as a frozen embedding extractor. MANTIS is licensed under the Apache License 2.0. Please refer to the original MANTIS repository and its `LICENSE` file for full licensing details.

If you use the MANTIS model or embeddings in your work, please also cite the MANTIS technical report:

```bibtex
@article{feofanov2025mantis,
  title={Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification},
  author={Vasilii Feofanov and Songkang Wen and Marius Alonso and Romain Ilbert and Hongbo Guo and Malik Tiomoko and Lujia Pan and Jianfeng Zhang and Ievgen Redko},
  journal={arXiv preprint arXiv:2502.15637},
  year={2025}
}

