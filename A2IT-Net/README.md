# A2IT-Net

Official PyTorch implementation of **A2IT-Net (Attribute-Aware Tabular Encoding and Gated Fusion for Image-Tabular Multimodal Diagnosis)**, an end-to-end multimodal framework for joint representation learning from images and structured tabular data.

A2IT-Net preserves fine-grained attribute semantics during tabular encoding and integrates modality-specific and shared information through a two-stage decoupled fusion strategy.

## Architecture

A2IT-Net consists of three main components:

1. **Image Encoder**  
   An ImageNet-pretrained ResNet-50 extracts a global visual representation.

2. **Adaptive Tabular Encoder (AdaTab)**
   - **AttEmb** independently embeds each numerical and categorical attribute.
   - **SemRes** models high-order inter-attribute dependencies while restoring original attribute semantics through adaptive gating.
   - **GloFoc** combines global average aggregation with attribute-aware focal aggregation.

3. **Two-Stage Decoupled Fusion (DecFusion)**
   - **DimCross** captures non-exclusive cross-modal interactions through dimension-wise bilinear gating.
   - **TriGate** adaptively routes visual-specific, tabular-specific, and shared representations through three decoupled paths.

The fused representation is passed to a classification head for end-to-end prediction.

## Installation

Python 3.10 or later is recommended.

```bash
cd A2IT-Net

python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

The reference environment uses PyTorch 2.7.1 and torchvision 0.22.1. Training supports both CPU and CUDA devices, with automatic mixed precision enabled by default on CUDA.

## Data Preparation

Each dataset is described by a JSON configuration in `configs/`. The metadata file may be provided as CSV or Excel and should contain:

- an image path or image identifier;
- the target label;
- numerical and categorical attributes;
- a patient identifier when patient-level splitting is required.

For datasets with an established benchmark protocol, predefined `fold` and `split` columns can be supplied to reproduce the original partitions.

Update the paths and column names in the selected configuration before training:

```json
{
  "data": {
    "metadata_path": "data/dataset/metadata.csv",
    "image_root": "data/dataset/images",
    "image_column": "image_id",
    "patient_id_column": "patient_id",
    "label_column": "label",
    "numerical_columns": ["age"],
    "categorical_columns": ["sex"]
  }
}
```

Ready-to-edit configurations are provided for:

- BRSET-3: `configs/brset3.json`
- BRSET-5: `configs/brset5.json`
- BHBC: `configs/bhbc.json`
- DVM: `configs/dvm.json`

## Training

Train and validate all five folds:

```bash
python train_cv.py \
  --config configs/brset3.json \
  --output-dir outputs/brset3
```

Train a specific fold:

```bash
python train_cv.py \
  --config configs/brset3.json \
  --output-dir outputs/brset3 \
  --fold 0
```

Select a device explicitly:

```bash
python train_cv.py \
  --config configs/brset3.json \
  --output-dir outputs/brset3 \
  --device cuda
```

Use `--no-pretrained` to initialize ResNet-50 without ImageNet weights.

## Default Training Settings

| Setting | Value |
| --- | ---: |
| Image size | 224 x 224 |
| Batch size | 16 |
| Optimizer | AdamW |
| Initial learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Scheduler | Cosine decay |
| Warmup | 10% |
| Dropout | 0.3 |
| Gradient clipping | 1.0 |
| Maximum epochs | 200 |
| Early-stopping patience | 50 |
| Cross-validation folds | 5 |
| Random seed | 42 |

All settings can be changed in the corresponding JSON configuration.

## Outputs

Training artifacts are saved under the specified output directory:

```text
outputs/<experiment>/
|-- config.json
|-- fold_0/
|   |-- best_model.pt
|   |-- history.json
|   `-- preprocessor.json
|-- fold_1/
|-- fold_2/
|-- fold_3/
|-- fold_4/
|-- heldout_test_manifest.csv
`-- validation_summary.json
```

`validation_summary.json` contains the best validation metrics for each fold together with their mean and standard deviation.

## Repository Structure

```text
A2IT-Net/
|-- a2it_net/
|   |-- data/
|   |-- models/
|   |-- training/
|   |-- metrics.py
|   `-- utils.py
|-- configs/
|-- tests/
|-- train_cv.py
|-- pyproject.toml
`-- requirements.txt
```

## Testing

Run the model and data-pipeline tests with:

```bash
python -m unittest discover -s tests -v
```
