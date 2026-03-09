# ERM

Empirical Risk Minimization (ERM) benchmarking framework for evaluating pretrained vision models across multiple datasets with in-domain and out-of-domain validation.

## Models

| Model | Config Name |
|-------|-------------|
| ResNet-18 | `resnet18` |
| EfficientNet-B0 | `efficientnet_b0` |
| ConvNeXt Base | `convnext_base` |
| ViT-Base/16 | `vit_base16` |
| Swin Transformer Base | `swin_base` |

All models use ImageNet-pretrained weights with the classification head replaced to match the target dataset's number of classes.

## Datasets

| Dataset | Classes | Config Name |
|---------|---------|-------------|
| BAR (Biased Activity Recognition) | 6 | `bar` |
| CelebA | 2 | `celeba` |
| NICO++ | 60 | `nico_pp` |

Expected data directory structure:

```
data/
├── bar/
│   ├── train/          # *.jpg
│   └── test/           # *.jpg
├── celeba/
│   ├── img_align_celeba/
│   ├── list_attr_celeba.txt
│   └── list_eval_partition.txt
└── nico_pp/
    └── track_1/
        ├── dg_label_id_mapping.json
        └── track_1/train/   # {context}/{class}/*.jpg
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train with a config file
python train.py configs/bar_resnet18.yaml

# Override the seed
python train.py configs/bar_resnet18.yaml --seed 99

# Smoke test (tiny subset, 2 epochs, auto-cleanup)
python train.py configs/bar_resnet18.yaml --test-mode
```

There are 15 pre-built config files in `configs/` — one for each (dataset, model) combination, named `{dataset}_{model}.yaml`.

## Output

Each run produces a directory at `logs/{dataset}_{model}_seed{seed}/` containing:

| File | Description |
|------|-------------|
| `metrics.csv` | Per-epoch train loss, train acc, ID val acc, OOD val acc |
| `test_results.csv` | Test accuracy for best-ID-val and best-OOD-val models |
| `best_id_val.pt` | Checkpoint with best in-domain validation accuracy |
| `best_ood_val.pt` | Checkpoint with best out-of-domain validation accuracy |
| `test_predictions_best_id_val.json` | Per-image test predictions (best ID model) |
| `test_predictions_best_ood_val.json` | Per-image test predictions (best OOD model) |

## Project Structure

```
├── train.py                    # Entry point
├── configs/                    # 15 YAML config files
├── src/
│   ├── data/
│   │   ├── data_factory.py     # Unified data dispatch
│   │   ├── bar_dataloader.py
│   │   ├── celeba_dataloader.py
│   │   └── nico_pp_dataloader.py
│   ├── models/
│   │   └── model_factory.py    # Model creation with registry
│   ├── training/
│   │   └── trainer.py          # Training loop and evaluation
│   └── utils/
│       ├── config.py           # YAML config dataclasses
│       ├── seed.py             # Reproducibility
│       ├── checkpoint.py       # Save/load checkpoints
│       └── logger.py           # CSV + optional W&B logging
└── data/                       # Dataset files (not tracked)
```
