# BloodMNIST Disease Classification

This project trains and evaluates CNN classifiers on `bloodmnist.npz`.

## Structure

- `src/analyze_data.py`: dataset summary, class distribution, sample grid.
- `src/train.py`: model training, validation, checkpointing, final test evaluation.
- `src/evaluate.py`: evaluate a saved checkpoint.
- `src/dataset.py`: NPZ loading, transforms, class weighting, dataloaders.
- `src/models.py`: `simple_cnn`, `improved_cnn`, and `resnet18` model definitions.
- `src/error_visuals.py`: checkpoint-based error example galleries for report figures.
- `outputs/`: generated metrics, checkpoints, figures, and tables.

## Quick Smoke Test

```bash
bash scripts/smoke_test.sh
```

This checks that data loading, training, evaluation, and figure export work.

## Full Experiments

```bash
bash scripts/run_all_experiments.sh
```

Optional environment variables:

```bash
EPOCHS=50 BATCH_SIZE=128 DEVICE=cuda NUM_WORKERS=4 bash scripts/run_all_experiments.sh
```

The three planned experiments are:

1. `simple_cnn_ce`: baseline CNN with cross entropy.
2. `improved_cnn_ce`: deeper CNN with augmentation.
3. `improved_cnn_weighted_ce`: improved CNN with class-weighted cross entropy.

The repository also includes an additional completed ResNet18 comparison run at `outputs/runs/resnet18_compare`.

## Single Run Examples

```bash
python -m src.train --model improved_cnn --loss ce --augment --run-name improved_cnn_ce
python -m src.train --model improved_cnn --loss weighted_ce --augment --run-name improved_cnn_weighted_ce --device cuda
python -m src.train --model resnet18 --loss ce --augment --lr 5e-4 --run-name resnet18_compare --device cuda
```

Evaluate an existing checkpoint:

```bash
python -m src.evaluate --checkpoint outputs/runs/improved_cnn_weighted_ce/checkpoints/best.pt --split test
```

## Outputs for Report and PPT

Each run directory contains:

- `run_config.json`
- `checkpoints/best.pt`
- `tables/history.csv`
- `tables/test_per_class_metrics.csv`
- `metrics/test_metrics.json`
- `figures/training_curves.png`
- `figures/class_distribution.png`
- `figures/test_confusion_matrix.png`
- `figures/test_misclassified_examples.png`

Use accuracy for overall performance, macro F1 for long-tail robustness, per-class recall for diagnostic sensitivity, and confusion matrices/error examples for discussion.

Generate additional report-ready error example figures from saved checkpoints:

```bash
python -m src.error_visuals --device cpu --num-workers 0
```

This writes high-confidence error grids and confusion-pair galleries under `outputs/summary/error_examples/`.
