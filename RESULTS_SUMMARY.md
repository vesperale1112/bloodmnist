# BloodMNIST Experiment Results

The current CNN experiment script can be run with:

```bash
DEVICE=cuda NUM_WORKERS=4 EPOCHS=50 BATCH_SIZE=128 bash scripts/run_all_experiments.sh
```

The scripts now include `simple_cnn_aug_ce` as an augmentation ablation for the baseline. Run `bash scripts/run_simple_cnn_aug.sh` before adding that row to the formal results table.

Newer training/evaluation code also records one-vs-rest AUC and calibration metrics (`ece`, `mce`, `nll`, `brier_score`). Existing checkpoints can be re-evaluated with `bash scripts/evaluate_checkpoints.sh`; ResNet does not need to be retrained just to generate those metrics.

## Main Results

| Run | Model | Loss | Augmentation | Best Epoch | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `resnet18_compare` | ResNet18 | CE | yes | 42 | 0.9702 | 0.9687 | 0.9703 |
| `improved_cnn_weighted_ce` | Improved CNN | weighted CE | yes | 33 | 0.9649 | 0.9629 | 0.9650 |
| `improved_cnn_ce` | Improved CNN | CE | yes | 41 | 0.9649 | 0.9614 | 0.9651 |
| `simple_cnn_ce` | Simple CNN | CE | no | 46 | 0.9456 | 0.9402 | 0.9459 |

## Takeaways

- The added ResNet18 comparison has the strongest overall result in the current repository: 0.9702 test accuracy and 0.9687 test macro F1.
- The improved CNN clearly outperforms the simple CNN baseline.
- Among the two Improved CNN variants, weighted cross entropy gives the better macro F1, which is useful for the class-imbalance discussion.
- Plain cross entropy has nearly identical accuracy and slightly higher weighted F1, but weighted CE remains better for the long-tail discussion because it improves class-balanced performance within the Improved CNN setup.
- Use `outputs/summary/tables/experiment_comparison.csv` and `outputs/summary/figures/experiment_comparison.png` for the report/PPT comparison.
- Additional error-example figures are available in `outputs/summary/error_examples/`.
