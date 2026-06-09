# BloodMNIST Experiment Results

The current CNN experiment script can be run with:

```bash
DEVICE=cuda NUM_WORKERS=4 EPOCHS=50 BATCH_SIZE=128 bash scripts/run_all_experiments.sh
```

The formal results now include `simple_cnn_aug_ce` as an augmentation ablation for the Simple CNN baseline.

Newer training/evaluation code also records one-vs-rest AUC and calibration metrics (`ece`, `mce`, `nll`, `brier_score`). Existing checkpoints can be re-evaluated with `bash scripts/evaluate_checkpoints.sh`; ResNet does not need to be retrained just to generate those metrics.

## Main Results

| Run | Model | Loss | Augmentation | Best Epoch | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test ECE |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `resnet18_compare` | ResNet18 | CE | yes | 42 | 0.9702 | 0.9687 | 0.9703 | 0.0198 |
| `improved_cnn_weighted_ce` | Improved CNN | weighted CE | yes | 33 | 0.9652 | 0.9632 | 0.9653 | 0.0052 |
| `improved_cnn_ce` | Improved CNN | CE | yes | 41 | 0.9649 | 0.9614 | 0.9651 | 0.0079 |
| `simple_cnn_aug_ce` | Simple CNN | CE | yes | 27 | 0.9506 | 0.9466 | 0.9508 | 0.0086 |
| `simple_cnn_ce` | Simple CNN | CE | no | 46 | 0.9456 | 0.9402 | 0.9459 | 0.0287 |

## Takeaways

- The added ResNet18 comparison has the strongest overall result in the current repository: 0.9702 test accuracy and 0.9687 test macro F1.
- Adding augmentation to the Simple CNN improves the baseline from 0.9456 to 0.9506 test accuracy and from 0.9402 to 0.9466 test macro F1.
- The improved CNN clearly outperforms the simple CNN baseline.
- Among the two Improved CNN variants, weighted cross entropy gives the better accuracy, macro F1, weighted F1, and ECE in the current re-evaluated table, which is useful for the class-imbalance discussion.
- Use `outputs/summary/tables/experiment_comparison.csv` and `outputs/summary/figures/experiment_comparison.png` for the report/PPT comparison.
- Additional error-example figures are available in `outputs/summary/error_examples/`.
