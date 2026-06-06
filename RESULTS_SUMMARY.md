# BloodMNIST Experiment Results

The full experiment command was:

```bash
DEVICE=cuda NUM_WORKERS=4 EPOCHS=50 BATCH_SIZE=128 bash scripts/run_all_experiments.sh
```

## Main Results

| Run | Model | Loss | Augmentation | Best Epoch | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `improved_cnn_weighted_ce` | Improved CNN | weighted CE | yes | 33 | 0.9649 | 0.9629 | 0.9650 |
| `improved_cnn_ce` | Improved CNN | CE | yes | 41 | 0.9649 | 0.9614 | 0.9651 |
| `simple_cnn_ce` | Simple CNN | CE | no | 46 | 0.9456 | 0.9402 | 0.9459 |

## Takeaways

- The improved CNN clearly outperforms the simple CNN baseline.
- Weighted cross entropy gives the best macro F1, which is the most useful headline metric for class-imbalanced data.
- Plain cross entropy has nearly identical accuracy and slightly higher weighted F1, but weighted CE is better for the long-tail discussion because it improves class-balanced performance.
- Use `outputs/summary/tables/experiment_comparison.csv` and `outputs/summary/figures/experiment_comparison.png` for the report/PPT comparison.

