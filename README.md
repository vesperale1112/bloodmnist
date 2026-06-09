# BloodMNIST 血细胞分类项目

本项目基于 `bloodmnist.npz` 完成 BloodMNIST 8 分类任务，目标是构建一套可复现、可分析、可直接支撑课程报告的医学图像分类流程。仓库目前已经包含数据分析、模型训练、测试评估、多实验汇总、混淆矩阵、错误样例和报告/PPT 可用图表。

## 项目做了什么

- 对 BloodMNIST 数据集做了类别分布统计、样本展示和 train mean/std 计算。
- 实现了纯 PyTorch 数据管线，包括标准化、训练增强、class weights 和 DataLoader。
- 实现并训练了 SimpleCNN、ImprovedCNN 和 ResNet18Light。
- 完成了四组 CNN 主线/消融实验和一组额外 ResNet18 对照实验。
- 保存了每组实验的 checkpoint、训练曲线、测试指标、逐类指标、AUC、ECE/calibration、混淆矩阵和错误分类样例。
- 生成了汇总对比表、对比图和高置信度错误样例图，便于写报告和做 PPT。

## 数据与任务

- 数据文件：`bloodmnist.npz`
- 输入图像：`28 x 28 x 3` RGB
- 类别数：8
- 数据划分：train 11,959 张，val 1,712 张，test 3,421 张
- 类别：`basophil`、`eosinophil`、`erythroblast`、`immature_granulocyte`、`lymphocyte`、`monocyte`、`neutrophil`、`platelet`

数据存在类别不均衡，因此项目同时关注 accuracy、macro F1、weighted F1、one-vs-rest AUC、ECE、per-class recall 和 confusion matrix。

## 当前正式实验

| run_name | 模型 | 损失函数 | 训练增强 | 参数量 | Best Epoch | Test Accuracy | Test Macro F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `resnet18_compare` | ResNet18Light | CE | yes | 11,172,936 | 42 | 0.9702 | 0.9687 |
| `improved_cnn_weighted_ce` | ImprovedCNN | weighted CE | yes | 305,000 | 33 | 0.9652 | 0.9632 |
| `improved_cnn_ce` | ImprovedCNN | CE | yes | 305,000 | 41 | 0.9649 | 0.9614 |
| `simple_cnn_aug_ce` | SimpleCNN | CE | yes | 421,960 | 27 | 0.9506 | 0.9466 |
| `simple_cnn_ce` | SimpleCNN | CE | no | 421,960 | 46 | 0.9456 | 0.9402 |

简要结论：

- ResNet18Light 是当前仓库中的最佳模型。
- `simple_cnn_aug_ce` 相比 `simple_cnn_ce` 提升了 test accuracy 和 macro F1，说明单独加入训练增强对 baseline 有稳定正向作用。
- ImprovedCNN 相比 `simple_cnn_aug_ce` 进一步提升，说明更深结构、BatchNorm、Dropout 和 global pooling 在训练增强之外继续带来收益。
- 在 ImprovedCNN 内部，weighted CE 的 accuracy 和 macro F1 均略高于普通 CE，更适合类别不均衡讨论。
- SimpleCNN 是必要 baseline，SimpleCNN + augmentation 是增强消融，用来支撑模型改进是否有效的对比。

更详细的中文项目说明见 `PROJECT_PLAN_CN.md`。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `src/analyze_data.py` | 数据统计、类别分布图、样本图 |
| `src/dataset.py` | NPZ 读取、标准化、训练增强、class weights、DataLoader |
| `src/models.py` | SimpleCNN、ImprovedCNN、ResNet18Light |
| `src/train.py` | 模型训练、验证、best checkpoint、测试评估、AUC/ECE 输出 |
| `src/evaluate.py` | 对已有 checkpoint 做独立评估并生成 AUC/ECE |
| `src/summarize_runs.py` | 汇总多组实验，生成对比表和对比图 |
| `src/error_visuals.py` | 生成高置信度错误样例和主要混淆类别图 |
| `outputs/dataset/` | 数据分析输出 |
| `outputs/runs/` | 每组实验的 checkpoint、metrics、tables、figures |
| `outputs/summary/` | 多实验汇总结果和报告素材 |

## 报告素材位置

常用素材：

- `outputs/dataset/figures/class_distribution.png`
- `outputs/dataset/figures/sample_grid.png`
- `outputs/summary/tables/experiment_comparison.csv`
- `outputs/summary/figures/experiment_comparison.png`
- `outputs/runs/*/figures/training_curves.png`
- `outputs/runs/*/figures/test_confusion_matrix.png`
- `outputs/runs/*/figures/test_misclassified_examples.png`
- `outputs/runs/*/tables/test_per_class_metrics.csv`
- `outputs/runs/*/eval_test/test_metrics.json`
- `outputs/runs/*/eval_test/figures/test_reliability_diagram.png`
- `outputs/runs/*/eval_test/tables/test_calibration_bins.csv`
- `outputs/summary/error_examples/test_model_error_overview.png`
- `outputs/summary/error_examples/*_test_top_confusion_pairs.png`

## 运行方式

当前仓库已经保存五组正式结果。下面命令主要用于重新复现、刷新某一组实验，或在已有 checkpoint 基础上重新生成新指标和汇总图表。

安装依赖：

```bash
python -m pip install -r requirements.txt
```

快速检查流程：

```bash
bash scripts/smoke_test.sh
```

推荐按任务运行：

```bash
# 数据统计和样本图
bash scripts/analyze_data.sh

# 如需刷新 SimpleCNN + augmentation 消融
DEVICE=cuda NUM_WORKERS=4 EPOCHS=50 BATCH_SIZE=128 bash scripts/run_simple_cnn_aug.sh

# 基于已有 checkpoint 重新生成 AUC/ECE/calibration 输出
DEVICE=cuda NUM_WORKERS=4 bash scripts/evaluate_checkpoints.sh

# 重新汇总结果表和对比图
bash scripts/summarize_results.sh
```

也可以分别运行这些训练脚本：

- `bash scripts/run_simple_cnn.sh`
- `bash scripts/run_simple_cnn_aug.sh`
- `bash scripts/run_improved_cnn.sh`
- `bash scripts/run_improved_cnn_weighted.sh`
- `bash scripts/run_resnet18.sh`

保留的一键入口：

```bash
DEVICE=cuda NUM_WORKERS=4 EPOCHS=50 BATCH_SIZE=128 bash scripts/run_all_experiments.sh
```

这个入口会运行 `simple_cnn_ce`、`simple_cnn_aug_ce`、`improved_cnn_ce` 和 `improved_cnn_weighted_ce`，但不自动重训 ResNet18。

当前已有 `resnet18_compare` checkpoint；若只需要补 AUC/ECE，直接运行 `scripts/evaluate_checkpoints.sh` 即可。只有想刷新 ResNet 训练结果时才需要运行：

```bash
bash scripts/run_resnet18.sh
```

重新训练或重新评估后，`test_metrics.json` 会包含 `macro_auc_ovr`、`weighted_auc_ovr`、`ece`、`mce`、`nll` 和 `brier_score`。当前推荐以每个 run 的 `eval_test/` 目录作为 AUC/ECE/calibration、reliability diagram 和 calibration bin 表的统一来源。

当前 summary 以 `outputs/runs/*/eval_test/test_metrics.json` 作为优先指标来源；如果某个 run 没有 `eval_test`，才回退到训练阶段写出的 `metrics/test_metrics.json`。

生成额外错误样例图：

```bash
bash scripts/generate_error_visuals.sh
```

## 写报告时的主线

建议报告按下面逻辑组织：

1. 用类别分布说明任务存在 class imbalance。
2. 用 SimpleCNN 作为 baseline。
3. 用 SimpleCNN + augmentation 单独说明训练增强对 baseline 的影响。
4. 用 ImprovedCNN 说明更深结构、BatchNorm、Dropout 和 global pooling 带来进一步提升。
5. 用 ImprovedCNN + weighted CE 单独讨论类别不均衡处理。
6. 用 ResNet18Light 展示当前最佳性能，同时讨论参数量和复杂度。
7. 用 confusion matrix 和 error examples 分析 `immature_granulocyte`、`monocyte`、`basophil` 等容易混淆的类别。
