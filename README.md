# BloodMNIST 血细胞分类项目

本项目基于 `bloodmnist.npz` 完成 BloodMNIST 8 分类任务，目标是构建一套可复现、可分析、可直接支撑课程报告的医学图像分类流程。仓库目前已经包含数据分析、模型训练、测试评估、多实验汇总、混淆矩阵、错误样例和报告/PPT 可用图表。

## 项目做了什么

- 对 BloodMNIST 数据集做了类别分布统计、样本展示和 train mean/std 计算。
- 实现了纯 PyTorch 数据管线，包括标准化、训练增强、class weights 和 DataLoader。
- 实现并训练了 SimpleCNN、ImprovedCNN 和 ResNet18Light。
- 完成了三组 CNN 主线实验和一组额外 ResNet18 对照实验。
- 保存了每组实验的 checkpoint、训练曲线、测试指标、逐类指标、混淆矩阵和错误分类样例。
- 生成了汇总对比表、对比图和高置信度错误样例图，便于写报告和做 PPT。

## 数据与任务

- 数据文件：`bloodmnist.npz`
- 输入图像：`28 x 28 x 3` RGB
- 类别数：8
- 数据划分：train 11,959 张，val 1,712 张，test 3,421 张
- 类别：`basophil`、`eosinophil`、`erythroblast`、`immature_granulocyte`、`lymphocyte`、`monocyte`、`neutrophil`、`platelet`

数据存在类别不均衡，因此项目同时关注 accuracy、macro F1、weighted F1、per-class recall 和 confusion matrix。

## 当前正式实验

| run_name | 模型 | 损失函数 | 训练增强 | 参数量 | Best Epoch | Test Accuracy | Test Macro F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `resnet18_compare` | ResNet18Light | CE | yes | 11,172,936 | 42 | 0.9702 | 0.9687 |
| `improved_cnn_weighted_ce` | ImprovedCNN | weighted CE | yes | 305,000 | 33 | 0.9649 | 0.9629 |
| `improved_cnn_ce` | ImprovedCNN | CE | yes | 305,000 | 41 | 0.9649 | 0.9614 |
| `simple_cnn_ce` | SimpleCNN | CE | no | 421,960 | 46 | 0.9456 | 0.9402 |

简要结论：

- ResNet18Light 是当前仓库中的最佳模型。
- ImprovedCNN 明显优于 SimpleCNN，说明更深结构、BatchNorm、Dropout、global pooling 和训练增强有效。
- 在 ImprovedCNN 内部，weighted CE 与普通 CE 的 accuracy 相同，但 macro F1 更高，更适合类别不均衡讨论。
- SimpleCNN 是必要 baseline，用来支撑模型改进是否有效的对比。

更详细的中文项目说明见 `PROJECT_PLAN_CN.md`。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `src/analyze_data.py` | 数据统计、类别分布图、样本图 |
| `src/dataset.py` | NPZ 读取、标准化、训练增强、class weights、DataLoader |
| `src/models.py` | SimpleCNN、ImprovedCNN、ResNet18Light |
| `src/train.py` | 模型训练、验证、best checkpoint、测试评估 |
| `src/evaluate.py` | 对已有 checkpoint 做独立评估 |
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
- `outputs/summary/error_examples/test_model_error_overview.png`
- `outputs/summary/error_examples/*_test_top_confusion_pairs.png`

## 运行方式

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

如果另一台电脑上的 Python 不叫 `python3`，运行脚本时可以通过 `PYTHON=/path/to/python` 指定解释器。

快速检查流程：

```bash
bash scripts/smoke_test.sh
```

运行默认 CNN 主线和增强消融实验：

```bash
DEVICE=cuda NUM_WORKERS=4 EPOCHS=50 BATCH_SIZE=128 bash scripts/run_all_experiments.sh
```

默认脚本会运行：

1. `simple_cnn_ce`
2. `simple_cnn_aug_ce`
3. `improved_cnn_ce`
4. `improved_cnn_weighted_ce`

其中 `simple_cnn_aug_ce` 只在 SimpleCNN baseline 上打开训练增强，用来把“augmentation 的影响”和“ImprovedCNN 结构本身的影响”分开讨论。

单独运行当前仓库已有的 ResNet18 对照实验：

```bash
python3 -m src.train \
  --model resnet18 \
  --loss ce \
  --augment \
  --lr 5e-4 \
  --weight-decay 1e-3 \
  --patience 10 \
  --run-name resnet18_compare \
  --device auto
```

重新生成实验汇总：

```bash
python3 -m src.summarize_runs --runs-dir outputs/runs --output-dir outputs/summary
```

生成额外错误样例图：

```bash
python3 -m src.error_visuals --device cpu --num-workers 0
```

## 写报告时的主线

建议报告按下面逻辑组织：

1. 用类别分布说明任务存在 class imbalance。
2. 用 SimpleCNN 作为 baseline。
3. 用 ImprovedCNN 说明更深结构、BatchNorm、Dropout 和训练增强带来提升。
4. 用 ImprovedCNN + weighted CE 单独讨论类别不均衡处理。
5. 用 ResNet18Light 展示当前最佳性能，同时讨论参数量和复杂度。
6. 用 confusion matrix 和 error examples 分析 `immature_granulocyte`、`monocyte`、`basophil` 等容易混淆的类别。
