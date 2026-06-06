# BloodMNIST 项目中文方案

## 项目目标

本项目基于 `bloodmnist.npz` 完成 BloodMNIST 8 分类疾病诊断任务。数据集已经划分为训练集、验证集和测试集，图像尺寸为 `28x28x3`，标签共 8 类。

代码目标不是只训练一个模型，而是形成一套完整、可复现、可用于报告和 PPT 展示的实验流程，包括数据分析、模型训练、性能评估、可视化结果和实验对比。

## 实验设计

本项目不加入迁移学习实验，重点用三组 CNN 实验形成递进式对比。设计思路是先建立一个简单 baseline，再加入更强的模型结构和数据增强，最后在主模型上处理类别不均衡问题。这样其他同学可以清楚判断：性能提升到底来自模型结构、数据增强，还是类别不均衡处理。

整体对比关系如下：

```text
simple_cnn_ce
  简单 CNN baseline
  普通交叉熵
  无训练增强

        ↓ 对比模型结构和增强策略是否有效

improved_cnn_ce
  改进 CNN 主模型
  普通交叉熵
  加入训练增强

        ↓ 控制模型结构不变，只改变损失函数

improved_cnn_weighted_ce
  改进 CNN 主模型
  class-weighted cross entropy
  加入训练增强
```

### 实验 1：`simple_cnn_ce`

这个实验是 baseline，目的是回答“只用一个简单 CNN 能达到什么水平”。它不是最终主模型，但可以作为后续改进的参照。

具体实现：

- 代码入口：`python -m src.train --model simple_cnn --loss ce --run-name simple_cnn_ce`
- 模型文件：`src/models.py` 中的 `SimpleCNN`
- 参数量：421,960
- 损失函数：普通 `CrossEntropyLoss`
- 数据增强：关闭，即训练集只做 `ToTensor` 和标准化
- 训练输出目录：`outputs/runs/simple_cnn_ce`

模型结构：

```text
Input: 3 x 28 x 28
Conv2d(3 -> 32, kernel=3, padding=1)
ReLU
MaxPool2d(2)              输出约 32 x 14 x 14
Conv2d(32 -> 64, kernel=3, padding=1)
ReLU
MaxPool2d(2)              输出约 64 x 7 x 7
Flatten
Dropout(0.25)
Linear(64*7*7 -> 128)
ReLU
Dropout(0.25)
Linear(128 -> 8)
```

为什么需要这一组：

- 它提供最基础的 CNN 结果，避免报告只展示一个模型而缺乏对比。
- 如果改进模型提升明显，可以说明更深结构、正则化和增强策略是有价值的。
- 它也能暴露简单模型的不足，例如对某些类别 recall 较低、混淆更多。

### 实验 2：`improved_cnn_ce`

这个实验是主模型的普通交叉熵版本，目的是验证改进 CNN 结构和数据增强是否比 baseline 更强。

具体实现：

- 代码入口：`python -m src.train --model improved_cnn --loss ce --augment --run-name improved_cnn_ce`
- 模型文件：`src/models.py` 中的 `ImprovedCNN`
- 参数量：305,000
- 损失函数：普通 `CrossEntropyLoss`
- 数据增强：开启
- 训练输出目录：`outputs/runs/improved_cnn_ce`

模型结构：

```text
Input: 3 x 28 x 28

ConvBlock 1:
  Conv2d(3 -> 32, kernel=3, padding=1, bias=False)
  BatchNorm2d(32)
  ReLU
  Conv2d(32 -> 32, kernel=3, padding=1, bias=False)
  BatchNorm2d(32)
  ReLU
  MaxPool2d(2)
  Dropout2d(0.05)

ConvBlock 2:
  Conv2d(32 -> 64, kernel=3, padding=1, bias=False)
  BatchNorm2d(64)
  ReLU
  Conv2d(64 -> 64, kernel=3, padding=1, bias=False)
  BatchNorm2d(64)
  ReLU
  MaxPool2d(2)
  Dropout2d(0.10)

ConvBlock 3:
  Conv2d(64 -> 128, kernel=3, padding=1, bias=False)
  BatchNorm2d(128)
  ReLU
  Conv2d(128 -> 128, kernel=3, padding=1, bias=False)
  BatchNorm2d(128)
  ReLU
  Dropout2d(0.15)

AdaptiveAvgPool2d(1 x 1)
Flatten
Dropout(0.35)
Linear(128 -> 128)
ReLU
Dropout(0.25)
Linear(128 -> 8)
```

与 baseline 的主要区别：

- 卷积层更深：从 2 个卷积层增加到 6 个卷积层，特征表达能力更强。
- 加入 BatchNorm：让训练更稳定，通常能加快收敛。
- 加入 Dropout/Dropout2d：减少过拟合。
- 使用 AdaptiveAvgPool：减少对固定空间展平特征的依赖，也减少分类头参数。
- 使用训练增强：提升模型对图像方向、轻微旋转和颜色差异的鲁棒性。

训练增强的具体内容在 `src/dataset.py`：

```text
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.2)
RandomRotation(degrees=12)
ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)
ToTensor
Normalize(mean=train_mean, std=train_std)
```

为什么需要这一组：

- 它验证模型架构改进和增强策略的效果。
- 它是后续类别加权实验的对照组。
- 如果该模型比 `simple_cnn_ce` 明显更好，报告中可以把提升归因于更强 CNN 表达能力和正则化/增强策略。

### 实验 3：`improved_cnn_weighted_ce`

这个实验保持 `ImprovedCNN` 结构和训练增强不变，只把损失函数从普通交叉熵换成 class-weighted cross entropy。目的是专门分析 BloodMNIST 类别分布不均衡时，类别加权是否能提高类别均衡指标。

具体实现：

- 代码入口：`python -m src.train --model improved_cnn --loss weighted_ce --augment --run-name improved_cnn_weighted_ce`
- 模型文件：仍然使用 `src/models.py` 中的 `ImprovedCNN`
- 参数量：305,000
- 损失函数：`CrossEntropyLoss(weight=class_weights)`
- 数据增强：开启，与 `improved_cnn_ce` 完全一致
- 训练输出目录：`outputs/runs/improved_cnn_weighted_ce`

类别权重计算逻辑在 `src/dataset.py` 的 `compute_class_weights`：

```text
class_weight[c] = total_train_samples / (num_classes * train_count[c])
然后再除以所有类别权重的均值，使权重尺度更稳定
```

本次训练集得到的类别权重大致为：

| class_id | class_name | train_count | weight |
| ---: | --- | ---: | ---: |
| 0 | basophil | 852 | 1.4975 |
| 1 | eosinophil | 2181 | 0.5850 |
| 2 | erythroblast | 1085 | 1.1759 |
| 3 | immature_granulocyte | 2026 | 0.6298 |
| 4 | lymphocyte | 849 | 1.5028 |
| 5 | monocyte | 993 | 1.2849 |
| 6 | neutrophil | 2330 | 0.5476 |
| 7 | platelet | 1643 | 0.7766 |

解释：

- 样本少的类别，例如 `basophil`、`lymphocyte`、`monocyte`，权重更高。
- 样本多的类别，例如 `neutrophil`、`eosinophil`，权重更低。
- 这样训练时模型对少数类错误会受到更大惩罚，有利于改善 macro F1 和少数类 recall。

为什么需要这一组：

- 它和 `improved_cnn_ce` 只差损失函数，因此是一个比较干净的控制变量实验。
- 如果 accuracy 不变但 macro F1 提升，可以说明类别加权主要改善了类别均衡表现，而不是单纯提高多数类正确率。
- 这组结果可以直接支撑报告中对“长尾分布”和“类别不均衡”的讨论。

### 三组实验共同训练设置

三组正式实验都通过 `scripts/run_all_experiments.sh` 运行，共用以下设置：

```text
DEVICE=cuda
NUM_WORKERS=4
EPOCHS=50
BATCH_SIZE=128
seed=42
optimizer=AdamW
learning_rate=1e-3
weight_decay=1e-4
early_stopping_patience=8
scheduler=ReduceLROnPlateau(mode=max, factor=0.5, patience=3)
best_checkpoint_metric=validation macro F1
```

训练流程：

1. 读取 `bloodmnist.npz` 中已经划分好的 train/val/test。
2. 只用训练集计算 RGB 三通道均值和标准差。
3. 对 train/val/test 使用相同标准化参数。
4. 每个 epoch 训练一次训练集，再在验证集上评估。
5. 用验证集 `macro_f1` 选择 best checkpoint。
6. 训练结束后载入 best checkpoint。
7. 最后在测试集上评估一次，保存 metrics、per-class 表格、混淆矩阵和错误分类样例。

### 评价指标设计

本项目不只看 accuracy，因为 BloodMNIST 训练集类别分布不均衡。例如训练集中 `neutrophil` 有 2330 张，而 `lymphocyte` 只有 849 张。如果只看 accuracy，模型可能偏向多数类但仍得到较高总体分数。

因此主要使用这些指标：

- `Accuracy`：总体分类正确率，适合给出直观性能。
- `Macro F1`：每个类别权重相同，适合评价类别不均衡场景。
- `Weighted F1`：按各类别样本数量加权，更接近整体数据分布下的平均表现。
- `Per-class precision/recall/F1`：用于分析具体哪些血细胞类别表现好或差。
- `Confusion matrix`：用于观察类别之间的混淆关系。

报告中建议把 `macro F1` 作为类别不均衡讨论的核心指标，把 `accuracy` 作为总体性能指标。

### 输出文件如何对应实验设计

每个正式实验目录都包含：

```text
outputs/runs/<run_name>/run_config.json
outputs/runs/<run_name>/checkpoints/best.pt
outputs/runs/<run_name>/metrics/test_metrics.json
outputs/runs/<run_name>/tables/test_per_class_metrics.csv
outputs/runs/<run_name>/tables/test_confusion_matrix.csv
outputs/runs/<run_name>/figures/training_curves.png
outputs/runs/<run_name>/figures/test_confusion_matrix.png
outputs/runs/<run_name>/figures/test_misclassified_examples.png
```

组员如果想检查某个实验具体怎么跑的，优先看对应目录下的 `run_config.json`。如果想检查最终表现，优先看 `metrics/test_metrics.json` 和 `tables/test_per_class_metrics.csv`。如果想写报告或 PPT，优先使用 `outputs/summary/` 和各 run 的 `figures/`。

## 代码结构

- `src/dataset.py`：读取 `bloodmnist.npz`，完成图像格式转换、归一化、增强、DataLoader 和类别权重计算。
- `src/models.py`：实现 `SimpleCNN` 和 `ImprovedCNN`。
- `src/train.py`：训练入口，支持 GPU/CPU 自动选择、early stopping、best checkpoint 保存和最终测试集评估。
- `src/evaluate.py`：对已有 checkpoint 进行独立评估。
- `src/analyze_data.py`：生成类别分布图、样本图和数据摘要。
- `src/summarize_runs.py`：汇总多组实验结果，生成对比表和对比图。
- `outputs/`：保存正式实验的指标、图表、checkpoint 和报告素材。

## 运行方式

完整实验命令：

```bash
DEVICE=cuda NUM_WORKERS=4 EPOCHS=50 BATCH_SIZE=128 bash scripts/run_all_experiments.sh
```

如果没有可用 GPU，代码会在 `DEVICE=auto` 时自动回退到 CPU：

```bash
DEVICE=auto EPOCHS=50 BATCH_SIZE=128 bash scripts/run_all_experiments.sh
```

快速检查代码是否能跑通：

```bash
bash scripts/smoke_test.sh
```

## 当前正式结果

| 实验 | 模型 | 损失函数 | 最佳 epoch | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `improved_cnn_weighted_ce` | Improved CNN | weighted CE | 33 | 0.9649 | 0.9629 | 0.9650 |
| `improved_cnn_ce` | Improved CNN | CE | 41 | 0.9649 | 0.9614 | 0.9651 |
| `simple_cnn_ce` | Simple CNN | CE | 46 | 0.9456 | 0.9402 | 0.9459 |

结论上，`ImprovedCNN` 明显优于简单 CNN。加权交叉熵的总体准确率与普通交叉熵相同，但 macro F1 更高，因此更适合作为处理类别不均衡问题的主结果。

## 报告和 PPT 可用素材

可以直接引用以下文件：

- `outputs/summary/tables/experiment_comparison.csv`：三组实验对比表。
- `outputs/summary/figures/experiment_comparison.png`：三组实验指标对比图。
- `outputs/dataset/figures/class_distribution.png`：类别分布图。
- `outputs/dataset/figures/sample_grid.png`：各类别样本展示。
- `outputs/runs/*/figures/training_curves.png`：训练曲线。
- `outputs/runs/*/figures/test_confusion_matrix.png`：测试集混淆矩阵。
- `outputs/runs/*/figures/test_misclassified_examples.png`：错误分类样例。
- `outputs/runs/*/tables/test_per_class_metrics.csv`：每一类 precision、recall、F1。

## 结果分析思路

报告中建议重点分析：

- Baseline CNN 到 Improved CNN 的性能提升，说明更深卷积结构、BatchNorm、Dropout 和增强策略有效。
- Accuracy 与 macro F1 的差异，说明只看总体准确率不足以评价长尾类别表现。
- Weighted CE 对 macro F1 的提升，说明类别不均衡处理对类别均衡指标有帮助。
- 混淆矩阵中容易混淆的类别，结合错误分类样例讨论模型局限。

## 模型不足

- 数据只有图像级标签，缺乏像素级或细胞区域精细标注，模型无法显式定位关键病理区域。
- 类别分布不均衡，少数类诊断表现更容易受到影响。
- 图像分辨率只有 `28x28`，细胞形态细节有限。
- CNN 的可解释性有限，临床诊断依据不够透明。
- 数据集规模有限，模型泛化能力仍需要更多外部数据验证。

## 可行改进方向

- 使用 GAN 或 diffusion 生成少数类样本，增强长尾类别。
- 尝试 focal loss、class-balanced loss 或 weighted sampler。
- 使用 Grad-CAM 或 attention 可视化提升可解释性。
- 引入更高分辨率图像、细胞区域标注或像素级标注。
- 在后续扩展中再考虑医学图像预训练模型或迁移学习。
