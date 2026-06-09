# BloodMNIST 项目说明（Project Description CN）

## 1. 项目一句话概述

本仓库围绕 `bloodmnist.npz` 完成 BloodMNIST 血细胞图像 8 分类任务。当前项目不只是训练一个模型，而是搭建了一套从数据分析、训练、评估、实验对比到报告素材导出的完整流程，并已经保存了五组正式实验结果：

- `simple_cnn_ce`：简单 CNN baseline。
- `simple_cnn_aug_ce`：简单 CNN + 训练增强，用来做 baseline augmentation 消融。
- `improved_cnn_ce`：改进 CNN，加训练增强，使用普通交叉熵。
- `improved_cnn_weighted_ce`：改进 CNN，加训练增强，使用类别加权交叉熵。
- `resnet18_compare`：额外加入的 ResNet18 对照实验，是当前仓库中效果最好的模型。

文档目标是让组员快速看清楚：数据是什么、代码做了什么、每个实验为什么设计、当前结果如何、报告中可以引用哪些素材。

## 2. 数据集与任务

### 2.1 任务定义

- 数据文件：`bloodmnist.npz`
- 图像格式：RGB 图像，尺寸为 `28 x 28 x 3`
- 分类类别数：8 类
- 数据划分：官方数据中已经提供 train、val、test 三个 split，本项目直接使用这些划分，不重新随机切分。
- 标签顺序：使用 MedMNIST 官方 BloodMNIST 标签顺序。

8 个类别分别是：

| class_id | class_name |
| ---: | --- |
| 0 | basophil |
| 1 | eosinophil |
| 2 | erythroblast |
| 3 | immature_granulocyte |
| 4 | lymphocyte |
| 5 | monocyte |
| 6 | neutrophil |
| 7 | platelet |

### 2.2 数据分布

当前仓库已经通过 `src/analyze_data.py` 生成了数据集摘要，输出在 `outputs/dataset/`。

| class_id | class_name | train | val | test | weighted CE 权重 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | basophil | 852 | 122 | 244 | 1.4975 |
| 1 | eosinophil | 2181 | 312 | 624 | 0.5850 |
| 2 | erythroblast | 1085 | 155 | 311 | 1.1759 |
| 3 | immature_granulocyte | 2026 | 290 | 579 | 0.6298 |
| 4 | lymphocyte | 849 | 122 | 243 | 1.5028 |
| 5 | monocyte | 993 | 143 | 284 | 1.2849 |
| 6 | neutrophil | 2330 | 333 | 666 | 0.5476 |
| 7 | platelet | 1643 | 235 | 470 | 0.7766 |

整体样本数：

- train：11,959
- val：1,712
- test：3,421

可以看到训练集类别分布不均衡，例如 `neutrophil` 有 2330 张，而 `lymphocyte` 只有 849 张，`basophil` 只有 852 张。因此项目不能只看 accuracy，还需要重点看 macro F1 和 per-class recall。

### 2.3 数据预处理与增强

数据管线在 `src/dataset.py` 中实现，主要做了这些事：

1. 从 `bloodmnist.npz` 中读取 `train_images`、`val_images`、`test_images` 和对应标签。
2. 将图像从 NHWC 转成 PyTorch 使用的 CHW。
3. 将像素值从 `[0, 255]` 缩放到 `[0, 1]`。
4. 只用训练集计算 RGB 三通道均值和标准差。
5. train、val、test 都使用同一组训练集均值和标准差做标准化。
6. 训练增强只在 train split 上启用，val/test 不做随机增强。

当前代码中的训练增强是纯 PyTorch 实现，避免依赖 torchvision transforms：

```text
水平翻转：概率 0.5
垂直翻转：概率 0.2
标准化：使用 train mean/std
```

注意：当前代码没有使用旋转、ColorJitter 或随机裁剪。报告中如果描述 augmentation，应按上面这版当前实现来写。

类别权重通过 `compute_class_weights` 计算：

```text
class_weight[c] = total_train_samples / (num_classes * train_count[c])
然后除以所有类别权重的均值，使平均权重约为 1
```

这组权重只在 `improved_cnn_weighted_ce` 的损失函数中真正使用。其他实验虽然在 `run_config.json` 中也记录了权重元数据，但训练损失仍是普通交叉熵。

## 3. 当前仓库做了哪些工作

### 3.1 数据分析

已完成内容：

- 统计 train/val/test 中每个类别的样本数。
- 计算训练集 RGB mean/std。
- 生成类别分布图。
- 生成按类别排列的样本展示图。

对应文件：

- `src/analyze_data.py`
- `outputs/dataset/dataset_summary.json`
- `outputs/dataset/tables/class_distribution.csv`
- `outputs/dataset/figures/class_distribution.png`
- `outputs/dataset/figures/sample_grid.png`

报告用途：

- 用类别分布图说明 BloodMNIST 存在类别不均衡。
- 用样本展示图说明输入图像分辨率较低，但不同细胞形态和颜色仍有可学习特征。
- 用 mean/std 说明标准化参数只从训练集得到，避免使用验证集或测试集信息。

### 3.2 模型实现

模型定义集中在 `src/models.py`，当前正式结果使用了三类模型：

- `SimpleCNN`：小型 baseline CNN。
- `ImprovedCNN`：更深的 CNN，加入 BatchNorm、Dropout2d、Dropout 和 global average pooling。
- `ResNet18Light`：纯 PyTorch 轻量 ResNet18，不依赖 torchvision 的模型实现，针对 `28 x 28` 小图像做了入口层调整。

此外，`src/models.py` 中还保留了 `ImprovedCNN_SE` 的类定义，但当前训练入口 `src/train.py` 的正式 choices 中没有开放这个模型，当前输出结果也没有包含这组实验。因此报告主线不需要把它作为正式实验写入。

### 3.3 训练与评估流程

训练入口是 `src/train.py`。它完成的流程如下：

1. 固定随机种子 `seed=42`。
2. 根据参数构建 DataLoader。
3. 根据 `--model` 构建模型。
4. 根据 `--loss` 选择普通 CE 或 weighted CE。
5. 使用 AdamW 优化器。
6. 使用 `ReduceLROnPlateau`，监控验证集 macro F1。
7. 每个 epoch 保存训练和验证指标。
8. 当验证集 macro F1 刷新时保存 best checkpoint。
9. 训练结束后载入 best checkpoint。
10. 在 test set 上做最终评估。
11. 保存 metrics、per-class 表格、AUC、ECE/calibration、混淆矩阵、训练曲线和错误分类样例。

关键设计点：

- best checkpoint 的选择标准是 validation macro F1，而不是 validation accuracy。
- 这样做是因为数据存在类别不均衡，macro F1 更能反映少数类表现。
- 重新训练或重新评估后，metrics 中会包含 one-vs-rest macro/weighted AUC、ECE、MCE、NLL 和 Brier score。
- 训练时如果使用 CUDA，会启用 PyTorch AMP autocast 和 GradScaler。
- `--weighted-sampler` 也在代码中支持，但当前五组正式实验都没有使用 weighted sampler。

### 3.4 结果汇总与错误分析

结果汇总脚本是 `src/summarize_runs.py`。它会扫描 `outputs/runs/` 中的正式 run，跳过 `smoke_` 和 `check_` 开头的临时 run，然后生成：

- `outputs/summary/tables/experiment_comparison.csv`
- `outputs/summary/figures/experiment_comparison.png`

错误样例脚本是 `src/error_visuals.py`。它会读取各模型 checkpoint，在测试集上生成高置信度错误样例和主要混淆类别样例：

- `outputs/summary/error_examples/test_model_error_overview.png`
- `outputs/summary/error_examples/*_test_high_confidence_errors.png`
- `outputs/summary/error_examples/*_test_top_confusion_pairs.png`

这些图适合放在报告的 error analysis 或 limitations 部分。

## 4. 五组实验设计

### 4.1 实验 1：`simple_cnn_ce`

目的：建立最基础的 CNN baseline，回答“只用一个简单 CNN 能做到什么水平”。

运行命令：

```bash
python -m src.train --model simple_cnn --loss ce --run-name simple_cnn_ce
```

配置摘要：

- 模型：`SimpleCNN`
- 损失函数：普通 `CrossEntropyLoss`
- 训练增强：关闭
- 参数量：421,960
- batch size：128
- epochs：50
- optimizer：AdamW
- learning rate：1e-3
- weight decay：1e-4
- early stopping patience：8
- best checkpoint 指标：validation macro F1
- 输出目录：`outputs/runs/simple_cnn_ce`

模型结构：

```text
Input: 3 x 28 x 28
Conv2d(3 -> 32, kernel=3, padding=1)
ReLU
MaxPool2d(2)
Conv2d(32 -> 64, kernel=3, padding=1)
ReLU
MaxPool2d(2)
Flatten
Dropout(0.25)
Linear(64*7*7 -> 128)
ReLU
Dropout(0.25)
Linear(128 -> 8)
```

报告中怎么解释：

- 这是 baseline，不是最终主模型。
- 它提供后续模型改进的参照。
- 它的参数量比 ImprovedCNN 更大，但性能更低，说明参数量本身不是关键，更合理的结构、归一化、正则化和增强更重要。

### 4.2 实验 2：`simple_cnn_aug_ce`

目的：在 SimpleCNN 结构和普通 CE 不变的前提下，只打开训练增强，单独验证 augmentation 对 baseline 的影响。

运行命令：

```bash
python -m src.train --model simple_cnn --loss ce --augment --run-name simple_cnn_aug_ce
```

配置摘要：

- 模型：`SimpleCNN`
- 损失函数：普通 `CrossEntropyLoss`
- 训练增强：开启水平/垂直翻转
- 参数量：421,960
- batch size：128
- epochs：50
- optimizer：AdamW
- learning rate：1e-3
- weight decay：1e-4
- early stopping patience：8
- best checkpoint 指标：validation macro F1
- 输出目录：`outputs/runs/simple_cnn_aug_ce`

设计上的控制变量：

```text
simple_cnn_ce:
  SimpleCNN + ordinary CE + no augmentation

simple_cnn_aug_ce:
  SimpleCNN + ordinary CE + augmentation
```

报告中怎么解释：

- 这组实验不是更换模型，而是只验证训练增强是否能改善 baseline。
- 它相对 `simple_cnn_ce` 的 test accuracy 从 0.9456 提升到 0.9506，macro F1 从 0.9402 提升到 0.9466。
- 它的 ECE 从 0.0287 降到 0.0086，说明增强后的 baseline 置信度校准也更好。
- 它仍低于 ImprovedCNN，说明单靠 augmentation 有帮助，但更合理的网络结构仍然重要。

### 4.3 实验 3：`improved_cnn_ce`

目的：验证更深 CNN 结构和训练增强是否能比 baseline 更好。

运行命令：

```bash
python -m src.train --model improved_cnn --loss ce --augment --run-name improved_cnn_ce
```

配置摘要：

- 模型：`ImprovedCNN`
- 损失函数：普通 `CrossEntropyLoss`
- 训练增强：开启水平/垂直翻转
- 参数量：305,000
- batch size：128
- epochs：50
- optimizer：AdamW
- learning rate：1e-3
- weight decay：1e-4
- early stopping patience：8
- best checkpoint 指标：validation macro F1
- 输出目录：`outputs/runs/improved_cnn_ce`

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

相对 baseline 的改进：

- 卷积层从 2 层增加到 6 层，特征表达能力更强。
- 每个卷积块加入 BatchNorm，提高训练稳定性。
- 使用 Dropout2d 和 Dropout，减少过拟合。
- 使用 AdaptiveAvgPool2d，让分类头更轻量。
- 训练时加入随机翻转增强，提高对图像方向变化的鲁棒性。

报告中怎么解释：

- 它是项目中的主力 CNN 模型。
- 它和 `simple_cnn_ce` 的对比可以说明模型结构与增强策略带来的提升。
- 它和 `improved_cnn_weighted_ce` 的对比可以单独分析类别加权损失的影响。

### 4.4 实验 4：`improved_cnn_weighted_ce`

目的：在 ImprovedCNN 结构不变、训练增强不变的前提下，只改变损失函数，专门分析类别不均衡处理是否有帮助。

运行命令：

```bash
python -m src.train --model improved_cnn --loss weighted_ce --augment --run-name improved_cnn_weighted_ce
```

配置摘要：

- 模型：`ImprovedCNN`
- 损失函数：`CrossEntropyLoss(weight=class_weights)`
- 训练增强：开启水平/垂直翻转
- 参数量：305,000
- batch size：128
- epochs：50
- optimizer：AdamW
- learning rate：1e-3
- weight decay：1e-4
- early stopping patience：8
- best checkpoint 指标：validation macro F1
- 输出目录：`outputs/runs/improved_cnn_weighted_ce`

设计上的控制变量：

```text
improved_cnn_ce:
  ImprovedCNN + augmentation + ordinary CE

improved_cnn_weighted_ce:
  ImprovedCNN + augmentation + weighted CE
```

两组只改变 loss，因此适合在报告里分析 class imbalance 的影响。

报告中怎么解释：

- 少数类如 `basophil`、`lymphocyte`、`monocyte` 的权重更高。
- 多数类如 `neutrophil`、`eosinophil` 的权重更低。
- 如果 accuracy 接近但 macro F1 提升，可以说明类别加权主要改善了类别均衡表现，而不是只提高多数类正确率。

### 4.5 实验 5：`resnet18_compare`

目的：在手写 CNN 之外，加入一个更强的残差网络对照，判断当前任务是否还能从更深网络结构中获益。

当前仓库保存的运行配置对应命令可概括为：

```bash
python -m src.train \
  --model resnet18 \
  --loss ce \
  --augment \
  --lr 5e-4 \
  --weight-decay 1e-3 \
  --patience 10 \
  --run-name resnet18_compare
```

配置摘要：

- 模型：`ResNet18Light`
- 预训练：关闭，`pretrained=false`
- 损失函数：普通 `CrossEntropyLoss`
- 训练增强：开启水平/垂直翻转
- 参数量：11,172,936
- batch size：128
- epochs：50
- learning rate：5e-4
- weight decay：1e-3
- early stopping patience：10
- best checkpoint 指标：validation macro F1
- 输出目录：`outputs/runs/resnet18_compare`

ResNet18Light 的结构特点：

- 使用纯 PyTorch 手写实现，不依赖 torchvision 的 ResNet 模型。
- 入口层针对 `28 x 28` 小图像调整为 `3 x 3` 卷积。
- 去掉标准 ResNet 中对小图像过于激进的 `7 x 7` conv 和初始 maxpool。
- 仍保留 ResNet18 的核心思想：残差连接和多 stage 特征提取。
- stage 通道数为 64、128、256、512，每个 stage 2 个 BasicBlock。
- 最后使用 AdaptiveAvgPool2d 和 Linear 分类到 8 类。

报告中怎么解释：

- ResNet18 是额外加入的强对照，不是 CNN baseline/augmentation/ImprovedCNN 递进消融的一部分。
- 它当前取得了最高 test accuracy 和 test macro F1，可以作为最终性能上限或最佳模型展示。
- 它的参数量明显大于 ImprovedCNN，因此报告需要同时讨论性能提升和模型复杂度提升。
- 由于本次是从零训练，不是 ImageNet 迁移学习，所以不要把结果解释成 pretrained transfer learning 的效果。

## 5. 训练设置与当前正式结果

### 5.1 训练设置对比

| run_name | model | loss | augment | 参数量 | lr | weight_decay | patience | best epoch |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `resnet18_compare` | ResNet18Light | CE | yes | 11,172,936 | 5e-4 | 1e-3 | 10 | 42 |
| `improved_cnn_weighted_ce` | ImprovedCNN | weighted CE | yes | 305,000 | 1e-3 | 1e-4 | 8 | 33 |
| `improved_cnn_ce` | ImprovedCNN | CE | yes | 305,000 | 1e-3 | 1e-4 | 8 | 41 |
| `simple_cnn_aug_ce` | SimpleCNN | CE | yes | 421,960 | 1e-3 | 1e-4 | 8 | 27 |
| `simple_cnn_ce` | SimpleCNN | CE | no | 421,960 | 1e-3 | 1e-4 | 8 | 46 |

补充说明：

- 四个 CNN run 的保存配置记录为 `device=cuda`，硬件为 NVIDIA GeForce RTX 4070 Laptop GPU。
- `resnet18_compare` 的保存配置记录为 `device=auto`，实际记录设备为 CPU。
- 五组实验都使用 `seed=42`。
- 五组实验都没有使用 weighted sampler。

### 5.2 测试集结果

结果来自 `outputs/summary/tables/experiment_comparison.csv`。

| run_name | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test ECE | 测试集错误数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `resnet18_compare` | 0.9702 | 0.9687 | 0.9703 | 0.0198 | 102 / 3421 |
| `improved_cnn_weighted_ce` | 0.9652 | 0.9632 | 0.9653 | 0.0052 | 119 / 3421 |
| `improved_cnn_ce` | 0.9649 | 0.9614 | 0.9651 | 0.0079 | 120 / 3421 |
| `simple_cnn_aug_ce` | 0.9506 | 0.9466 | 0.9508 | 0.0086 | 169 / 3421 |
| `simple_cnn_ce` | 0.9456 | 0.9402 | 0.9459 | 0.0287 | 186 / 3421 |

主要结论：

1. ResNet18Light 当前表现最好，test accuracy 为 0.9702，test macro F1 为 0.9687。
2. `simple_cnn_aug_ce` 明确优于 `simple_cnn_ce`，说明训练增强对 baseline 有正向作用。
3. ImprovedCNN 明显优于 SimpleCNN + augmentation，说明更深结构、BatchNorm、Dropout 和 global pooling 在增强之外继续带来提升。
4. 在 ImprovedCNN 内部比较时，weighted CE 的 accuracy、macro F1 和 weighted F1 都略高于普通 CE。
5. weighted CE 的 macro F1 为 0.9632，对普通 CE 的 0.9614，说明它对类别均衡指标有小幅帮助。
6. 如果报告要强调 class imbalance，推荐把 `improved_cnn_weighted_ce` 作为类别不均衡处理的主结果；如果报告要展示最高性能，推荐把 `resnet18_compare` 作为最佳模型。

### 5.3 逐类表现要点

ResNet18Light 的测试集逐类 F1：

| class_name | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| basophil | 0.9623 | 0.9426 | 0.9524 |
| eosinophil | 0.9968 | 0.9920 | 0.9944 |
| erythroblast | 1.0000 | 0.9743 | 0.9870 |
| immature_granulocyte | 0.9227 | 0.9275 | 0.9251 |
| lymphocyte | 0.9522 | 0.9835 | 0.9676 |
| monocyte | 0.9349 | 0.9613 | 0.9479 |
| neutrophil | 0.9774 | 0.9730 | 0.9752 |
| platelet | 1.0000 | 1.0000 | 1.0000 |

可以重点写的观察：

- `platelet` 在五组实验中都非常容易分类，ResNet18 和 SimpleCNN 都达到 F1=1.0000。
- `eosinophil` 和 `neutrophil` 整体表现也较强，可能与样本数较多、形态特征较明显有关。
- `immature_granulocyte` 是比较困难的类别，ResNet18 的 F1 也只有 0.9251，是 ResNet18 中最低的类别 F1。
- `monocyte` 和 `basophil` 在 baseline 中表现较弱，改进模型和 ResNet18 后有明显提升。
- weighted CE 对 `basophil` recall 提升明显：`improved_cnn_weighted_ce` 为 0.9631，而 `improved_cnn_ce` 为 0.9344。

### 5.4 主要混淆关系

从测试集混淆矩阵和错误样例看，错误主要集中在以下方向：

- `immature_granulocyte` 与 `monocyte` 之间容易混淆。
- `immature_granulocyte` 与 `neutrophil` 之间容易混淆。
- `basophil` 被错分成 `immature_granulocyte` 的情况在多个模型中出现。
- SimpleCNN 中 `monocyte -> immature_granulocyte` 和 `neutrophil -> immature_granulocyte` 的错误更多，说明 baseline 对相近形态类别区分能力不足。

各模型 top confusion pairs 可在下面这些图中查看：

- `outputs/summary/error_examples/resnet18_compare_test_top_confusion_pairs.png`
- `outputs/summary/error_examples/improved_cnn_weighted_ce_test_top_confusion_pairs.png`
- `outputs/summary/error_examples/improved_cnn_ce_test_top_confusion_pairs.png`
- `outputs/summary/error_examples/simple_cnn_aug_ce_test_top_confusion_pairs.png`
- `outputs/summary/error_examples/simple_cnn_ce_test_top_confusion_pairs.png`

## 6. 代码结构与文件职责

| 路径 | 作用 |
| --- | --- |
| `bloodmnist.npz` | BloodMNIST 数据文件 |
| `requirements.txt` | Python 依赖 |
| `scripts/run_all_experiments.sh` | 默认完整 CNN 实验脚本，跑四组 CNN 主线/消融实验并生成 summary |
| `scripts/smoke_test.sh` | 快速 smoke test，用少量 batch 检查流程是否能跑通 |
| `src/analyze_data.py` | 数据集统计、类别分布图、样本图 |
| `src/dataset.py` | 数据读取、标准化、增强、DataLoader、class weights |
| `src/models.py` | SimpleCNN、ImprovedCNN、ResNet18Light 等模型定义 |
| `src/train.py` | 训练入口，保存 checkpoint、metrics、表格和图 |
| `src/evaluate.py` | 对已有 checkpoint 做独立评估 |
| `src/metrics.py` | accuracy、macro F1、weighted F1、AUC、ECE/calibration、per-class metrics 和 confusion matrix |
| `src/visualize.py` | 训练曲线、混淆矩阵、样本图、错误样例图 |
| `src/summarize_runs.py` | 汇总多组 run，生成实验对比表和对比图 |
| `src/error_visuals.py` | 生成高置信度错误样例和 top confusion pairs 图 |
| `outputs/dataset/` | 数据分析输出 |
| `outputs/runs/` | 每个实验的 checkpoint、metrics、tables、figures |
| `outputs/summary/` | 多实验汇总结果和报告素材 |

## 7. 报告写作建议

### 7.1 推荐报告结构

1. 任务介绍：BloodMNIST 血细胞 8 分类，输入为 `28 x 28` RGB 图像。
2. 数据分析：展示类别分布和样本图，说明类别不均衡。
3. 方法设计：介绍数据预处理、标准化、训练增强、评价指标。
4. 模型设计：按 SimpleCNN、SimpleCNN + augmentation、ImprovedCNN、weighted CE、ResNet18Light 的顺序讲清楚递进关系。
5. 实验设置：写 batch size、epochs、optimizer、learning rate、early stopping、best checkpoint 选择标准。
6. 结果对比：引用 `experiment_comparison.csv` 和 `experiment_comparison.png`。
7. 逐类分析：重点讨论 macro F1、少数类 recall、`immature_granulocyte` 等难分类类别。
8. 错误分析：引用 confusion matrix 和 top confusion pairs。
9. 局限性与未来工作：讨论低分辨率、类别不均衡、可解释性和泛化能力。

### 7.2 哪些结果适合放进正文

建议正文必须放：

- `outputs/dataset/figures/class_distribution.png`
- `outputs/dataset/figures/sample_grid.png`
- `outputs/summary/figures/experiment_comparison.png`
- `outputs/runs/resnet18_compare/figures/test_confusion_matrix.png`
- `outputs/runs/improved_cnn_weighted_ce/figures/test_confusion_matrix.png`

建议正文或附录放：

- `outputs/runs/*/figures/training_curves.png`
- `outputs/runs/*/figures/test_misclassified_examples.png`
- `outputs/runs/*/figures/test_reliability_diagram.png`
- `outputs/runs/*/tables/test_calibration_bins.csv`
- `outputs/runs/*/metrics/test_metrics.json` 中的 `macro_auc_ovr`、`weighted_auc_ovr`、`ece`、`mce`、`nll`、`brier_score`
- `outputs/summary/error_examples/test_model_error_overview.png`
- `outputs/summary/error_examples/*_test_high_confidence_errors.png`
- `outputs/summary/error_examples/*_test_top_confusion_pairs.png`

AUC 适合补充医学分类任务中的排序/区分能力分析，ECE 和 reliability diagram 适合讨论模型置信度是否校准。

### 7.3 结果解读主线

可以按下面逻辑写：

1. SimpleCNN 已经能达到 0.9456 accuracy，说明 BloodMNIST 的图像信息足以支持 CNN 学习。
2. SimpleCNN + augmentation 提升到 0.9506 accuracy 和 0.9466 macro F1，说明训练增强单独就能改善 baseline。
3. ImprovedCNN 提升到 0.9649 accuracy，说明更深卷积结构、BatchNorm、Dropout 和 global pooling 在增强之外继续有效。
4. weighted CE 在 ImprovedCNN 上进一步提升到 0.9652 accuracy 和 0.9632 macro F1，说明它更适合支持类别不均衡讨论。
5. ResNet18Light 进一步提升到 0.9702 accuracy 和 0.9687 macro F1，说明残差结构对该任务仍有收益。
6. 但 ResNet18Light 参数量约 1117 万，远高于 ImprovedCNN 的 30.5 万，因此需要在性能和模型复杂度之间做讨论。

## 8. 如何运行

### 8.1 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 8.2 快速检查

```bash
bash scripts/smoke_test.sh
```

这个脚本会生成 `outputs/smoke_dataset/` 和一个 `smoke_simple_cnn` run，用来检查数据读取、训练、评估和图像导出是否正常。

### 8.3 按任务运行脚本

推荐先按下面顺序跑：

```bash
# 数据统计和样本图
bash scripts/analyze_data.sh

# 新增 SimpleCNN + augmentation 消融实验
DEVICE=cuda NUM_WORKERS=4 EPOCHS=50 BATCH_SIZE=128 bash scripts/run_simple_cnn_aug.sh

# 给已有 checkpoint 补 AUC/ECE/calibration 输出
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

### 8.4 一键运行 CNN 实验

```bash
DEVICE=cuda NUM_WORKERS=4 EPOCHS=50 BATCH_SIZE=128 bash scripts/run_all_experiments.sh
```

如果没有 GPU，可以使用：

```bash
DEVICE=auto NUM_WORKERS=0 EPOCHS=50 BATCH_SIZE=128 bash scripts/run_all_experiments.sh
```

当前 `scripts/run_all_experiments.sh` 默认运行：

1. `simple_cnn_ce`
2. `simple_cnn_aug_ce`
3. `improved_cnn_ce`
4. `improved_cnn_weighted_ce`

其中 `simple_cnn_aug_ce` 是新增的增强消融实验，用来单独验证训练增强对 SimpleCNN baseline 的影响。注意：该脚本仍不自动跑 `resnet18_compare`。

### 8.5 ResNet18 怎么处理

当前仓库已经有 `resnet18_compare` 的 checkpoint。如果只是为了补 AUC/ECE/calibration，不需要重训 ResNet，运行 `bash scripts/evaluate_checkpoints.sh` 即可。

只有想刷新 ResNet18 的训练结果时，才运行：

```bash
bash scripts/run_resnet18.sh
```

### 8.6 重新汇总结果

```bash
bash scripts/summarize_results.sh
```

### 8.7 生成错误样例图

```bash
bash scripts/generate_error_visuals.sh
```

### 8.8 重新评估已有 checkpoint 以生成 AUC/ECE

如果已有 checkpoint 是旧代码训练出来的，可以不用重训，直接运行：

```bash
bash scripts/evaluate_checkpoints.sh
```

这会在每个已有 run 目录下生成 `eval_test/`，其中包含新的 AUC、ECE、calibration bins 和 reliability diagram。

## 9. 局限性与后续改进

当前项目的局限性：

- 输入图像只有 `28 x 28`，很多细胞形态细节会丢失。
- 数据只有图像级类别标签，没有细胞区域标注或像素级病理标注。
- 类别不均衡仍然存在，weighted CE 只带来小幅 macro F1 提升。
- 模型可解释性有限，目前主要通过混淆矩阵和错误样例做分析。
- 当前 ResNet18 虽然表现最好，但参数量明显更大，需要考虑复杂度和训练成本。

可行改进方向：

- 尝试 focal loss、class-balanced loss 或 weighted sampler。
- 针对少数类做更系统的数据增强。
- 加入 Grad-CAM 等可解释性方法。
- 尝试更轻量的 residual CNN，寻找性能和参数量的平衡点。
- 如果课程允许，可以尝试医学图像预训练或迁移学习，但需要和当前从零训练结果区分清楚。
