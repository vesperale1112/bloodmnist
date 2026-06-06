# BloodMNIST 项目中文方案

## 项目目标

本项目基于 `bloodmnist.npz` 完成 BloodMNIST 8 分类疾病诊断任务。数据集已经划分为训练集、验证集和测试集，图像尺寸为 `28x28x3`，标签共 8 类。

代码目标不是只训练一个模型，而是形成一套完整、可复现、可用于报告和 PPT 展示的实验流程，包括数据分析、模型训练、性能评估、可视化结果和实验对比。

## 实验设计

本项目不加入迁移学习实验，重点完成三组 CNN 对比：

1. `simple_cnn_ce`
   - 使用简单 CNN 作为 baseline。
   - 使用普通交叉熵损失。
   - 不使用复杂数据增强。

2. `improved_cnn_ce`
   - 使用更深的 CNN 作为主模型。
   - 加入 BatchNorm、Dropout 和全局池化。
   - 使用随机翻转、轻微旋转、颜色扰动等训练增强。

3. `improved_cnn_weighted_ce`
   - 在主模型基础上使用 class-weighted cross entropy。
   - 重点观察类别不均衡场景下 macro F1 和少数类 recall 的变化。

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

