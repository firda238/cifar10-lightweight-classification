# 基于 CIFAR-10 的轻量化目标识别方法及实现

## 项目简介

本项目对应课程题目“轻量化目标识别方法及实现”。项目使用 CIFAR-10 数据集完成 10 类图像分类，实现基础 CNN 和轻量化 CNN 两种网络结构，对比不同模型在准确率、参数量、模型大小和推理速度上的差异，并补充旋转、翻转、噪声添加等数据增强策略。

部署部分采用“边缘端部署模拟”方案：将 PyTorch 模型导出为 ONNX，并使用 ONNX Runtime 在 CPU 环境下完成推理验证。该流程可迁移到树莓派、普通 CPU 设备或移动端 ONNX Runtime 环境。当前项目不虚假声明已经完成 Android/iOS 真机部署。

本项目选择 CNN 路线，重点关注轻量化结构设计。ViT/MobileViT 可作为后续扩展方向。

## 功能完成情况

| 题目要求 | 本项目实现 | 状态 |
| --- | --- | --- |
| 使用 MNIST/CIFAR-10 | 使用 CIFAR-10 数据集 | 已完成 |
| 实现 CNN 或 Transformer | 实现 baseline CNN 与 lightweight CNN | 已完成 |
| 对比不同网络结构准确率 | 输出 `results/model_compare.csv` 和 `results/model_compare_table.png` | 已完成 |
| 优化数据增强策略 | 支持 `none` / `rotate` / `noise` / `strong` | 已完成 |
| 旋转、翻转、噪声添加 | `rotate` 含旋转和翻转，`noise`/`strong` 含高斯噪声 | 已完成 |
| 参数量/模型大小/推理速度统计 | 每次训练自动写入 `metrics.json` 和对比表 | 已完成 |
| ONNX 导出 | 提供 `export_onnx.py` | 已完成 |
| 移动端或边缘端部署 | ONNX 导出 + ONNX Runtime CPU 推理模拟 | 已完成 |

## 项目结构

```text
.
├── data/                    # CIFAR-10 下载目录，不提交原始数据
├── models/                  # baseline CNN 和 lightweight CNN
├── checkpoints/             # 训练权重和 ONNX 文件，不提交大文件
├── results/                 # 曲线图、混淆矩阵、指标表
├── report/                  # Word 报告
├── scripts/create_report.py # 生成 Word 报告
├── train.py                 # 训练入口
├── test.py                  # 测试入口
├── export_onnx.py           # PyTorch checkpoint 导出 ONNX
├── infer_onnx.py            # ONNX Runtime 推理
├── summarize_results.py     # 实验对比表图片生成
├── visualize_samples.py     # CIFAR-10 样本可视化
├── DEPLOYMENT.md            # 边缘端部署说明
└── requirements.txt         # Python 依赖
```

## 环境安装

建议使用 Python 3.10 或以上版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

脚本默认优先使用 Apple Silicon 的 MPS；如果 MPS 不可用，则自动回退到 CPU。

## 推荐实验参数

| 参数 | 推荐值 |
| --- | --- |
| 数据集 | CIFAR-10 |
| 图像大小 | 32x32 |
| Batch Size | 64 |
| Epoch | 20 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss | CrossEntropyLoss |
| 设备 | MPS 或 CPU |
| 模型数量 | 基础 CNN + 轻量化 CNN |

## 数据增强策略

| 参数值 | 对应策略 | 说明 |
| --- | --- | --- |
| `none` | 无增强 | 基础对照组 |
| `rotate` | 随机裁剪 + 水平翻转 + 随机旋转 | 提升位置、方向和角度变化适应能力 |
| `noise` | 随机裁剪 + 水平翻转 + 高斯噪声 | 模拟图像采集噪声，提高鲁棒性 |
| `strong` | 随机裁剪 + 水平翻转 + 随机旋转 + 颜色扰动 + 高斯噪声 | 更强的数据扰动，用于验证泛化能力 |

噪声添加由 `data_utils.py` 中的 `AddGaussianNoise` transform 实现，位置在 `ToTensor()` 之后、`Normalize()` 之前。

## 模型结构

### Baseline CNN

基础 CNN 使用三组卷积、ReLU 和最大池化模块，最后通过全连接层输出 10 类分类结果。该模型作为对照组，用于观察轻量化结构是否能在较低参数量下保持接近的分类性能。

### Lightweight CNN

轻量化 CNN 使用深度可分离卷积、Batch Normalization、Dropout 和 Global Average Pooling 降低参数量与模型大小。当前实验中，baseline CNN 参数量约 620k，lightweight CNN 参数量约 44k，参数量减少约 92.85%。

## 快速验证

下面命令只训练少量样本，用于确认新增 `strong` 数据增强和训练流程可以跑通。

```bash
python train.py --model lightweight --augment strong --epochs 1 --limit-train 512 --limit-test 256 --batch-size 64
```

如果不希望覆盖正式结果，可以指定临时输出目录：

```bash
python train.py --model lightweight --augment strong --epochs 1 --limit-train 512 --limit-test 256 --batch-size 64 --checkpoint-dir tmp/checkpoints --results-dir tmp/results
```

## 完整实验命令

基础 CNN 无增强：

```bash
python train.py --model baseline --augment none --epochs 20 --batch-size 64
```

基础 CNN 使用旋转/翻转增强：

```bash
python train.py --model baseline --augment rotate --epochs 20 --batch-size 64
```

基础 CNN 使用噪声增强：

```bash
python train.py --model baseline --augment noise --epochs 20 --batch-size 64
```

基础 CNN 使用强增强：

```bash
python train.py --model baseline --augment strong --epochs 20 --batch-size 64
```

轻量化 CNN 无增强：

```bash
python train.py --model lightweight --augment none --epochs 20 --batch-size 64
```

轻量化 CNN 使用旋转/翻转增强：

```bash
python train.py --model lightweight --augment rotate --epochs 20 --batch-size 64
```

轻量化 CNN 使用噪声增强：

```bash
python train.py --model lightweight --augment noise --epochs 20 --batch-size 64
```

轻量化 CNN 使用强增强：

```bash
python train.py --model lightweight --augment strong --epochs 20 --batch-size 64
```

## 测试命令

测试训练好的轻量化 CNN：

```bash
python test.py --model lightweight --checkpoint checkpoints/lightweight_rotate.pth --batch-size 64
```

## 实验结果

当前仓库已经保留 4 组 20 epoch 实验结果：

| 模型 | 数据增强 | 测试准确率 | 参数量 | 模型大小 | 推理时间 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | none | 0.7758 | 620362 | 2.37 MB | 0.299 ms |
| baseline | rotate | 0.7935 | 620362 | 2.37 MB | 0.259 ms |
| lightweight | none | 0.7820 | 44362 | 0.185 MB | 0.938 ms |
| lightweight | rotate | 0.7791 | 44362 | 0.185 MB | 1.177 ms |

训练脚本会自动保存：

```text
checkpoints/<model>_<augment>.pth
results/<model>_<augment>/accuracy_curve.png
results/<model>_<augment>/loss_curve.png
results/<model>_<augment>/confusion_matrix.png
results/<model>_<augment>/metrics.json
results/model_compare.csv
```

将 `results/model_compare.csv` 渲染为表格图片：

```bash
python summarize_results.py
```

生成 CIFAR-10 样本图：

```bash
python visualize_samples.py
```

## ONNX/边缘端部署

本项目将 PyTorch 模型导出为 ONNX。ONNX 模型可在树莓派、普通 CPU 设备或移动端 ONNX Runtime 环境中运行。当前仓库提供 CPU 环境下的 ONNX Runtime 推理脚本，用于模拟边缘端部署。

导出轻量化 CNN：

```bash
python export_onnx.py --model lightweight --checkpoint checkpoints/lightweight_rotate.pth --output checkpoints/lightweight_cnn.onnx
```

使用 ONNX Runtime 测试推理时间：

```bash
python infer_onnx.py --onnx checkpoints/lightweight_cnn.onnx --repeat 20
```

如需真实树莓派部署，可复制 `checkpoints/lightweight_cnn.onnx`、`infer_onnx.py`、`requirements.txt` 和必要的模型预处理代码到树莓派运行。完整步骤见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 报告文件

报告位于：

```text
report/计算机视觉大作业报告.docx
```

重新生成报告：

```bash
python scripts/create_report.py
```

报告脚本会读取 `results/model_compare.csv`，自动写入已有实验结果，并说明 ONNX Runtime 边缘端推理模拟、不足与后续改进方向。
