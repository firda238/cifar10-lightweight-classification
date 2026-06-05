# 基于 CIFAR-10 的轻量化目标识别方法及实现

本项目实现计算机视觉大作业设计大纲中的核心内容：使用 CIFAR-10 完成图像分类，重点比较基础 CNN 和轻量化 CNN，并提供数据增强对比、模型指标统计、ONNX 导出与 ONNX Runtime 推理测试。

## 目录结构

```text
.
├── data/                    # CIFAR-10 下载目录
├── models/                  # 模型定义
├── checkpoints/             # 训练权重和 ONNX 文件
├── results/                 # 曲线图、混淆矩阵、指标表
├── report/                  # Word 报告
├── train.py                 # 训练入口
├── test.py                  # 测试入口
├── export_onnx.py           # ONNX 导出
├── infer_onnx.py            # ONNX Runtime 推理
├── visualize_samples.py     # 数据集样本可视化
└── summarize_results.py     # 实验对比表可视化
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
| `none` | A | 无数据增强 |
| `rotate` | B | 随机裁剪 + 水平翻转 + 随机旋转 |

## 快速验证

下面命令只训练少量样本，用于确认代码可以跑通。

```bash
python train.py --model lightweight --augment rotate --epochs 1 --limit-train 512 --limit-test 256 --batch-size 64
```

## 完整实验命令

基础 CNN 无增强：

```bash
python train.py --model baseline --augment none --epochs 20 --batch-size 64
```

基础 CNN 使用增强：

```bash
python train.py --model baseline --augment rotate --epochs 20 --batch-size 64
```

轻量化 CNN 无增强：

```bash
python train.py --model lightweight --augment none --epochs 20 --batch-size 64
```

轻量化 CNN 使用增强：

```bash
python train.py --model lightweight --augment rotate --epochs 20 --batch-size 64
```

## 测试命令

测试训练好的轻量化 CNN：

```bash
python test.py --model lightweight --checkpoint checkpoints/lightweight_rotate.pth --batch-size 64
```

## 结果图生成

训练脚本会自动保存：

```text
checkpoints/<model>_<augment>.pth
results/<model>_<augment>/accuracy_curve.png
results/<model>_<augment>/loss_curve.png
results/<model>_<augment>/confusion_matrix.png
results/<model>_<augment>/metrics.json
results/model_compare.csv
```

生成 CIFAR-10 样本图：

```bash
python visualize_samples.py
```

将 `results/model_compare.csv` 渲染为表格图片：

```bash
python summarize_results.py
```

## ONNX 部署模拟

导出轻量化 CNN：

```bash
python export_onnx.py --model lightweight --checkpoint checkpoints/lightweight_rotate.pth --output checkpoints/lightweight_cnn.onnx
```

使用 ONNX Runtime 测试推理时间：

```bash
python infer_onnx.py --onnx checkpoints/lightweight_cnn.onnx
```

也可以指定一张图片：

```bash
python infer_onnx.py --onnx checkpoints/lightweight_cnn.onnx --image path/to/image.png
```

## 报告

报告草稿位于：

```text
report/计算机视觉大作业报告.docx
```

报告中的实验结果表格可在完成训练后，根据 `results/model_compare.csv` 和生成的曲线图、混淆矩阵补充最终数值。
