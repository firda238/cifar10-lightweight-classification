# ONNX 与边缘端部署说明

## 部署目标

本项目完成 PyTorch 模型到 ONNX 的导出，并使用 ONNX Runtime 在 CPU 环境下完成推理验证。该流程可作为树莓派、普通 CPU 边缘设备和移动端 ONNX Runtime 环境的部署模拟。

当前仓库不声明已经完成 Android/iOS 真机部署；已完成的是 ONNX 导出与 ONNX Runtime CPU 推理验证，可迁移到树莓派或移动端 ONNX Runtime 环境。

## ONNX 导出命令

使用训练好的轻量化 CNN checkpoint 导出 ONNX：

```bash
python export_onnx.py \
  --model lightweight \
  --checkpoint checkpoints/lightweight_rotate.pth \
  --output checkpoints/lightweight_cnn.onnx
```

导出的 ONNX 文件默认输入为：

```text
input:  [batch, 3, 32, 32]
output: [batch, 10]
```

## ONNX Runtime 推理命令

使用随机输入测试推理速度：

```bash
python infer_onnx.py --onnx checkpoints/lightweight_cnn.onnx --repeat 20
```

使用指定图片测试：

```bash
python infer_onnx.py --onnx checkpoints/lightweight_cnn.onnx --image path/to/image.png --repeat 20
```

`infer_onnx.py` 会执行与训练阶段一致的图像预处理：缩放到 32x32、转 Tensor、按 CIFAR-10 均值和标准差归一化。

## 树莓派部署步骤

1. 在开发机导出 ONNX：

```bash
python export_onnx.py --model lightweight --checkpoint checkpoints/lightweight_rotate.pth --output checkpoints/lightweight_cnn.onnx
```

2. 将以下文件复制到树莓派：

```text
checkpoints/lightweight_cnn.onnx
infer_onnx.py
requirements.txt
```

3. 在树莓派安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy pillow torch torchvision
```

4. 运行推理：

```bash
python infer_onnx.py --onnx checkpoints/lightweight_cnn.onnx --repeat 20
```

5. 如果使用自己的图片：

```bash
python infer_onnx.py --onnx checkpoints/lightweight_cnn.onnx --image sample.png --repeat 20
```

## Android/iOS 可行方案

移动端可以采用 ONNX Runtime Mobile 或平台原生推理框架：

- Android：将 `lightweight_cnn.onnx` 放入 app assets，通过 ONNX Runtime Android API 加载模型，使用 Kotlin/Java 完成图片预处理和推理。
- iOS：将 `lightweight_cnn.onnx` 加入 app bundle，通过 ONNX Runtime iOS API 加载模型，使用 Swift/Objective-C 完成预处理和推理。
- 也可以进一步转换到移动端更常见的格式，例如 Core ML 或 TensorFlow Lite，但这超出当前项目实现范围。

当前课程项目选择 ONNX Runtime CPU 推理作为边缘端部署模拟，因为它能验证模型离开 PyTorch 训练框架后仍可独立运行。

## 为什么轻量化模型适合边缘端

边缘设备通常受限于 CPU 性能、内存、存储空间和功耗。轻量化模型更适合部署到这类环境，原因包括：

- 参数量更少，模型文件更小，便于存储和传输。
- 深度可分离卷积降低卷积层计算成本。
- Global Average Pooling 减少全连接层参数。
- ONNX Runtime 可以在无 PyTorch 环境下执行推理，降低部署依赖。

## 当前模型指标

基于已有 20 epoch 实验结果：

| 模型 | 参数量 | 模型大小 | 推理时间 | 测试准确率 |
| --- | ---: | ---: | ---: | ---: |
| baseline + rotate | 620362 | 2.37 MB | 0.259 ms | 0.7935 |
| lightweight + rotate | 44362 | 0.185 MB | 1.177 ms | 0.7791 |

轻量化 CNN 相比 baseline CNN：

- 参数量从约 620k 降至约 44k，减少约 92.85%。
- 模型大小从约 2.37 MB 降至约 0.185 MB，减少约 92.18%。
- 在显著降低模型体积的同时，测试准确率仍保持在接近 baseline 的水平。

## 注意事项

- `checkpoints/lightweight_cnn.onnx` 属于生成文件，GitHub 仓库默认不提交大模型文件；交作业时可根据需要单独附带。
- 树莓派或移动端的真实推理速度会受硬件、系统、ONNX Runtime 版本和线程设置影响。
- 当前 `infer_onnx.py` 是单张图像推理示例，不包含摄像头实时采集或移动端 UI。
