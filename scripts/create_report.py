from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
REPORT_PATH = REPORT_DIR / "计算机视觉大作业报告.docx"
COMPARE_CSV = ROOT / "results" / "model_compare.csv"


def fmt_float(value: str | float | None, digits: int = 4) -> str:
    if value in {None, ""}:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_mb(value: str | float | None) -> str:
    return f"{float(value):.3f} MB" if value not in {None, ""} else "-"


def fmt_ms(value: str | float | None) -> str:
    return f"{float(value):.3f} ms" if value not in {None, ""} else "-"


def load_compare_rows() -> list[dict[str, str]]:
    if not COMPARE_CSV.exists():
        return []
    with COMPARE_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_row(rows: list[dict[str, str]], model: str, augment: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("model") == model and row.get("augment") == augment:
            return row
    return None


def percent_reduction(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return (before - after) / before * 100.0


def text_run(text: str, bold: bool = False, size: int | None = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if size:
        props.append(f'<w:sz w:val="{size}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f"<w:r>{rpr}<w:t>{escape(text)}</w:t></w:r>"


def paragraph(text: str = "", style: str | None = None, align: str | None = None) -> str:
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    return f"<w:p>{ppr_xml}{text_run(text)}</w:p>"


def heading(text: str, level: int) -> str:
    return paragraph(text, style=f"Heading{level}")


def bullet(text: str) -> str:
    return paragraph("• " + text)


def code(text: str) -> str:
    lines = text.splitlines() or [text]
    return "".join(paragraph(line, style="Code") for line in lines)


def table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(value: str) -> str:
        return (
            '<w:tc><w:tcPr><w:tcW w:w="2400" w:type="dxa"/></w:tcPr>'
            f"{paragraph(value)}</w:tc>"
        )

    def row(values: list[str]) -> str:
        return "<w:tr>" + "".join(cell(value) for value in values) + "</w:tr>"

    borders = (
        "<w:tblPr><w:tblBorders>"
        '<w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/>'
        '<w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/>'
        '<w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/>'
        "</w:tblBorders></w:tblPr>"
    )
    return "<w:tbl>" + borders + row(headers) + "".join(row(item) for item in rows) + "</w:tbl>"


def result_table_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    if not rows:
        return [["暂无", "暂无", "暂无", "暂无", "暂无", "暂无", "暂无"]]
    return [
        [
            row.get("model", "-"),
            row.get("augment", "-"),
            fmt_float(row.get("accuracy"), 4),
            fmt_float(row.get("loss"), 4),
            row.get("params", "-"),
            fmt_mb(row.get("model_size_mb")),
            fmt_ms(row.get("inference_ms")),
        ]
        for row in rows
    ]


def lightweight_analysis(rows: list[dict[str, str]]) -> list[str]:
    baseline = find_row(rows, "baseline", "rotate") or find_row(rows, "baseline", "none")
    lightweight = find_row(rows, "lightweight", "rotate") or find_row(rows, "lightweight", "none")
    if not baseline or not lightweight:
        return ["当前未读取到完整 baseline 与 lightweight 对比结果，生成报告后可先运行完整实验再重新生成。"]

    base_params = float(baseline["params"])
    light_params = float(lightweight["params"])
    base_size = float(baseline["model_size_mb"])
    light_size = float(lightweight["model_size_mb"])
    base_acc = float(baseline["accuracy"])
    light_acc = float(lightweight["accuracy"])
    return [
        f"baseline 参数量约 {base_params / 1000:.0f}k，lightweight 参数量约 {light_params / 1000:.0f}k，参数量减少约 {percent_reduction(base_params, light_params):.2f}%。",
        f"模型大小从约 {base_size:.2f} MB 降至约 {light_size:.3f} MB，减少约 {percent_reduction(base_size, light_size):.2f}%。",
        f"在已有实验中，baseline 测试准确率约 {base_acc:.4f}，lightweight 测试准确率约 {light_acc:.4f}，轻量化模型以明显更小的体积保持了接近的识别能力。",
    ]


def document_body() -> str:
    rows = load_compare_rows()
    analysis = lightweight_analysis(rows)
    parts = [
        paragraph("基于 CIFAR-10 的轻量化目标识别方法及实现", style="Title", align="center"),
        paragraph("计算机视觉大作业报告", align="center"),
        heading("摘要", 1),
        paragraph(
            "本文围绕 CIFAR-10 图像分类任务，设计并实现基础 CNN 和轻量化 CNN 两类模型。"
            "实验从准确率、损失、参数量、模型大小、单张图片推理时间和 FPS 等指标进行比较，"
            "并使用无增强、旋转翻转增强、噪声增强和强增强策略分析数据增强对模型泛化能力的影响。"
            "在部署部分，本文将训练后的轻量化模型导出为 ONNX 格式，并使用 ONNX Runtime 完成 CPU 环境下的边缘端推理模拟。"
        ),
        paragraph("关键词：CIFAR-10；图像分类；轻量化 CNN；深度可分离卷积；数据增强；ONNX Runtime"),
        heading("第 1 章 绪论", 1),
        heading("1.1 研究背景", 2),
        paragraph(
            "目标识别是计算机视觉中的基础任务之一，广泛应用于智能终端、工业检测、自动驾驶和移动机器人等场景。"
            "随着模型部署场景逐渐从云端扩展到移动端和边缘设备，模型不仅需要具备较高的识别准确率，还需要满足参数量小、推理速度快和存储开销低等要求。"
        ),
        heading("1.2 本文主要工作", 2),
        bullet("完成 CIFAR-10 数据集加载、预处理和样本可视化。"),
        bullet("实现基础 CNN 模型作为对照组。"),
        bullet("设计基于深度可分离卷积和全局平均池化的轻量化 CNN。"),
        bullet("补充旋转、翻转、颜色扰动和高斯噪声等数据增强策略。"),
        bullet("统计模型参数量、模型大小、推理时间、FPS、准确率和损失。"),
        bullet("导出 ONNX 模型并完成 ONNX Runtime 边缘端推理模拟。"),
        bullet("本项目选择 CNN 路线，ViT/MobileViT 作为后续扩展方向。"),
        heading("第 2 章 数据集与预处理", 1),
        paragraph(
            "CIFAR-10 数据集包含 60000 张 32x32 彩色图像，共 10 个类别。其中训练集 50000 张，测试集 10000 张。"
            "类别包括 airplane、automobile、bird、cat、deer、dog、frog、horse、ship 和 truck。"
        ),
        table(
            ["策略参数", "数据增强方式", "目的"],
            [
                ["none", "无增强", "基础对照组"],
                ["rotate", "随机裁剪 + 水平翻转 + 随机旋转", "提升位置、方向和角度变化适应能力"],
                ["noise", "随机裁剪 + 水平翻转 + 高斯噪声", "模拟采集噪声，提高鲁棒性"],
                ["strong", "随机裁剪 + 水平翻转 + 随机旋转 + 颜色扰动 + 高斯噪声", "更强扰动，用于验证泛化能力"],
            ],
        ),
        paragraph("其中高斯噪声在 ToTensor() 之后、Normalize() 之前加入，避免对已经标准化后的张量直接加噪。"),
        heading("第 3 章 模型设计", 1),
        heading("3.1 基础 CNN 模型", 2),
        paragraph("基础 CNN 由三组卷积、ReLU 和最大池化组成，最后通过全连接层输出 10 类分类结果。该模型结构简单，便于作为 baseline 分析后续改进是否有效。"),
        code("Input 32x32x3 -> Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC -> Output"),
        heading("3.2 轻量化 CNN 模型", 2),
        paragraph("轻量化 CNN 使用深度可分离卷积替代部分标准卷积，并结合 Batch Normalization、Dropout 和 Global Average Pooling 降低参数量与计算开销。"),
        table(
            ["改进方法", "作用"],
            [
                ["深度可分离卷积", "减少卷积层参数量"],
                ["Batch Normalization", "加快训练收敛"],
                ["Dropout", "缓解过拟合"],
                ["Global Average Pooling", "减少全连接层参数量"],
                ["减少通道数", "降低模型计算量"],
            ],
        ),
        heading("第 4 章 实验设计与实现", 1),
        table(
            ["项目", "设置"],
            [
                ["编程语言", "Python"],
                ["深度学习框架", "PyTorch"],
                ["数据集", "CIFAR-10"],
                ["图像大小", "32x32"],
                ["训练轮数", "20"],
                ["Batch Size", "64"],
                ["优化器", "Adam"],
                ["学习率", "0.001"],
                ["损失函数", "CrossEntropyLoss"],
                ["设备", "MPS 或 CPU"],
                ["模型数量", "基础 CNN + 轻量化 CNN"],
                ["部署方式", "ONNX + ONNX Runtime CPU 推理模拟"],
            ],
        ),
        paragraph("核心训练命令示例："),
        code("python train.py --model lightweight --augment strong --epochs 20 --batch-size 64"),
        heading("第 5 章 实验结果与分析", 1),
        paragraph("报告生成脚本会自动读取 results/model_compare.csv，并将已有实验结果写入下表。"),
        table(
            ["模型", "增强", "准确率", "损失", "参数量", "模型大小", "推理时间"],
            result_table_rows(rows),
        ),
        heading("5.1 轻量化效果分析", 2),
        *[bullet(item) for item in analysis],
        heading("5.2 数据增强分析", 2),
        paragraph(
            "从已有结果看，baseline CNN 在 rotate 策略下准确率高于无增强策略，说明裁剪、翻转和旋转有助于提升泛化能力。"
            "新增的 noise 和 strong 策略已经在代码中支持，可继续运行完整 20 epoch 实验补充对比。"
        ),
        heading("第 6 章 ONNX Runtime 边缘端推理模拟", 1),
        paragraph(
            "本文采用 ONNX 导出和 ONNX Runtime 推理测试模拟边缘端部署。该流程可以验证 PyTorch 模型能否脱离训练框架运行，"
            "并统计 CPU 环境下的单张图像推理时间。ONNX 模型可迁移到树莓派、普通 CPU 设备或移动端 ONNX Runtime 环境。"
        ),
        code(
            "python export_onnx.py --model lightweight --checkpoint checkpoints/lightweight_rotate.pth --output checkpoints/lightweight_cnn.onnx\n"
            "python infer_onnx.py --onnx checkpoints/lightweight_cnn.onnx --repeat 20"
        ),
        paragraph("当前项目完成 ONNX 导出与 ONNX Runtime CPU 推理验证，不声明已经完成 Android/iOS 真机部署。"),
        heading("第 7 章 不足与改进方向", 1),
        bullet("当前未实现真实 Android/iOS App，仅完成 ONNX Runtime CPU 推理模拟。"),
        bullet("当前未实现 ViT 或 MobileViT，后续可扩展小型 ViT/MobileViT 结构进行对比。"),
        bullet("后续可尝试模型量化、剪枝、知识蒸馏，进一步压缩模型体积和提升边缘端速度。"),
        bullet("后续可在树莓派上进行真实硬件实测，记录端侧推理延迟、功耗和稳定性。"),
        heading("第 8 章 总结", 1),
        paragraph(
            "本文实现了基于 CIFAR-10 的轻量化目标识别系统。通过模型结构对比、数据增强策略扩展、指标统计和 ONNX Runtime 推理模拟，"
            "项目覆盖了课程题目中关于数据集、CNN 模型、结构对比、增强策略和边缘端部署的核心要求。"
        ),
        heading("参考文献", 1),
        paragraph("[1] Krizhevsky A. Learning Multiple Layers of Features from Tiny Images, 2009."),
        paragraph("[2] Howard A. G. et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, 2017."),
        paragraph("[3] PyTorch Documentation. https://pytorch.org/docs/"),
        paragraph("[4] ONNX Runtime Documentation. https://onnxruntime.ai/docs/"),
    ]
    return "".join(parts)


def write_docx(path: Path, body: str) -> None:
    document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1247" w:right="1417" w:bottom="1247" w:left="1417" w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>'''
    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="21"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Courier New" w:eastAsia="Courier New"/><w:sz w:val="18"/></w:rPr></w:style>
</w:styles>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    now = datetime.now(timezone.utc).isoformat()
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>基于 CIFAR-10 的轻量化目标识别方法及实现</dc:title>
  <dc:creator>Codex</dc:creator>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Codex</Application></Properties>'''

    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles_xml)
        docx.writestr("word/_rels/document.xml.rels", doc_rels)
        docx.writestr("docProps/core.xml", core)
        docx.writestr("docProps/app.xml", app)


def main() -> None:
    write_docx(REPORT_PATH, document_body())
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
