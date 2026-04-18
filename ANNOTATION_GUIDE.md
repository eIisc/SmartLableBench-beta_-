# YOLOE 提示词数据集自动标注指南

## 功能

- 基于提示词词表批量标注图片数据集
- 输出检测框标签（YOLO det 格式）
- 输出分割标签（YOLO seg 格式）
- 可选输出可视化图片
- 输出统计摘要 summary.json

## 输入准备

1. 模型文件（默认）
- yoloe-26l-seg.pt

2. 提示词文件
- 推荐使用 your_4000_prompts.txt
- 每行一个词，支持注释行（# 开头）

3. 图片目录
- 支持递归扫描 jpg/jpeg/png/bmp/webp

## 快速开始

统一入口（推荐）：

python main.py annotate --images your_dataset/images --prompts-file your_4000_prompts.txt --output your_dataset/labels_auto --save-vis

## Qt 图形界面（推荐）

安装依赖：

python -m pip install PySide6

启动界面：

python main.py gui

界面能力：

- 选择图片目录、提示词文件、输出目录
- 维护模型列表（可添加多个模型串行标注）
- 设置 conf / iou / imgsz / open-vocab
- 一键开始和停止任务
- 实时日志输出
- 可启用 LLM 自动扩展提示词（面向非标准目标）
- 默认支持本地 Ollama 扩词（无需云端）
- 预览图在同一窗口内实时刷新（无需弹出外部窗口）
- 显示进度、det、seg、未识别+失败数量

多模型输出会自动按模型名分目录，例如：

- your_dataset/labels_auto/yoloe-26l-seg/
- your_dataset/labels_auto/another_model/

## 输出结构

- labels_auto/
- labels_auto/classes.txt
- labels_auto/labels_det/*.txt
- labels_auto/labels_seg/*.txt
- labels_auto/vis/*（使用 --save-vis 时）
- labels_auto/summary.json

## 参数说明

- --model 模型路径（默认 yoloe-26l-seg.pt）
- --images 图片目录（必填）
- --output 输出目录（默认 dataset_annotations）
- --prompts-file 提示词文件路径
- --prompt-text 直接输入提示词字符串（可替代 prompts-file）
- --conf 置信度阈值，默认 0.25
- --iou NMS IoU 阈值，默认 0.6
- --imgsz 推理尺寸，默认 1024
- --save-vis 保存可视化结果
- --open-vocab 开放词表模式：auto/on/off
- --enable-llm-expand 启用 LLM 扩展提示词
- --llm-provider LLM 提供方（当前支持 openai-compatible）
- --llm-model LLM 模型名（如 gpt-4o-mini / qwen 等）
- --llm-api-base OpenAI 兼容接口地址
- --llm-api-key LLM 密钥（也可用环境变量 LLM_API_KEY）
- --llm-max-terms 扩展后最多提示词数
- --domain-hint 任务领域提示（如 abnormal cell / wall crack）

## 非标准目标推荐流程

以异常细胞为例：

python main.py annotate --images your_dataset/images --prompts-file seed_prompts.txt --enable-llm-expand --domain-hint "abnormal cell microscopy" --open-vocab on --output your_dataset/labels_cell

以墙体裂缝为例：

python main.py annotate --images your_dataset/images --prompts-file seed_prompts.txt --enable-llm-expand --domain-hint "wall crack concrete defect" --open-vocab on --output your_dataset/labels_crack

任意冷门目标示例（只是输入种子词即可）：

python main.py annotate --images your_dataset/images --prompt-text "rare defect object" --enable-llm-expand --domain-hint "industrial anomaly inspection" --open-vocab on --output your_dataset/labels_custom

程序会额外输出：

- prompts_used.txt（最终参与检测的扩展词表）
- summary.json（记录扩展模式与错误回退信息）

## API 扩词（推荐）

- provider: `openai-compatible`
- api base: `https://api.openai.com/v1` 或你的兼容网关
- model 示例: `gpt-4o-mini`
- 在 GUI 填入 `api key`

示例（青苹果）：

1. 提示词文本输入：`青苹果`
2. 勾选：`启用API扩词` + `强制扩词`
3. 运行后会自动扩展为 `green apple / unripe apple / apple fruit ...` 等词再喂给 YOLO

## 本地 LLM（Ollama）备用

- provider: `ollama`
- api base: `http://127.0.0.1:11434`
- model 示例: `qwen2.5:7b-instruct`

先确保本地模型已拉取并可对话，再开始标注。

## 空标签排查

如果看到大量空 txt：

1. 查看日志中的“提示词来源”是否正确（文本输入/文件）
2. 适当降低 `conf`（如 0.2 -> 0.1）
3. 检查 `open-vocab` 是否使用 `auto` 或 `on`
4. 查看 `prompts_used.txt`，确认扩词结果是否符合目标外观

## 强制扩词

- 勾选 GUI 中“强制扩词”，或命令行加 `--force-llm-expand`
- 若本地 LLM 不可用，任务会直接失败并提示，不会静默回退

## 速度和准确性建议

- 要更快：降低 imgsz（如 768）并提高 conf（如 0.35）
- 要更准：提高 imgsz（如 1280）并适当降低 conf（如 0.2）
- 先用 200 张样本调参，再全量跑

## 常见问题

1. set_classes 失败
- 确认 mobileclip2_b.ts 已在项目目录
- 确认环境已安装 ultralytics 与 CLIP 依赖

2. 标签为空
- 降低 conf，或检查提示词是否过窄
- 检查图片质量与分辨率
