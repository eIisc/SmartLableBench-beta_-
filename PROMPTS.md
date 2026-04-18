# 提示词支持说明（YOLOE 开放词表版）

当前程序已重构为开放词表流程，适配 4000+ 提示词：

- 不再依赖 model.names 的固定类别
- 支持从文本文件批量加载提示词并调用 set_classes
- 适用于 yoloe-26l-seg.pt 这类 YOLOE 模型
- 如果项目根目录存在 your_4000_prompts.txt，会自动加载该词表

## 快速开始

1. 准备提示词文件（每行一个词）

可直接复制并改造模板文件：


2. 运行摄像头检测（4000+ 词表示例）

python notrain.py --model yoloe-26l-seg.pt --prompts-file your_4000_prompts.txt

或直接把词表文件命名为 your_4000_prompts.txt 放到项目根目录，然后运行：

python notrain.py --model yoloe-26l-seg.pt

3. 也可直接传入少量提示词

python notrain.py --model yoloe-26l-seg.pt --prompt-text "person,car,dog"

## 说明

- 提示词文件支持注释行（以 # 开头）
- 中文快速别名仍可用（人、车、狗、手机等）
- 如果不提供提示词，程序会回退到模型默认类别
- YOLOE 在首次 set_classes 时会下载 mobileclip2_b.ts；网络不稳定时可手动下载到项目目录
