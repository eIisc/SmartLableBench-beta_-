from ultralytics import YOLO
import argparse


def load_prompts_file(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(description="导出模型支持的全部提示词到 Markdown")
    parser.add_argument("--model", default="yoloe-26x-seg.pt", help="模型路径")
    parser.add_argument("--prompts-file", default=None, help="可选：提示词文件（每行一个词）")
    parser.add_argument("--out", default="PROMPTS.md", help="输出 Markdown 文件")
    args = parser.parse_args(argv)

    if args.prompts_file:
        prompts = load_prompts_file(args.prompts_file)
        id_to_name = {i: p for i, p in enumerate(prompts)}
        source = f"提示词文件: {args.prompts_file}"
    else:
        model = YOLO(args.model)
        names = model.names
        if isinstance(names, dict):
            id_to_name = {int(k): str(v) for k, v in names.items()}
        else:
            id_to_name = {i: str(v) for i, v in enumerate(names)}
        source = f"模型内置 names: {args.model}"

    total = len(id_to_name)
    lines = [
        "# 提示词支持列表",
        "",
        source,
        f"类别总数: {total}",
        "",
        "说明:",
        "- 英文提示词默认不区分大小写",
        "- 多提示词可使用 , ， 空格 、 ; ； 分隔",
        "- all / * / 全部 表示不过滤",
        "",
        "## 全部类别词",
        "",
    ]

    for idx in sorted(id_to_name):
        lines.append(f"- {idx}: {id_to_name[idx]}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"导出完成: {args.out} (共 {total} 类)")


if __name__ == "__main__":
    main()
