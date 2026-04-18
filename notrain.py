from ultralytics import YOLO
import argparse
import cv2
import os
import re
import time
import torch

from prompt_expander import expand_prompts


PROMPT_ALIASES = {
    "人": "person",
    "行人": "person",
    "车": "car",
    "汽车": "car",
    "单车": "bicycle",
    "自行车": "bicycle",
    "摩托": "motorcycle",
    "摩托车": "motorcycle",
    "公交车": "bus",
    "卡车": "truck",
    "猫": "cat",
    "狗": "dog",
    "手机": "cell phone",
    "电脑": "laptop",
    "笔记本": "laptop",
}


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def dedupe_keep_order(items):
    seen = set()
    output = []
    for item in items:
        key = normalize_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item.strip())
    return output


def parse_inline_prompts(prompt_text):
    if not prompt_text:
        return []
    tokens = [t for t in re.split(r"[，,、\s;；]+", prompt_text) if t.strip()]
    return dedupe_keep_order([PROMPT_ALIASES.get(normalize_text(t), t) for t in tokens])


def should_reject_by_negative(label_name, negative_prompts):
    if not negative_prompts:
        return False
    ln = normalize_text(label_name)
    for neg in negative_prompts:
        n = normalize_text(neg)
        if not n:
            continue
        if n == ln or (n in ln) or (ln in n):
            return True
    return False


def load_prompts_from_file(prompt_file):
    if not prompt_file:
        return []
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"未找到提示词文件: {prompt_file}")

    items = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower() in {"all", "*", "全部"}:
                continue
            items.append(line)
    return dedupe_keep_order(items)


def build_prompt_list(args):
    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        source = f"文件 {args.prompts_file}"
        return prompts, source

    if args.prompt_text is not None and args.prompt_text.strip() != "":
        prompts = parse_inline_prompts(args.prompt_text)
        source = "命令行参数 --prompt-text"
        return prompts, source

    auto_file = "your_4000_prompts.txt"
    if os.path.exists(auto_file):
        prompts = load_prompts_from_file(auto_file)
        source = f"自动检测文件 {auto_file}"
        return prompts, source

    print("请输入提示词，多个词可用逗号/空格分隔；直接回车表示不过滤")
    raw = input("提示词: ").strip()
    if normalize_text(raw) in {"", "all", "*", "全部"}:
        return [], "交互输入"
    prompts = parse_inline_prompts(raw)
    return prompts, "交互输入"


def main(argv=None):
    parser = argparse.ArgumentParser(description="YOLOE 开放词表摄像头识别（支持 4000+ 提示词）")
    parser.add_argument("--model", default="yoloe-26x-seg.pt", help="模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument("--prompt-text", default=None, help="直接传入提示词文本")
    parser.add_argument("--negative-prompt-text", default="", help="负提示词文本")
    parser.add_argument("--prompts-file", default=None, help="提示词文件路径，每行一个词")
    parser.add_argument("--imgsz", type=int, default=960, help="推理输入尺寸")
    parser.add_argument("--device", default="auto", help="推理设备: auto/cpu/cuda:0")
    parser.add_argument("--enable-llm-expand", action="store_true", help="启用 LLM 扩词")
    parser.add_argument("--llm-provider", default="openai-compatible", help="LLM 提供方(openai-compatible/ollama)")
    parser.add_argument("--llm-model", default="deepseek-chat", help="LLM 模型名称")
    parser.add_argument("--llm-api-base", default="https://api.deepseek.com", help="LLM 接口基础地址")
    parser.add_argument("--llm-api-key", default=None, help="LLM 接口密钥")
    parser.add_argument("--llm-max-terms", type=int, default=80, help="扩展后最大提示词数")
    parser.add_argument("--llm-timeout", type=int, default=40, help="LLM 请求超时秒数")
    parser.add_argument("--domain-hint", default="", help="领域提示")
    parser.add_argument("--component-combine", action="store_true", help="将识别类别统一显示为固定词")
    parser.add_argument("--component-name", default="", help="成分组合固定类别名")
    parser.add_argument("--preview-file", default=None, help="将实时预览帧写入该文件路径（供GUI内嵌显示）")
    parser.add_argument("--preview-write-interval-ms", type=int, default=80, help="预览文件写入间隔毫秒")
    parser.add_argument("--stats-interval", type=int, default=3, help="进度统计输出间隔帧数")
    parser.add_argument("--no-window", action="store_true", help="不弹出 OpenCV 窗口（用于 GUI 嵌入模式）")
    args = parser.parse_args(argv)

    req_device = str(args.device or "auto").strip().lower()
    if req_device == "auto":
        runtime_device = "0" if torch.cuda.is_available() else "cpu"
    elif req_device.startswith("cuda"):
        if torch.cuda.is_available():
            if ":" in req_device:
                runtime_device = req_device.split(":", 1)[1].strip() or "0"
            else:
                runtime_device = "0"
        else:
            print("CUDA 不可用，已自动回退到 CPU")
            runtime_device = "cpu"
    else:
        runtime_device = req_device

    print(f"推理设备: 请求={args.device} 实际={runtime_device}")

    if not os.path.exists(args.model):
        print(f"未找到模型文件: {args.model}")
        print("请检查模型路径，或使用 --model 指定正确路径")
        return

    model = YOLO(args.model)
    builtin_count = len(model.names) if hasattr(model, "names") else 0
    print(f"模型已加载: {args.model}")
    print(f"模型当前默认类别数: {builtin_count}")

    prompts, source = build_prompt_list(args)
    negative_prompts = parse_inline_prompts(args.negative_prompt_text)

    if prompts and args.enable_llm_expand:
        print(
            f"开始LLM扩词: provider={args.llm_provider} model={args.llm_model} timeout={args.llm_timeout}s"
        )
        prompts, info = expand_prompts(
            seed_prompts=prompts,
            enable_llm=True,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_api_base=args.llm_api_base,
            llm_api_key=args.llm_api_key,
            llm_max_terms=args.llm_max_terms,
            llm_timeout=args.llm_timeout,
            domain_hint=args.domain_hint,
        )
        print(f"提示词扩展模式: {info.get('mode', 'unknown')} | 扩展后数量: {len(prompts)}")
        if info.get("error"):
            print(f"LLM 扩展异常: {info.get('error')}")

    if prompts:
        print(f"从 {source} 读取到提示词 {len(prompts)} 个，正在设置开放词表...")
        try:
            model.set_classes(prompts)
        except Exception as e:
            print("设置开放词表失败，请检查 CLIP 依赖是否安装完整")
            print("如网络不稳定，请先手动下载 mobileclip2_b.ts 到项目目录后重试")
            print(f"错误信息: {e}")
            return

        preview = ", ".join(prompts[:12])
        if len(prompts) > 12:
            preview += ", ..."
        print(f"开放词表已启用，词表大小: {len(prompts)}")
        print(f"提示词预览: {preview}")
    else:
        print("未指定提示词，使用模型默认类别检测")

    if negative_prompts:
        print(f"负提示词: {', '.join(negative_prompts[:12])}")

    combine_name = str(args.component_name or "").strip()
    use_component_combine = bool(args.component_combine and combine_name)
    if args.component_combine and not combine_name:
        print("你启用了成分组合，但未提供固定类别名，已自动回退为普通显示")

    if use_component_combine:
        print(f"成分组合已启用，统一类别名: {combine_name}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"摄像头打开失败: {args.camera}")
        return

    if args.preview_file:
        preview_path = os.path.abspath(args.preview_file)
        print(f"PREVIEW_FILE: {preview_path}")
    else:
        preview_path = None

    if not args.no_window:
        cv2.namedWindow("YOLOE 实时识别", cv2.WINDOW_NORMAL)
        print("按 Q 键退出")
    else:
        print("无窗口模式运行中，可通过外部停止进程")

    frame_idx = 0
    total_det = 0
    no_result_count = 0
    total_conf_sum = 0.0
    total_conf_count = 0
    last_preview_write_ts = 0.0
    stats_interval = max(1, int(args.stats_interval))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头失败")
            break

        results = model(frame, conf=args.conf, imgsz=args.imgsz, device=runtime_device, verbose=False)
        r = results[0]
        frame_show = frame.copy()
        frame_det = 0
        names = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for i, box in enumerate(xyxy):
                cls_name = str(names.get(int(clss[i]), clss[i]))
                if should_reject_by_negative(cls_name, negative_prompts):
                    continue
                x1, y1, x2, y2 = box.tolist()
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = max(x1 + 1, x2)
                y2 = max(y1 + 1, y2)
                cv2.rectangle(frame_show, (x1, y1), (x2, y2), (34, 197, 94), 2)
                draw_name = combine_name if use_component_combine else cls_name
                label = f"{draw_name} {float(confs[i]):.2f}"
                cv2.putText(frame_show, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (34, 197, 94), 2)
                frame_det += 1
                total_conf_sum += float(confs[i])
                total_conf_count += 1

        total_det += frame_det
        if frame_det == 0:
            no_result_count += 1

        if preview_path:
            now_ts = time.monotonic()
            if (now_ts - last_preview_write_ts) * 1000.0 >= float(max(20, args.preview_write_interval_ms)):
                cv2.imwrite(preview_path, frame_show)
                last_preview_write_ts = now_ts

        if not args.no_window:
            cv2.imshow("YOLOE 实时识别", frame_show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            time.sleep(0.02)

        frame_idx += 1
        if frame_idx % stats_interval == 0:
            avg_conf = (total_conf_sum / float(total_conf_count)) if total_conf_count > 0 else 0.0
            print(
                f"进度 {frame_idx}/{frame_idx} | det {total_det} | seg 0 | 未识别 {no_result_count} | 失败 0 | 平均置信 {avg_conf:.3f}"
            )
        if frame_idx % 60 == 0:
            print(f"STEP: 摄像头识别中 | 帧 {frame_idx}")

    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()