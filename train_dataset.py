import argparse
import random
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import gc
import importlib
import subprocess
import sys

from ultralytics import YOLO

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve_annotation_root(annotation_dir: Path):
    """Resolve actual annotation root that contains labels_det/classes.txt.

    Supports both direct root (annotation_dir/labels_det) and parent folders that
    contain one or more annotation subfolders.
    """
    ann_dir = Path(annotation_dir)
    labels_dir = ann_dir / "labels_det"
    classes_file = ann_dir / "classes.txt"

    if labels_dir.exists() and classes_file.exists():
        return ann_dir, labels_dir, classes_file

    candidates = []
    if ann_dir.exists() and ann_dir.is_dir():
        for p in ann_dir.iterdir():
            if not p.is_dir():
                continue
            c_labels = p / "labels_det"
            c_classes = p / "classes.txt"
            if c_labels.exists() and c_classes.exists():
                candidates.append(p)

    if len(candidates) == 1:
        resolved = candidates[0]
        return resolved, resolved / "labels_det", resolved / "classes.txt"

    if len(candidates) > 1:
        candidate_text = ", ".join(str(x) for x in candidates)
        raise RuntimeError(
            "annotation-dir 下存在多个可用标注目录，请指定更精确路径: " + candidate_text
        )

    raise FileNotFoundError(
        f"未找到检测标签目录与类别文件: {ann_dir} (期望包含 labels_det 和 classes.txt)"
    )


def _infer_is_seg_model(model_path: str):
    stem = Path(str(model_path or "")).stem.lower()
    # Common naming convention: xxx-seg.pt / xxx_seg.pt
    return ("-seg" in stem) or ("_seg" in stem) or stem.endswith("seg")


def _resolve_labels_dir(ann_dir: Path, args):
    mode = str(getattr(args, "label_kind", "auto") or "auto").strip().lower()
    det_dir = ann_dir / "labels_det"
    seg_dir = ann_dir / "labels_seg"

    if mode == "det":
        if not det_dir.exists():
            raise FileNotFoundError(f"未找到检测标签目录: {det_dir}")
        return det_dir, "det"

    if mode == "seg":
        if not seg_dir.exists():
            raise FileNotFoundError(f"未找到分割标签目录: {seg_dir}")
        return seg_dir, "seg"

    # auto mode
    prefer_seg = _infer_is_seg_model(getattr(args, "model", ""))
    if prefer_seg and seg_dir.exists():
        return seg_dir, "seg"
    if det_dir.exists():
        return det_dir, "det"
    if seg_dir.exists():
        return seg_dir, "seg"
    raise FileNotFoundError(f"未找到可用标签目录: {det_dir} 或 {seg_dir}")


def _validate_label_style(labels_dir: Path, label_kind: str):
    """Quick sanity check to avoid cryptic ultralytics runtime crashes."""
    for p in labels_dir.glob("*.txt"):
        raw = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            continue
        for line in raw.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if label_kind == "seg":
                if len(parts) <= 5:
                    raise RuntimeError(
                        f"检测到分割训练但标签像检测框格式(仅{len(parts)}列): {p} | 行: {line[:120]}"
                    )
            else:
                if len(parts) < 5:
                    raise RuntimeError(f"检测到无效检测标签(<5列): {p} | 行: {line[:120]}")
            return


def _apply_memory_safety(args):
    """Log memory status without mutating user-provided workers/batch."""
    if getattr(args, "no_safe_memory", False):
        return

    if psutil is None:
        print("内存保护: 未安装 psutil，跳过自适应限流")
        return

    vm = psutil.virtual_memory()
    total_gb = float(vm.total) / (1024.0 ** 3)
    cpu_mode = str(args.device).lower() == "cpu"
    cuda_ok = bool(torch is not None and torch.cuda.is_available())
    vram_total_gb = 0.0
    if cuda_ok:
        try:
            dev = torch.cuda.current_device()
            prop = torch.cuda.get_device_properties(dev)
            vram_total_gb = float(getattr(prop, "total_memory", 0.0) or 0.0) / (1024.0 ** 3)
        except Exception:
            vram_total_gb = 0.0

    if cpu_mode:
        print("内存保护: CPU模式，仅监控不自动调整workers/batch")
    else:
        print("内存保护: GPU模式，仅监控不自动调整workers/batch")

    print(
        f"内存保护启用: RAM={total_gb:.1f}GB 当前占用={vm.percent:.1f}% "
        f"workers={args.workers} batch={args.batch} "
        f"RAM阈值={args.max_ram_percent:.1f}% VRAM阈值={args.max_vram_percent:.1f}%"
    )
    if cuda_ok and vram_total_gb > 0:
        print(f"显存信息: 总显存={vram_total_gb:.1f}GB")


def _ensure_onnx_requirements_in_current_env():
    """Ensure ONNX export deps are available in current interpreter environment."""
    required_modules = ["onnx", "onnxruntime", "onnxslim"]
    missing = []
    for mod in required_modules:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)

    if not missing:
        return

    pkg_map = {
        "onnx": "onnx>=1.12.0,<2.0.0",
        "onnxruntime": "onnxruntime",
        "onnxslim": "onnxslim>=0.1.71",
    }
    install_list = [pkg_map[m] for m in missing if m in pkg_map]
    print(f"ONNX导出依赖缺失(当前环境): {', '.join(missing)}")
    print("尝试在当前解释器环境安装依赖...")
    cmd = [sys.executable, "-m", "pip", "install", *install_list]
    subprocess.check_call(cmd)

    importlib.invalidate_caches()
    for mod in missing:
        importlib.import_module(mod)


def collect_images(images_dir: Path):
    files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def read_classes(classes_file: Path):
    if not classes_file.exists():
        raise FileNotFoundError(f"未找到类别文件: {classes_file}")
    items = []
    for line in classes_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            items.append(s)
    if not items:
        raise RuntimeError("classes.txt 为空，无法训练")
    return items


def make_split(items, val_ratio: float, seed: int):
    data = list(items)
    random.Random(seed).shuffle(data)
    n = len(data)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 1
    n_val = min(n - 1, n_val) if n > 1 else 1
    val_items = data[:n_val]
    train_items = data[n_val:] if n > 1 else data
    return train_items, val_items


def copy_split(items, labels_dir: Path, img_out: Path, lbl_out: Path):
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in items:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        # 空标签不参与训练
        if not label_path.read_text(encoding="utf-8").strip():
            continue
        shutil.copy2(str(image_path), str(img_out / image_path.name))
        shutil.copy2(str(label_path), str(lbl_out / f"{image_path.stem}.txt"))
        count += 1
    return count


def write_data_yaml(dataset_dir: Path, names):
    data_file = dataset_dir / "data.yaml"
    lines = [
        f"path: {dataset_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    data_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_file


def _read_image_size(image_path: Path):
    # 优先使用 OpenCV 读取尺寸；如果不可用则使用内置格式解析。
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h), 3
    except Exception:
        pass

    from imghdr import what as image_what

    kind = image_what(str(image_path))
    if kind in {"jpeg", "png", "bmp", "tiff", "webp"}:
        # 无法可靠解析所有格式尺寸时，给出明确错误，避免写出错误坐标。
        raise RuntimeError(f"无法读取图像尺寸，请安装 OpenCV 后重试: {image_path}")
    raise RuntimeError(f"不支持的图片格式或无法读取尺寸: {image_path}")


def _to_voc_bbox(xc, yc, bw, bh, img_w, img_h):
    x_center = float(xc) * float(img_w)
    y_center = float(yc) * float(img_h)
    box_w = float(bw) * float(img_w)
    box_h = float(bh) * float(img_h)

    xmin = int(round(x_center - box_w / 2.0))
    ymin = int(round(y_center - box_h / 2.0))
    xmax = int(round(x_center + box_w / 2.0))
    ymax = int(round(y_center + box_h / 2.0))

    xmin = max(0, min(img_w - 1, xmin))
    ymin = max(0, min(img_h - 1, ymin))
    xmax = max(0, min(img_w - 1, xmax))
    ymax = max(0, min(img_h - 1, ymax))

    if xmax <= xmin:
        xmax = min(img_w - 1, xmin + 1)
    if ymax <= ymin:
        ymax = min(img_h - 1, ymin + 1)
    return xmin, ymin, xmax, ymax


def _write_voc_xml(xml_path: Path, image_path: Path, img_w: int, img_h: int, img_c: int, objects):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = image_path.parent.name
    ET.SubElement(root, "filename").text = image_path.name
    ET.SubElement(root, "path").text = str(image_path.resolve())

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(int(img_w))
    ET.SubElement(size, "height").text = str(int(img_h))
    ET.SubElement(size, "depth").text = str(int(img_c))
    ET.SubElement(root, "segmented").text = "0"

    for name, (xmin, ymin, xmax, ymax) in objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(name)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(xmin))
        ET.SubElement(bnd, "ymin").text = str(int(ymin))
        ET.SubElement(bnd, "xmax").text = str(int(xmax))
        ET.SubElement(bnd, "ymax").text = str(int(ymax))

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)


def _build_voc_split(items, labels_dir: Path, names, img_out: Path, xml_out: Path):
    img_out.mkdir(parents=True, exist_ok=True)
    xml_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in items:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        raw = label_path.read_text(encoding="utf-8").strip()
        if not raw:
            continue

        img_w, img_h, img_c = _read_image_size(image_path)
        objects = []
        for line in raw.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_idx = int(float(parts[0]))
                xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                if cls_idx < 0 or cls_idx >= len(names):
                    continue
                bbox = _to_voc_bbox(xc, yc, bw, bh, img_w, img_h)
                objects.append((names[cls_idx], bbox))
            except Exception:
                continue

        if not objects:
            continue

        shutil.copy2(str(image_path), str(img_out / image_path.name))
        xml_path = xml_out / f"{image_path.stem}.xml"
        _write_voc_xml(xml_path, image_path, img_w, img_h, img_c, objects)
        count += 1

    return count


def build_dataset(args):
    ann_dir = Path(args.annotation_dir)
    images_dir = Path(args.images)

    ann_dir, labels_dir, classes_file = _resolve_annotation_root(ann_dir)
    labels_dir, label_kind = _resolve_labels_dir(ann_dir, args)

    if not ann_dir.exists():
        raise FileNotFoundError(f"未找到标注目录: {ann_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"未找到图片目录: {images_dir}")

    print(f"标注目录解析结果: {ann_dir}")
    print(f"标签目录解析结果: {labels_dir} (kind={label_kind})")

    _validate_label_style(labels_dir, label_kind)

    names = read_classes(classes_file)
    images = collect_images(images_dir)
    if not images:
        raise RuntimeError("图片目录为空，无法训练")

    # 仅保留有标签文件的样本
    labeled = [p for p in images if (labels_dir / f"{p.stem}.txt").exists()]
    if len(labeled) < 2:
        raise RuntimeError("有效标注样本不足(至少2张)")

    train_items, val_items = make_split(labeled, args.val_ratio, args.seed)

    dataset_dir = Path(args.dataset_out)
    if args.rebuild_dataset and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    train_img_dir = dataset_dir / "images" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_lbl_dir = dataset_dir / "labels" / "val"

    n_train = copy_split(train_items, labels_dir, train_img_dir, train_lbl_dir)
    n_val = copy_split(val_items, labels_dir, val_img_dir, val_lbl_dir)

    if n_train == 0 or n_val == 0:
        raise RuntimeError("拆分后训练集或验证集为空，请检查标注质量与 val-ratio")

    data_yaml = write_data_yaml(dataset_dir, names)
    return data_yaml, n_train, n_val


def build_dataset_voc_xml(args):
    ann_dir = Path(args.annotation_dir)
    images_dir = Path(args.images)

    ann_dir, labels_dir, classes_file = _resolve_annotation_root(ann_dir)
    labels_dir, label_kind = _resolve_labels_dir(ann_dir, args)

    if not ann_dir.exists():
        raise FileNotFoundError(f"未找到标注目录: {ann_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"未找到图片目录: {images_dir}")

    print(f"标注目录解析结果: {ann_dir}")
    print(f"标签目录解析结果: {labels_dir} (kind={label_kind})")

    _validate_label_style(labels_dir, label_kind)

    names = read_classes(classes_file)
    images = collect_images(images_dir)
    if not images:
        raise RuntimeError("图片目录为空，无法生成数据集")

    labeled = [p for p in images if (labels_dir / f"{p.stem}.txt").exists()]
    if len(labeled) < 2:
        raise RuntimeError("有效标注样本不足(至少2张)")

    train_items, val_items = make_split(labeled, args.val_ratio, args.seed)

    dataset_dir = Path(args.dataset_out)
    if args.rebuild_dataset and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    train_img_dir = dataset_dir / "images" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    train_xml_dir = dataset_dir / "annotations" / "train"
    val_xml_dir = dataset_dir / "annotations" / "val"

    n_train = _build_voc_split(train_items, labels_dir, names, train_img_dir, train_xml_dir)
    n_val = _build_voc_split(val_items, labels_dir, names, val_img_dir, val_xml_dir)

    if n_train == 0 or n_val == 0:
        raise RuntimeError("拆分后训练集或验证集为空，请检查标注质量与 val-ratio")

    classes_out = dataset_dir / "classes.txt"
    classes_out.write_text("\n".join(names) + "\n", encoding="utf-8")
    return dataset_dir, n_train, n_val


def run_train(args):
    if args.dataset_format == "voc_xml":
        dataset_dir, n_train, n_val = build_dataset_voc_xml(args)
        print(f"VOC(XML) 数据集生成完成: train={n_train}, val={n_val}")
        print(f"dataset_dir: {dataset_dir}")
        print("当前训练流程仅支持 YOLO+YAML，已按所选格式完成导出并跳过训练")
        return

    data_yaml, n_train, n_val = build_dataset(args)
    print(f"数据集拆分完成: train={n_train}, val={n_val}")
    print(f"data.yaml: {data_yaml}")

    dataset_out_dir = Path(args.dataset_out).resolve()
    # 训练输出强制落在数据输出目录下，仅保留 best 运行目录。
    args.project = str(dataset_out_dir)
    args.name = "best"

    target_save_dir = Path(args.project) / args.name
    if target_save_dir.exists() and target_save_dir.is_dir():
        shutil.rmtree(target_save_dir)

    _apply_memory_safety(args)

    model = YOLO(args.model)

    def on_fit_epoch_end(trainer):
        try:
            epoch_now = int(getattr(trainer, "epoch", 0)) + 1
            epoch_total = int(getattr(trainer, "epochs", args.epochs))
            tloss = getattr(trainer, "tloss", None)
            loss_val = None
            if tloss is not None:
                try:
                    if hasattr(tloss, "mean"):
                        loss_val = float(tloss.mean())
                    else:
                        vals = list(tloss)
                        if vals:
                            loss_val = float(sum(float(x) for x in vals) / len(vals))
                except Exception:
                    loss_val = None

            metrics = getattr(trainer, "metrics", {}) or {}
            map50 = metrics.get("metrics/mAP50(B)", metrics.get("metrics/mAP50(M)", None))
            map5095 = metrics.get("metrics/mAP50-95(B)", metrics.get("metrics/mAP50-95(M)", None))
            loss_s = f"{loss_val:.6f}" if loss_val is not None else "na"
            map50_s = f"{float(map50):.6f}" if map50 is not None else "na"
            map95_s = f"{float(map5095):.6f}" if map5095 is not None else "na"
            print(
                f"TRAIN_METRIC: epoch={epoch_now}/{epoch_total} loss={loss_s} map50={map50_s} map50_95={map95_s}"
            )
            if (not args.no_safe_memory) and psutil is not None:
                vm = psutil.virtual_memory()
                print(f"TRAIN_RAM: used={vm.percent:.1f}% available_gb={vm.available / (1024.0 ** 3):.2f}")
            if (not args.no_safe_memory) and torch is not None and torch.cuda.is_available():
                try:
                    dev = torch.cuda.current_device()
                    prop = torch.cuda.get_device_properties(dev)
                    total = float(getattr(prop, "total_memory", 0.0) or 0.0)
                    used = float(torch.cuda.memory_reserved(dev))
                    pct = (used / total) * 100.0 if total > 0 else 0.0
                    print(f"TRAIN_VRAM: used={pct:.1f}% reserved_gb={used / (1024.0 ** 3):.2f}")
                except Exception:
                    pass
            gc.collect()
        except Exception:
            pass

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    print(f"开始训练: model={args.model} epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        workers=args.workers,
    )

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"训练完成，结果目录: {save_dir}")
    print(f"最佳模型: {best_pt}")
    print(f"TRAIN_SAVE_DIR: {save_dir}")
    print(f"TRAIN_BEST: {best_pt}")

    # 仅清理训练运行目录，保留数据集目录(images/labels/annotations等)与 best。
    project_dir = Path(args.project)
    keep_dir = project_dir / "best"
    if project_dir.exists() and project_dir.is_dir() and keep_dir.exists():
        for p in project_dir.iterdir():
            if not p.is_dir() or p.resolve() == keep_dir.resolve():
                continue
            name = p.name.lower()
            if name.startswith("exp") or (name.startswith("best") and name != "best"):
                shutil.rmtree(p, ignore_errors=True)

    if args.export_onnx:
        if not best_pt.exists():
            print("未找到 best.pt，跳过ONNX导出")
            return
        print("开始导出ONNX...")
        try:
            _ensure_onnx_requirements_in_current_env()
        except Exception as e:
            raise RuntimeError(
                "ONNX依赖安装/校验失败，请在当前解释器环境手动执行: "
                f"{sys.executable} -m pip install onnx>=1.12.0,<2.0.0 onnxruntime onnxslim>=0.1.71 | 原因: {e}"
            )
        export_model = YOLO(str(best_pt))
        onnx_path = export_model.export(
            format="onnx",
            imgsz=args.export_imgsz or args.imgsz,
            dynamic=True,
            simplify=True,
            opset=args.onnx_opset,
        )
        print(f"ONNX导出完成: {onnx_path}")
        print(f"TRAIN_ONNX: {onnx_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="基于已标注数据集进行训练并可导出ONNX")
    parser.add_argument("--annotation-dir", required=True, help="标注输出目录(含 labels_det/classes.txt)")
    parser.add_argument("--images", required=True, help="原始图片目录")
    parser.add_argument(
        "--label-kind",
        default="auto",
        choices=["auto", "det", "seg"],
        help="标签类型: auto/det/seg。auto 会按模型名优先选择分割标签。",
    )
    parser.add_argument("--dataset-out", default="train_dataset", help="自动拆分后的数据集输出目录")
    parser.add_argument(
        "--dataset-format",
        default="yolo_yaml",
        choices=["yolo_yaml", "voc_xml"],
        help="数据集输出格式: yolo_yaml/voc_xml",
    )
    parser.add_argument("--rebuild-dataset", action="store_true", help="重建 dataset-out 目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--model", default="yoloe-26x-seg.pt", help="训练起始模型")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=1024, help="训练输入尺寸")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--max-ram-percent", type=float, default=88.0, help="内存保护阈值(百分比)")
    parser.add_argument("--max-vram-percent", type=float, default=92.0, help="显存保护阈值(百分比)")
    parser.add_argument("--no-safe-memory", action="store_true", help="关闭内存保护(不建议)")
    parser.add_argument("--device", default="auto", help="训练设备: auto/cpu/0")
    parser.add_argument("--project", default="runs/train", help="训练输出project")
    parser.add_argument("--name", default="exp_auto", help="训练输出name")

    parser.add_argument("--export-onnx", action="store_true", help="训练后导出ONNX")
    parser.add_argument("--export-imgsz", type=int, default=0, help="ONNX导出尺寸(0表示沿用训练imgsz)")
    parser.add_argument("--onnx-opset", type=int, default=12, help="ONNX opset")

    args = parser.parse_args(argv)

    if args.device == "auto":
        use_cuda = bool(torch is not None and torch.cuda.is_available())
        args.device = "0" if use_cuda else "cpu"
        print(f"训练设备自动选择: {args.device} (cuda_available={use_cuda})")

    run_train(args)


if __name__ == "__main__":
    main()
