import argparse
import sys
from pathlib import Path


def run_annotate(rest_args):
    from annotate_dataset import main as annotate_main

    annotate_main(rest_args)


def run_camera(rest_args):
    from notrain import main as camera_main

    camera_main(rest_args)


def run_gui(_rest_args):
    from annotation_qt import main as gui_main

    gui_main()


def run_export(rest_args):
    from export_model_prompts import main as export_main

    export_main(rest_args)


def run_train(rest_args):
    from train_dataset import main as train_main

    train_main(rest_args)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    # Be tolerant to legacy forwarding patterns from older launchers.
    while argv and argv[0] in ("-u", "-m"):
        argv.pop(0)
    if argv and Path(argv[0]).name.lower() in {"main.py", "__main__.py"}:
        argv.pop(0)

    parser = argparse.ArgumentParser(description="YOLO 标注工具统一入口")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["gui", "annotate", "camera", "export-prompts", "train"],
        help="子命令: gui/annotate/camera/export-prompts/train",
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    known = parser.parse_args(argv)
    rest = known.rest

    if known.command is None:
        # 无参数时默认启动 GUI，便于双击/直接运行
        run_gui(rest)
    elif known.command == "gui":
        run_gui(rest)
    elif known.command == "annotate":
        run_annotate(rest)
    elif known.command == "camera":
        run_camera(rest)
    elif known.command == "export-prompts":
        run_export(rest)
    elif known.command == "train":
        run_train(rest)
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
