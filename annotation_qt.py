import os
import re
import shlex
import socket
import subprocess
import sys
import math
import time
from pathlib import Path
from urllib.parse import urlparse

from prompt_expander import suggest_annotation_setup_from_description

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QToolButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None

try:
    import matplotlib as mpl
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MPL = True
except Exception:
    HAS_MPL = False


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class AnnotationWorker(QThread):
    log = Signal(str)
    step = Signal(str)
    progress = Signal(int, int)
    stats = Signal(int, int, int, int, int, float)
    preview = Signal(str)
    llm_prompt = Signal(str)
    paused = Signal(str)
    resumed = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, commands, cwd):
        super().__init__()
        self.commands = commands
        self.cwd = cwd
        self._stop_requested = False
        self._current_process = None
        self._pause_after_image_requested = False
        self._runtime_paused = False

    def stop(self):
        self._stop_requested = True
        if self._current_process and self._current_process.poll() is None:
            self._current_process.terminate()

    def request_pause_after_current(self):
        self._pause_after_image_requested = True

    def cancel_pause_request(self):
        self._pause_after_image_requested = False

    def resume_after_pause(self):
        self._runtime_paused = False

    def pause_now(self):
        if self._runtime_paused:
            return True
        ok = self._suspend_current_process()
        if ok:
            self._runtime_paused = True
            self.paused.emit("任务已暂停")
        return ok

    def resume_now(self):
        if not self._runtime_paused:
            return True
        ok = self._resume_current_process()
        if ok:
            self._runtime_paused = False
            self.resumed.emit("继续运行")
        return ok

    def _suspend_current_process(self):
        if psutil is None or self._current_process is None:
            return False
        try:
            p = psutil.Process(self._current_process.pid)
            children = p.children(recursive=True)
            for c in children:
                c.suspend()
            p.suspend()
            return True
        except Exception:
            return False

    def _resume_current_process(self):
        if psutil is None or self._current_process is None:
            return False
        try:
            p = psutil.Process(self._current_process.pid)
            children = p.children(recursive=True)
            p.resume()
            for c in children:
                c.resume()
            return True
        except Exception:
            return False

    def run(self):
        pattern = re.compile(
            r"进度\s+(\d+)/(\d+)\s+\|\s+det\s+(\d+)\s+\|\s+seg\s+(\d+)\s+\|\s+未识别\s+(\d+)\s+\|\s+失败\s+(\d+)\s+\|\s+平均置信\s+([0-9]*\.?[0-9]+)"
        )
        try:
            total = len(self.commands)
            for idx, cmd in enumerate(self.commands, start=1):
                if self._stop_requested:
                    self.failed.emit("任务已停止")
                    return

                self.progress.emit(idx - 1, total)
                self.log.emit(f"\n===== 模型任务 {idx}/{total} =====")
                self.log.emit("命令: " + " ".join([shlex.quote(c) for c in cmd]))

                self._current_process = subprocess.Popen(
                    cmd,
                    cwd=self.cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                assert self._current_process.stdout is not None
                for line in self._current_process.stdout:
                    text = line.rstrip("\n")
                    self.log.emit(text)

                    if text.startswith("STEP:"):
                        step_text = text.split(":", 1)[1].strip()
                        if step_text:
                            self.step.emit(step_text)

                    if text.startswith("PREVIEW_FILE:"):
                        p = text.split(":", 1)[1].strip()
                        self.preview.emit(p)

                    # 逐词输出保留在日志，不写入分组展示区
                    if text.startswith("LLM_PROMPT:"):
                        pass

                    if text.startswith("LLM_GROUP:"):
                        group_line = text.split(":", 1)[1].strip()
                        if group_line:
                            self.llm_prompt.emit(group_line)

                    if text.startswith("NEG_LLM_PROMPT:"):
                        p = text.split(":", 1)[1].strip()
                        if p:
                            self.llm_prompt.emit("[负向] " + p)

                    if text.startswith("NEG_LLM_GROUP:"):
                        group_line = text.split(":", 1)[1].strip()
                        if group_line:
                            self.llm_prompt.emit("[负向分组] " + group_line)

                    m = pattern.search(text)
                    if m:
                        done = int(m.group(1))
                        total_imgs = int(m.group(2))
                        det = int(m.group(3))
                        seg = int(m.group(4))
                        unknown = int(m.group(5))
                        failed = int(m.group(6))
                        avg_conf = float(m.group(7))
                        _ = failed
                        self.stats.emit(done, total_imgs, det, seg, unknown, avg_conf)
                        if self._pause_after_image_requested and not self._runtime_paused:
                            suspended = self._suspend_current_process()
                            if suspended:
                                self._runtime_paused = True
                                self._pause_after_image_requested = False
                                self.paused.emit("已在当前图片完成后暂停识别")
                                while self._runtime_paused and not self._stop_requested:
                                    time.sleep(0.15)
                                if self._stop_requested:
                                    self._current_process.terminate()
                                    self.failed.emit("任务已停止")
                                    return
                                if self._resume_current_process():
                                    self.resumed.emit("继续识别")
                            else:
                                self._pause_after_image_requested = False
                                self.log.emit("STEP: 暂停请求未生效（缺少进程挂起能力），已继续运行")

                    if self._stop_requested:
                        self._current_process.terminate()
                        self.failed.emit("任务已停止")
                        return

                code = self._current_process.wait()
                self._current_process = None
                if code != 0:
                    self.failed.emit(f"任务失败，退出码: {code}")
                    return

                self.progress.emit(idx, total)

            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))


class PromptAssistantWorker(QThread):
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, description, provider, model, api_base, api_key, timeout):
        super().__init__()
        self.description = description
        self.provider = provider
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout

    def run(self):
        try:
            result = suggest_annotation_setup_from_description(
                user_description=self.description,
                llm_provider=self.provider,
                llm_model=self.model,
                llm_api_base=self.api_base,
                llm_api_key=self.api_key,
                llm_timeout=self.timeout,
            )
            self.done.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能标注工作台")
        self.current_language = "zh"
        self.resize(1600, 900)
        self.setMinimumSize(1200, 750)
        self.workspace = str(Path(__file__).resolve().parent)
        self.worker = None
        self.assistant_worker = None
        self.current_preview_path = ""
        self.preview_paused = False
        self.current_task = "annotate"
        self.train_best_path = ""
        self.train_onnx_path = ""
        self._train_hist_epoch = []
        self._train_hist_progress = []
        self._train_hist_loss = []
        self._train_hist_map50 = []
        self._train_hist_map95 = []
        self._stats_chart_start_ts = None
        self._stats_x_hist = []
        self._stats_fps_hist = []
        self._stats_segavg_hist = []
        self._run_pause_requested = False
        self._run_paused = False
        self._llm_progress_active = False
        self._llm_progress_mode = ""
        self._llm_progress_timer = None
        self._llm_step = 0
        self._llm_total_steps = 5
        self._llm_wait_seconds = 0
        self._invalid_flash_timers = []
        self._mascot_mode = "idle"
        self._mascot_frame_idx = 0
        self._mascot_timer = None
        self._deploy_preview_timer = None
        self._deploy_preview_file = ""
        self._deploy_preview_mtime = -1.0
        self._deploy_stats_start_ts = None
        self._deploy_stats_x_hist = []
        self._deploy_stats_fps_hist = []
        self._deploy_stats_segavg_hist = []
        self.annotate_splitter = None
        self.train_top_splitter = None
        self.deploy_top_splitter = None
        if HAS_MPL:
            self._ensure_mpl_chinese_font()
        self._build_ui()
        self._style()
        QTimer.singleShot(0, self._apply_aspect_layout)
        self._start_motion_effects()
        self.set_status("ready", "就绪")

    def _ensure_mpl_chinese_font(self):
        try:
            mpl.rcParams["font.sans-serif"] = [
                "Microsoft YaHei",
                "SimHei",
                "Source Han Sans CN",
                "PingFang SC",
                "Noto Sans CJK SC",
                "WenQuanYi Micro Hei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ]
            mpl.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

    def _to_rel_display_path(self, p):
        try:
            wp = Path(self.workspace).resolve()
            pp = Path(p).resolve()
            return str(pp.relative_to(wp))
        except Exception:
            return str(p)

    def _tr(self, key):
        zh = {
            "window_title": "智能标注工作台",
            "top_title": "智能标注工作台",
            "language": "语言",
            "status_ready": "就绪",
            "status_running": "运行中",
            "status_error": "异常",
            "step_prefix": "步骤",
            "step_wait": "等待开始",
            "tab_annotate": "标注",
            "tab_train": "训练",
            "tab_deploy": "快速部署",
            "btn_start": "开始",
            "btn_stop": "停止",
            "btn_clear": "清空日志",
            "btn_help": "帮助",
            "btn_pause": "暂停识别",
            "btn_wait_current": "等待当前图完成...",
            "btn_train": "开始训练",
            "btn_deploy_start": "启动摄像头识别",
            "btn_deploy_stop": "停止部署",
            "deploy_idle": "待机",
            "preview_area": "预览区",
            "deploy_preview_area": "部署预览区",
            "footer": "北京建筑大学 @Iisc",
            "prompt_placeholder": "例如: 青苹果",
            "composition_name_placeholder": "统一类别名，例如: 苹果",
            "negative_prompt_placeholder": "例如: 轿车, sedan",
            "scene_desc_placeholder": "正确示例：笔记本电脑，游戏本，带电源\n错误示例：桌子上的笔记本（主题不明，笔记本容易引起异议）",
            "llm_key_placeholder": "请输入API Key",
            "domain_placeholder": "领域提示(可选)",
            "llm_terms_placeholder": "这里显示 LLM 扩展后的提示词",
            "log_placeholder": "这里显示运行日志",
            "train_log_placeholder": "这里显示训练日志",
            "deploy_log_placeholder": "这里显示快速部署日志",
            "deploy_prompt_placeholder": "例如: 笔记本电脑, 电源适配器",
            "deploy_negative_prompt_placeholder": "例如: 手机, 平板",
            "deploy_target_desc_placeholder": "目标描述（可用于智能建议）",
            "label_images": "图片",
            "label_output": "输出",
            "label_target_desc": "目标描述",
            "label_prompt_text": "提示词文本",
            "label_negative_prompt": "负提示词",
            "label_model": "模型",
            "label_conf": "置信度",
            "label_iou": "NMS阈值",
            "label_imgsz": "输入尺寸",
            "label_device": "设备",
            "label_llm_provider": "扩词提供方",
            "label_api_base": "接口地址",
            "label_api_key": "接口密钥",
            "label_timeout": "超时(秒)",
            "label_domain": "领域提示",
            "scene_hint": "建议写法: 目标+背景+角度+排除项。自然语言措辞会直接影响结果。",
            "target_mode": "目标模式",
            "target_mode_single": "单目标",
            "target_mode_multi": "多目标",
            "label_output_mode": "标签输出",
            "label_output_det": "仅检测",
            "label_output_seg": "仅分割",
            "label_output_both": "检测+分割",
            "force_cpu": "仅CPU",
            "train_title": "快速训练",
            "train_subtitle": "简化流程：填路径 -> 设轮次 -> 一键启动",
            "train_trend_title": "训练趋势（4折线）",
            "deploy_title": "快速部署",
            "deploy_subtitle": "摄像头实时识别组合提示词目标（默认启用成分组合）",
            "label_deploy_model": "模型",
            "label_deploy_camera": "摄像头",
            "label_deploy_conf": "置信阈值",
            "label_deploy_imgsz": "输入尺寸",
            "label_deploy_prompt": "提示词",
            "label_deploy_negative": "负提示词",
            "label_deploy_target": "目标描述",
            "label_deploy_component": "组合名称",
            "label_deploy_target_mode": "目标模式",
            "label_chart_granularity": "综合粒度",
            "label_llm_result": "LLM扩词结果",
            "enable_box_expand": "启用框扩张",
            "enable_second_check": "启用二次特性检查",
            "pixel_refine": "像素相关性修正",
            "hard_small_target": "高难小目标模式",
            "enable_llm": "启用API扩词",
            "force_llm": "强制扩词",
            "tooltip_box_expand": "可选补齐目标边缘部位，如鸟类爪子/尾巴/鸟喙",
            "tooltip_second_check": "使用 yoloe-26l-seg.pt 进行二次确认，降低误检",
            "tooltip_pixel_refine": "严格框修正，优先保证目标完整且避免越界到其他物体",
            "tooltip_hard_small": "适合目标占比小且背景干扰大的场景：抑制大面积框并提高精度",
            "train_export_onnx": "训练后导出ONNX",
            "train_pause": "暂停训练",
            "train_resume": "继续训练",
            "train_terminate": "终止训练",
            "train_format_yolo": "YOLO + YAML（可直接训练）",
            "train_format_voc": "Pascal VOC XML（仅导出）",
            "train_format_label": "数据集生成格式",
            "train_format_tip": "提示: 选择 XML 时会先生成 VOC 数据集，当前流程不执行训练",
            "train_ann_dir": "标注目录",
            "train_img_dir": "图片目录",
            "train_data_out": "数据输出",
            "train_model_raw": "训练模型(raw)",
            "train_model_manual": "手动模型",
            "train_refresh": "刷新",
            "train_browse": "浏览",
            "train_epoch": "Epoch",
            "train_batch": "Batch",
            "train_val": "Val",
            "train_max_ram": "最大RAM(%)",
            "train_max_vram": "最大VRAM(%)",
            "stats_auto_reset": "完成后清零统计",
            "stats_reset": "清零统计",
        }
        en = {
            "window_title": "Intelligent Annotation Workbench",
            "top_title": "Intelligent Annotation Workbench",
            "language": "Language",
            "status_ready": "Ready",
            "status_running": "Running",
            "status_error": "Error",
            "step_prefix": "Step",
            "step_wait": "Waiting to start",
            "tab_annotate": "Annotate",
            "tab_train": "Train",
            "tab_deploy": "Quick Deploy",
            "btn_start": "Start",
            "btn_stop": "Stop",
            "btn_clear": "Clear Logs",
            "btn_help": "Help",
            "btn_pause": "Pause Recognition",
            "btn_wait_current": "Waiting for current image...",
            "btn_train": "Start Training",
            "btn_deploy_start": "Start Camera Detection",
            "btn_deploy_stop": "Stop Deploy",
            "deploy_idle": "Idle",
            "preview_area": "Preview",
            "deploy_preview_area": "Deploy Preview",
            "footer": "Beijing University of Civil Engineering and Architecture @Iisc",
            "prompt_placeholder": "e.g. green apple",
            "composition_name_placeholder": "Unified class name, e.g. apple",
            "negative_prompt_placeholder": "e.g. car, sedan",
            "scene_desc_placeholder": "Good: laptop, gaming notebook, with charger\nBad: laptop on table (ambiguous focus)",
            "llm_key_placeholder": "Enter API Key",
            "domain_placeholder": "Domain hint (optional)",
            "llm_terms_placeholder": "LLM expanded prompts appear here",
            "log_placeholder": "Runtime logs appear here",
            "train_log_placeholder": "Training logs appear here",
            "deploy_log_placeholder": "Quick deploy logs appear here",
            "deploy_prompt_placeholder": "e.g. laptop, power adapter",
            "deploy_negative_prompt_placeholder": "e.g. phone, tablet",
            "deploy_target_desc_placeholder": "Target description (for smart suggestion)",
            "label_images": "Images",
            "label_output": "Output",
            "label_target_desc": "Target Description",
            "label_prompt_text": "Prompt Text",
            "label_negative_prompt": "Negative Prompt",
            "label_model": "Model",
            "label_conf": "Confidence",
            "label_iou": "NMS IoU",
            "label_imgsz": "Input Size",
            "label_device": "Device",
            "label_llm_provider": "Expansion Provider",
            "label_api_base": "API Base URL",
            "label_api_key": "API Key",
            "label_timeout": "Timeout (s)",
            "label_domain": "Domain Hint",
            "scene_hint": "Suggested: target + background + angle + exclusions. Wording directly affects results.",
            "target_mode": "Target Mode",
            "target_mode_single": "Single Target",
            "target_mode_multi": "Multi Target",
            "label_output_mode": "Label Output",
            "label_output_det": "Det Only",
            "label_output_seg": "Seg Only",
            "label_output_both": "Det + Seg",
            "force_cpu": "CPU Only",
            "train_title": "Quick Training",
            "train_subtitle": "Simplified flow: set paths -> set epochs -> one-click start",
            "train_trend_title": "Training Trend (4 Curves)",
            "deploy_title": "Quick Deploy",
            "deploy_subtitle": "Real-time camera recognition for combined prompt targets (combine enabled by default)",
            "label_deploy_model": "Model",
            "label_deploy_camera": "Camera",
            "label_deploy_conf": "Confidence Threshold",
            "label_deploy_imgsz": "Input Size",
            "label_deploy_prompt": "Prompt",
            "label_deploy_negative": "Negative Prompt",
            "label_deploy_target": "Target Description",
            "label_deploy_component": "Combined Name",
            "label_deploy_target_mode": "Target Mode",
            "label_chart_granularity": "Granularity",
            "label_llm_result": "LLM Expanded Prompts",
            "enable_box_expand": "Enable Box Expansion",
            "enable_second_check": "Enable Secondary Check",
            "pixel_refine": "Pixel Correlation Refinement",
            "hard_small_target": "Hard Small Target Mode",
            "enable_llm": "Enable API Expansion",
            "force_llm": "Force Expansion",
            "tooltip_box_expand": "Optionally expands boxes to include edge parts such as claws, tails, or beaks.",
            "tooltip_second_check": "Use yoloe-26l-seg.pt for second-pass verification to reduce false positives.",
            "tooltip_pixel_refine": "Apply strict box refinement to keep target complete and avoid crossing to other objects.",
            "tooltip_hard_small": "For tiny targets with noisy backgrounds: suppress large-area boxes and improve precision.",
            "train_export_onnx": "Export ONNX After Training",
            "train_pause": "Pause Training",
            "train_resume": "Resume Training",
            "train_terminate": "Terminate Training",
            "train_format_yolo": "YOLO + YAML (Trainable)",
            "train_format_voc": "Pascal VOC XML (Export Only)",
            "train_format_label": "Dataset Format",
            "train_format_tip": "Tip: XML option exports VOC first; no training is launched in this flow.",
            "train_ann_dir": "Annotation Dir",
            "train_img_dir": "Image Dir",
            "train_data_out": "Data Output",
            "train_model_raw": "Train Model (raw)",
            "train_model_manual": "Manual Model",
            "train_refresh": "Refresh",
            "train_browse": "Browse",
            "train_epoch": "Epoch",
            "train_batch": "Batch",
            "train_val": "Val",
            "train_max_ram": "Max RAM (%)",
            "train_max_vram": "Max VRAM (%)",
            "stats_auto_reset": "Reset Stats After Finish",
            "stats_reset": "Reset Stats",
        }
        table = zh if self.current_language == "zh" else en
        return table.get(key, key)

    def _extract_step_text(self):
        txt = self.step_label.text().strip()
        for p in ("步骤: ", "Step: "):
            if txt.startswith(p):
                return txt[len(p):].strip()
        return txt

    def _on_language_changed(self, _text):
        self.current_language = "en" if self.lang_combo.currentIndex() == 1 else "zh"
        self._apply_language()

    def _apply_language(self):
        self.setWindowTitle(self._tr("window_title"))
        self.title_label.setText(self._tr("top_title"))
        self.lang_label.setText(self._tr("language"))
        self.tabs.setTabText(0, self._tr("tab_annotate"))
        self.tabs.setTabText(1, self._tr("tab_train"))
        self.tabs.setTabText(2, self._tr("tab_deploy"))
        self.footer.setText(self._tr("footer"))

        self.btn_run.setText(self._tr("btn_start"))
        self.btn_stop.setText(self._tr("btn_stop"))
        self.btn_clear.setText(self._tr("btn_clear"))
        self.btn_help.setText(self._tr("btn_help"))
        self.btn_pause_preview.setText(self._tr("btn_pause") if not self._run_pause_requested else self._tr("btn_wait_current"))
        self.btn_train.setText(self._tr("btn_train"))
        if hasattr(self, "btn_train_pause"):
            self.btn_train_pause.setText(self._tr("train_resume") if self._run_paused else self._tr("train_pause"))
        if hasattr(self, "btn_train_stop"):
            self.btn_train_stop.setText(self._tr("train_terminate"))
        self.btn_deploy_start.setText(self._tr("btn_deploy_start"))
        self.btn_deploy_stop.setText(self._tr("btn_deploy_stop"))

        self.preview.setText(self._tr("preview_area"))
        self.deploy_preview.setText(self._tr("deploy_preview_area"))
        self.deploy_state_text.setText(self._tr("deploy_idle"))
        self.lbl_target_mode.setText(self._tr("target_mode"))
        self.target_mode_combo.blockSignals(True)
        cur_target_mode = self.target_mode_combo.currentData()
        self.target_mode_combo.clear()
        self.target_mode_combo.addItem(self._tr("target_mode_single"), "single")
        self.target_mode_combo.addItem(self._tr("target_mode_multi"), "multi")
        idx = 0 if cur_target_mode != "multi" else 1
        self.target_mode_combo.setCurrentIndex(idx)
        self.target_mode_combo.blockSignals(False)
        self.lbl_label_output_mode.setText(self._tr("label_output_mode"))
        self.label_output_mode_combo.blockSignals(True)
        cur_label_mode = self.label_output_mode_combo.currentData()
        self.label_output_mode_combo.clear()
        self.label_output_mode_combo.addItem(self._tr("label_output_both"), "both")
        self.label_output_mode_combo.addItem(self._tr("label_output_det"), "det")
        self.label_output_mode_combo.addItem(self._tr("label_output_seg"), "seg")
        mode_idx = {"both": 0, "det": 1, "seg": 2}.get(cur_label_mode, 0)
        self.label_output_mode_combo.setCurrentIndex(mode_idx)
        self.label_output_mode_combo.blockSignals(False)
        self.force_cpu.setText(self._tr("force_cpu"))
        self.enable_box_expand.setText(self._tr("enable_box_expand"))
        self.enable_second_check.setText(self._tr("enable_second_check"))
        self.pixel_refine.setText(self._tr("pixel_refine"))
        self.hard_small_target.setText(self._tr("hard_small_target"))
        self.enable_llm.setText(self._tr("enable_llm"))
        self.force_llm.setText(self._tr("force_llm"))
        self.enable_box_expand.setToolTip(self._tr("tooltip_box_expand"))
        self.enable_second_check.setToolTip(self._tr("tooltip_second_check"))
        self.pixel_refine.setToolTip(self._tr("tooltip_pixel_refine"))
        self.hard_small_target.setToolTip(self._tr("tooltip_hard_small"))

        self.lbl_images.setText(self._tr("label_images"))
        self.lbl_output.setText(self._tr("label_output"))
        self.lbl_target_desc.setText(self._tr("label_target_desc"))
        self.lbl_prompt_text.setText(self._tr("label_prompt_text"))
        self.lbl_negative_prompt.setText(self._tr("label_negative_prompt"))
        self.lbl_model.setText(self._tr("label_model"))
        self.lbl_conf.setText(self._tr("label_conf"))
        self.lbl_iou.setText(self._tr("label_iou"))
        self.lbl_imgsz.setText(self._tr("label_imgsz"))
        self.lbl_device.setText(self._tr("label_device"))
        self.lbl_llm_provider.setText(self._tr("label_llm_provider"))
        self.lbl_api_base.setText(self._tr("label_api_base"))
        self.lbl_api_key.setText(self._tr("label_api_key"))
        self.lbl_timeout.setText(self._tr("label_timeout"))
        self.lbl_domain.setText(self._tr("label_domain"))
        self.scene_hint_label.setText(self._tr("scene_hint"))

        self.train_title_label.setText(self._tr("train_title"))
        self.train_subtitle_label.setText(self._tr("train_subtitle"))
        self.train_trend_title_label.setText(self._tr("train_trend_title"))
        self.lbl_train_format.setText(self._tr("train_format_label"))
        self.train_format_tip_label.setText(self._tr("train_format_tip"))
        self.lbl_train_ann_dir.setText(self._tr("train_ann_dir"))
        self.lbl_train_img_dir.setText(self._tr("train_img_dir"))
        self.lbl_train_data_out.setText(self._tr("train_data_out"))
        self.lbl_train_model_raw.setText(self._tr("train_model_raw"))
        self.lbl_train_model_manual.setText(self._tr("train_model_manual"))
        self.btn_refresh_models.setText(self._tr("train_refresh"))
        self.btn_pick_ann.setText(self._tr("train_browse"))
        self.btn_pick_img.setText(self._tr("train_browse"))
        self.btn_pick_dataset_out.setText(self._tr("train_browse"))
        self.btn_pick_train_model.setText(self._tr("train_browse"))
        self.lbl_train_epoch.setText(self._tr("train_epoch"))
        self.lbl_train_batch.setText(self._tr("train_batch"))
        self.lbl_train_val.setText(self._tr("train_val"))
        if hasattr(self, "lbl_train_max_ram"):
            self.lbl_train_max_ram.setText(self._tr("train_max_ram"))
        if hasattr(self, "lbl_train_max_vram"):
            self.lbl_train_max_vram.setText(self._tr("train_max_vram"))
        self.train_export_onnx.setText(self._tr("train_export_onnx"))
        cur_fmt_idx = self.train_dataset_format.currentIndex()
        self.train_dataset_format.blockSignals(True)
        self.train_dataset_format.clear()
        self.train_dataset_format.addItems([self._tr("train_format_yolo"), self._tr("train_format_voc")])
        self.train_dataset_format.setCurrentIndex(max(0, min(cur_fmt_idx, self.train_dataset_format.count() - 1)))
        self.train_dataset_format.blockSignals(False)
        self.deploy_title_label.setText(self._tr("deploy_title"))
        self.deploy_subtitle_label.setText(self._tr("deploy_subtitle"))
        self.lbl_deploy_model.setText(self._tr("label_deploy_model"))
        self.lbl_deploy_camera.setText(self._tr("label_deploy_camera"))
        self.lbl_deploy_conf.setText(self._tr("label_deploy_conf"))
        self.lbl_deploy_imgsz.setText(self._tr("label_deploy_imgsz"))
        self.lbl_deploy_prompt.setText(self._tr("label_deploy_prompt"))
        self.lbl_deploy_negative.setText(self._tr("label_deploy_negative"))
        self.lbl_deploy_target.setText(self._tr("label_deploy_target"))
        self.lbl_deploy_component.setText(self._tr("label_deploy_component"))
        self.lbl_deploy_target_mode.setText(self._tr("label_deploy_target_mode"))
        self.deploy_target_mode_combo.blockSignals(True)
        cur_deploy_mode = self.deploy_target_mode_combo.currentData()
        self.deploy_target_mode_combo.clear()
        self.deploy_target_mode_combo.addItem(self._tr("target_mode_single"), "single")
        self.deploy_target_mode_combo.addItem(self._tr("target_mode_multi"), "multi")
        self.deploy_target_mode_combo.setCurrentIndex(0 if cur_deploy_mode != "multi" else 1)
        self.deploy_target_mode_combo.blockSignals(False)
        self._on_deploy_target_mode_changed(self.deploy_target_mode_combo.currentIndex())
        self.lbl_chart_granularity_runtime.setText(self._tr("label_chart_granularity"))
        self.lbl_chart_granularity_deploy.setText(self._tr("label_chart_granularity"))
        self.lbl_llm_result_title.setText(self._tr("label_llm_result"))
        self._update_train_memory_info()
        self.auto_reset_stats.setText(self._tr("stats_auto_reset"))
        self.btn_reset_stats.setText(self._tr("stats_reset"))

        self.prompt_text_edit.setPlaceholderText(self._tr("prompt_placeholder"))
        self.composition_name_edit.setPlaceholderText(self._tr("composition_name_placeholder"))
        self.negative_prompt_text_edit.setPlaceholderText(self._tr("negative_prompt_placeholder"))
        self.scene_desc_edit.setPlaceholderText(self._tr("scene_desc_placeholder"))
        self.llm_api_key.setPlaceholderText(self._tr("llm_key_placeholder"))
        self.domain.setPlaceholderText(self._tr("domain_placeholder"))
        self.llm_terms_view.setPlaceholderText(self._tr("llm_terms_placeholder"))
        self.log.setPlaceholderText(self._tr("log_placeholder"))
        self.train_log.setPlaceholderText(self._tr("train_log_placeholder"))
        self.deploy_log.setPlaceholderText(self._tr("deploy_log_placeholder"))
        self.deploy_prompt_edit.setPlaceholderText(self._tr("deploy_prompt_placeholder"))
        self.deploy_negative_prompt_edit.setPlaceholderText(self._tr("deploy_negative_prompt_placeholder"))
        self.deploy_target_desc_edit.setPlaceholderText(self._tr("deploy_target_desc_placeholder"))

        step_body = self._extract_step_text() or self._tr("step_wait")
        self.step_label.setText(f"{self._tr('step_prefix')}: {step_body}")

        status_now = self.status_badge.text().strip().upper()
        if status_now == "RUNNING":
            self.status_label.setText(self._tr("status_running"))
        elif status_now == "ERROR":
            self.status_label.setText(self._tr("status_error"))
        else:
            self.status_label.setText(self._tr("status_ready"))

    def _resolve_input_path(self, text):
        t = str(text or "").strip()
        if not t:
            return Path(self.workspace)
        p = Path(t)
        if p.is_absolute():
            return p
        return Path(self.workspace) / p

    def _message_box_stylesheet(self):
        return (
            "QMessageBox{background:#ffffff;color:#0f172a;}"
            "QLabel{background:#ffffff;color:#0f172a;}"
            "QPushButton{background:#0f4c81;color:#ffffff;border:none;border-radius:8px;padding:6px 12px;font-weight:700;}"
            "QPushButton:hover{background:#165d9c;}"
        )

    def _show_info(self, title, text):
        msg = QMessageBox(self)
        msg.setWindowTitle(str(title))
        msg.setIcon(QMessageBox.Information)
        msg.setText(str(text))
        msg.setStyleSheet(self._message_box_stylesheet())
        msg.exec()

    def _show_warning(self, title, text):
        msg = QMessageBox(self)
        msg.setWindowTitle(str(title))
        msg.setIcon(QMessageBox.Warning)
        msg.setText(str(text))
        msg.setStyleSheet(self._message_box_stylesheet())
        msg.exec()

    def _refresh_widget_style(self, widget):
        if widget is None:
            return
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _set_invalid(self, widget, invalid=True, blink=False):
        if widget is None:
            return
        widget.setProperty("invalid", "true" if invalid else "false")
        widget.setProperty("invalidBlink", "true" if (invalid and blink) else "false")
        self._refresh_widget_style(widget)

    def _flash_invalid_widgets(self, widgets):
        uniq = []
        for w in widgets:
            if w is not None and w not in uniq:
                uniq.append(w)
        if not uniq:
            return

        for w in uniq:
            self._set_invalid(w, invalid=True, blink=False)

        timer = QTimer(self)
        timer.setInterval(130)
        state = {"ticks": 0}

        def _tick():
            state["ticks"] += 1
            on = (state["ticks"] % 2) == 1
            for w in uniq:
                self._set_invalid(w, invalid=True, blink=on)
            if state["ticks"] >= 8:
                timer.stop()
                for w in uniq:
                    self._set_invalid(w, invalid=True, blink=False)
                try:
                    self._invalid_flash_timers.remove(timer)
                except ValueError:
                    pass

        timer.timeout.connect(_tick)
        timer.start()
        self._invalid_flash_timers.append(timer)

    def _clear_invalid_marks(self):
        fields = [
            self.images_edit,
            self.output_edit,
            self.prompt_text_edit,
            self.composition_name_edit,
            self.scene_desc_edit,
            self.model_list,
            self.deploy_model_edit,
            self.deploy_prompt_edit,
            self.deploy_negative_prompt_edit,
            self.deploy_target_desc_edit,
            self.deploy_component_name,
            self.train_ann_edit,
            self.train_images_edit,
            self.train_model_edit,
        ]
        for w in fields:
            self._set_invalid(w, invalid=False, blink=False)

    def _filter_overextended_prompt_text(self, prompt_text, scene_desc):
        terms = [t.strip() for t in re.split(r"[\n,，;；、|]+", str(prompt_text or "")) if t.strip()]
        if not terms:
            return ""

        desc = str(scene_desc or "").strip().lower()
        generic_terms = {
            "电子设备", "设备", "物体", "目标", "产品", "工具", "用品",
            "electronic device", "device", "object", "target", "product", "tool", "item",
        }
        kept = []
        dropped = []
        for term in terms:
            key = term.strip().lower()
            if key in generic_terms and key not in desc:
                dropped.append(term)
                continue
            kept.append(term)

        if not kept:
            kept = terms[:1]
        if dropped:
            self.append_log("STEP: 智能建议 - 已过滤泛化词: " + ", ".join(dropped))
        return ", ".join(kept)

    def _get_mascot_frames(self):
        return {
            "idle": ["  o  \n /|\\ \n / \\", "  o  \n \\|/ \n / \\"],
            "running": ["  o  \n /|_ \n / >", "  o  \n _|\\ \n< \\", "  o  \n /|_ \n / >"],
            "done": [" \\o/ \n  |  \n / \\", " \\o/ \n  |  \n _/\\_"],
            "error": ["  o  \n /|\\ \n /_\\", "  o  \n /|\\ \n / \\"],
        }

    def _set_mascot_mode(self, mode, step_text=None):
        mode = str(mode or "idle")
        if mode not in {"idle", "running", "done", "error"}:
            mode = "idle"
        if self._mascot_mode != mode:
            self._mascot_mode = mode
            self._mascot_frame_idx = 0
        if hasattr(self, "mascot_label"):
            frames = self._get_mascot_frames().get(self._mascot_mode, ["  o  \n /|\\ \n / \\"])
            if frames:
                self.mascot_label.setText(frames[self._mascot_frame_idx % len(frames)])
        if step_text is not None and hasattr(self, "mascot_step_label"):
            self.mascot_step_label.setText(str(step_text))

    def _tick_mascot(self):
        if not hasattr(self, "mascot_label"):
            return
        frames = self._get_mascot_frames().get(self._mascot_mode, ["  o  \n /|\\ \n / \\"])
        if not frames:
            return
        self._mascot_frame_idx = (self._mascot_frame_idx + 1) % len(frames)
        self.mascot_label.setText(frames[self._mascot_frame_idx])

    def _update_mascot_from_step(self, step_text):
        s = str(step_text or "").strip()
        if not s:
            return
        sl = s.lower()
        if ("异常" in s) or ("失败" in s) or ("中断" in s) or ("error" in sl):
            self._set_mascot_mode("error", s)
            return
        if ("完成" in s) or ("已完成" in s) or ("done" in sl):
            self._set_mascot_mode("done", s)
            return
        if ("暂停" in s) or ("已暂停" in s):
            self._set_mascot_mode("idle", s)
            return
        # 识别主流程阶段：读取图像/模型推理/写入标注/渲染预览/进度更新
        if (
            ("读取图像" in s)
            or ("模型推理" in s)
            or ("写入标注" in s)
            or ("渲染预览" in s)
            or ("进度" in s)
            or ("标注" in s)
            or ("识别" in s)
        ):
            self._set_mascot_mode("running", s)
            return
        # 其他文案默认跟随运行态，保证小人持续反映“在处理”。
        self._set_mascot_mode("running", s)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.title_label = QLabel("智能标注工作台")
        self.title_label.setObjectName("titleText")
        top.addWidget(self.title_label)
        top.addStretch(1)
        self.lang_label = QLabel("语言")
        self.lang_combo = NoWheelComboBox()
        self.lang_combo.addItems(["中文", "English"])
        self.lang_combo.setCurrentIndex(0)
        self.lang_combo.currentTextChanged.connect(self._on_language_changed)
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusText")
        self.step_label = QLabel("步骤: 等待开始")
        self.step_label.setObjectName("stepText")
        self.status_badge = QLabel("READY")
        self.status_badge.setObjectName("statusBadge")
        top.addWidget(self.lang_label)
        top.addWidget(self.lang_combo)
        top.addWidget(self.status_label)
        top.addWidget(self.step_label)
        top.addWidget(self.status_badge)
        layout.addLayout(top)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.footer = QLabel("北京建筑大学 @Iisc")
        self.footer.setObjectName("footerText")
        self.footer.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.footer)

        annotate_page = QWidget()
        annotate_page_layout = QVBoxLayout(annotate_page)

        train_page = QWidget()
        train_page_layout = QVBoxLayout(train_page)

        deploy_page = QWidget()
        deploy_page_layout = QVBoxLayout(deploy_page)

        self.tabs.addTab(annotate_page, "标注")
        self.tabs.addTab(train_page, "训练")
        self.tabs.addTab(deploy_page, "快速部署")

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        self.annotate_splitter = splitter

        left_content = QWidget()
        left_content.setObjectName("panelCard")
        left_content.setMinimumWidth(470)
        left_l = QVBoxLayout(left_content)
        left_l.setContentsMargins(12, 12, 12, 12)
        left_l.setSpacing(10)
        form = QGridLayout()

        self.images_edit = QLineEdit()
        self.output_edit = QLineEdit("dataset_annotations")
        self.prompt_text_edit = QLineEdit()
        self.prompt_text_edit.setPlaceholderText("例如: 青苹果")
        self.lbl_target_mode = QLabel("目标模式")
        self.target_mode_combo = NoWheelComboBox()
        self.target_mode_combo.addItem("单目标", "single")
        self.target_mode_combo.addItem("多目标", "multi")
        self.target_mode_combo.setCurrentIndex(0)
        self.lbl_label_output_mode = QLabel("标签输出")
        self.label_output_mode_combo = NoWheelComboBox()
        self.label_output_mode_combo.addItem("检测+分割", "both")
        self.label_output_mode_combo.addItem("仅检测", "det")
        self.label_output_mode_combo.addItem("仅分割", "seg")
        self.label_output_mode_combo.setCurrentIndex(0)
        self.composition_name_edit = QLineEdit()
        self.composition_name_edit.setPlaceholderText("统一类别名，例如: 苹果")
        self.negative_prompt_text_edit = QLineEdit()
        self.negative_prompt_text_edit.setPlaceholderText("例如: 轿车, sedan")
        self.scene_desc_edit = QPlainTextEdit()
        self.scene_desc_edit.setPlaceholderText(
            "正确示例：笔记本电脑，游戏本，带电源\n"
            "错误示例：桌子上的笔记本（主题不明，笔记本容易引起异议）"
        )
        self.scene_desc_edit.setToolTip(
            "建议包含: 目标主体 + 场景背景 + 拍摄角度 + 需排除对象。\n"
            "自然语言措辞会影响扩词和识别结果。"
        )
        self.scene_desc_edit.setMaximumHeight(88)
        self.model_edit = QLineEdit("yoloe-26x-seg.pt")
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(96)
        self.model_list.addItem(QListWidgetItem(self.model_edit.text()))

        btn_img = QPushButton("图片目录")
        btn_out = QPushButton("输出目录")
        btn_model = QPushButton("模型文件")
        btn_add = QPushButton("添加模型")
        btn_del = QPushButton("删除模型")
        btn_suggest = QPushButton("智能建议")
        btn_img.clicked.connect(self.pick_images)
        btn_out.clicked.connect(self.pick_output)
        btn_model.clicked.connect(self.pick_model)
        btn_add.clicked.connect(self.add_model)
        btn_del.clicked.connect(self.remove_model)
        btn_suggest.clicked.connect(self.run_prompt_assistant)

        composition_row = QWidget()
        composition_row_l = QHBoxLayout(composition_row)
        composition_row_l.setContentsMargins(0, 0, 0, 0)
        composition_row_l.setSpacing(4)
        composition_row_l.addWidget(self.lbl_target_mode)
        composition_row_l.addWidget(self.target_mode_combo)
        composition_row_l.addWidget(self.lbl_label_output_mode)
        composition_row_l.addWidget(self.label_output_mode_combo)
        composition_help_btn = QToolButton()
        composition_help_btn.setText("?")
        composition_help_btn.setObjectName("hintButton")
        composition_help_btn.setToolTip("点击查看示例图与说明")
        composition_help_btn.setAutoRaise(True)
        composition_help_btn.clicked.connect(
            lambda _=False: self.show_feature_help(
                "单/多目标模式",
                "成分组合.png",
                "单目标: 所有提示词标注统一写入同一类别名，更适合应对高难非典型目标。\n"
                "多目标: 保留每个提示词对应类别，并在LLM扩词后自动去重避免重叠提示词。",
            )
        )
        composition_row_l.addWidget(composition_help_btn)
        composition_row_l.addStretch(1)

        self.lbl_images = QLabel("图片")
        form.addWidget(self.lbl_images, 0, 0)
        form.addWidget(self.images_edit, 0, 1)
        form.addWidget(btn_img, 0, 2)
        self.lbl_output = QLabel("输出")
        form.addWidget(self.lbl_output, 1, 0)
        form.addWidget(self.output_edit, 1, 1)
        form.addWidget(btn_out, 1, 2)
        self.lbl_target_desc = QLabel("目标描述")
        form.addWidget(self.lbl_target_desc, 2, 0)
        form.addWidget(self.scene_desc_edit, 2, 1)
        form.addWidget(btn_suggest, 2, 2)
        self.scene_hint_label = QLabel("建议写法: 目标+背景+角度+排除项。自然语言措辞会直接影响结果。")
        self.scene_hint_label.setObjectName("trainModelInfo")
        self.scene_hint_label.setWordWrap(True)
        form.addWidget(self.scene_hint_label, 3, 1, 1, 2)
        self.lbl_prompt_text = QLabel("提示词文本")
        form.addWidget(self.lbl_prompt_text, 4, 0)
        form.addWidget(self.prompt_text_edit, 4, 1, 1, 2)
        form.addWidget(composition_row, 5, 1)
        form.addWidget(self.composition_name_edit, 5, 2)
        self.lbl_negative_prompt = QLabel("负提示词")
        form.addWidget(self.lbl_negative_prompt, 6, 0)
        form.addWidget(self.negative_prompt_text_edit, 6, 1, 1, 2)
        self.lbl_model = QLabel("模型")
        form.addWidget(self.lbl_model, 7, 0)
        form.addWidget(self.model_edit, 7, 1)
        form.addWidget(btn_model, 7, 2)

        model_row = QHBoxLayout()
        model_row.addWidget(btn_add)
        model_row.addWidget(btn_del)

        params = QGridLayout()
        self.conf = NoWheelDoubleSpinBox()
        self.conf.setRange(0.001, 1.0)
        self.conf.setValue(0.40)
        self.conf.setDecimals(3)
        self.iou = NoWheelDoubleSpinBox()
        self.iou.setRange(0.1, 1.0)
        self.iou.setValue(0.6)
        self.iou.setDecimals(3)
        self.imgsz = NoWheelSpinBox()
        self.imgsz.setRange(320, 4096)
        self.imgsz.setValue(1024)
        self.device = NoWheelComboBox()
        self.device.addItems(["auto", "cpu", "cuda:0"])
        self.force_cpu = QCheckBox("仅CPU")
        self.force_cpu.setToolTip("开启后强制使用CPU推理，并忽略GPU选择")
        self.force_cpu.toggled.connect(self._on_force_cpu_toggled)
        self.device.currentTextChanged.connect(lambda _t: self._update_train_memory_info())

        self.enable_box_expand = QCheckBox("启用框扩张")
        self.enable_box_expand.setChecked(False)
        self.enable_box_expand.setToolTip("可选补齐目标边缘部位，如鸟类爪子/尾巴/鸟喙")
        self.enable_second_check = QCheckBox("启用二次特性检查")
        self.enable_second_check.setChecked(True)
        self.enable_second_check.setToolTip("使用 yoloe-26l-seg.pt 进行二次确认，降低误检")
        self.pixel_refine = QCheckBox("像素相关性修正")
        self.pixel_refine.setChecked(True)
        self.pixel_refine.setToolTip("严格框修正，优先保证目标完整且避免越界到其他物体")
        self.hard_small_target = QCheckBox("高难小目标模式")
        self.hard_small_target.setChecked(False)
        self.hard_small_target.setToolTip("适合目标占比小且背景干扰大的场景：抑制大面积框并提高精度")
        self.enable_llm = QCheckBox("启用API扩词")
        self.enable_llm.setChecked(True)
        self.force_llm = QCheckBox("强制扩词")
        self.force_llm.setChecked(True)
        self.llm_provider = NoWheelComboBox()
        self.llm_provider.addItems(["openai-compatible", "ollama"])
        self.llm_provider.setCurrentText("openai-compatible")
        self.llm_model = QLineEdit("deepseek-chat")
        default_api_base = os.getenv("SLB_DEFAULT_API_BASE", "")
        default_api_key = os.getenv("SLB_DEFAULT_API_KEY", "")
        self.llm_base = QLineEdit(default_api_base)
        self.llm_api_key = QLineEdit()
        self.llm_api_key.setEchoMode(QLineEdit.Password)
        self.llm_api_key.setText(default_api_key)
        self.llm_api_key.setPlaceholderText("请输入API Key")
        self.llm_timeout = NoWheelSpinBox()
        self.llm_timeout.setRange(5, 300)
        self.llm_timeout.setValue(80)
        self.domain = QLineEdit()
        self.domain.setPlaceholderText("领域提示(可选)")

        def _with_help(check: QCheckBox, title: str, image_name: str, tip: str):
            wrap = QWidget()
            row = QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(check)
            q = QToolButton()
            q.setText("?")
            q.setObjectName("hintButton")
            q.setToolTip("点击查看示例图与说明")
            q.setAutoRaise(True)
            q.clicked.connect(lambda _=False, t=title, img=image_name, d=tip: self.show_feature_help(t, img, d))
            row.addWidget(q)
            row.addStretch(1)
            return wrap

        w_box_expand = _with_help(
            self.enable_box_expand,
            "框扩张",
            "框扩张.png",
            "框扩张: 在检测框基础上外扩以补齐边缘部位。\n"
            "适合目标局部被截断的情况；过大可能引入背景。\n"
            "默认不开启，除非置信度很高。",
        )
        w_pixel_refine = _with_help(
            self.pixel_refine,
            "像素相关性修正",
            "像素相关性.png",
            "像素相关性修正: 利用颜色/连通性约束修正框位置与范围。\n"
            "用于减少牵线和跨目标串联。",
        )
        w_second_check = _with_help(
            self.enable_second_check,
            "二次特性检查",
            "二次特性.png",
            "二次特性检查: 使用 yoloe-26l-seg.pt 再确认候选结果。\n"
            "能降低误检，但会增加推理时间。",
        )
        w_hard_small = _with_help(
            self.hard_small_target,
            "高难小目标模式",
            "高难小目标.png",
            "高难小目标模式: 优先保留小目标并抑制大面积区域。\n"
            "若没有有效小目标，会在候选区域内触发一次补检。",
        )
        self.lbl_conf = QLabel("置信度")
        params.addWidget(self.lbl_conf, 0, 0)
        params.addWidget(self.conf, 0, 1)
        self.lbl_iou = QLabel("NMS阈值")
        params.addWidget(self.lbl_iou, 0, 2)
        params.addWidget(self.iou, 0, 3)
        self.lbl_imgsz = QLabel("输入尺寸")
        params.addWidget(self.lbl_imgsz, 1, 0)
        params.addWidget(self.imgsz, 1, 1)
        self.lbl_device = QLabel("设备")
        params.addWidget(self.lbl_device, 1, 2)
        device_wrap = QWidget()
        device_layout = QHBoxLayout(device_wrap)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(8)
        device_layout.addWidget(self.device)
        device_layout.addWidget(self.force_cpu)
        params.addWidget(device_wrap, 1, 3)
        params.addWidget(w_box_expand, 2, 0)
        params.addWidget(w_pixel_refine, 2, 1)
        params.addWidget(w_hard_small, 2, 2)
        params.addWidget(w_second_check, 3, 0)
        params.addWidget(self.enable_llm, 3, 1)
        params.addWidget(self.force_llm, 3, 2)
        self.lbl_llm_provider = QLabel("扩词提供方")
        params.addWidget(self.lbl_llm_provider, 4, 0)
        params.addWidget(self.llm_provider, 4, 1)
        params.addWidget(self.llm_model, 4, 2, 1, 2)
        self.lbl_api_base = QLabel("接口地址")
        params.addWidget(self.lbl_api_base, 5, 0)
        params.addWidget(self.llm_base, 5, 1, 1, 3)
        self.lbl_api_key = QLabel("接口密钥")
        params.addWidget(self.lbl_api_key, 6, 0)
        params.addWidget(self.llm_api_key, 6, 1, 1, 3)
        self.lbl_timeout = QLabel("超时(秒)")
        params.addWidget(self.lbl_timeout, 7, 0)
        params.addWidget(self.llm_timeout, 7, 1)
        self.lbl_domain = QLabel("领域提示")
        params.addWidget(self.lbl_domain, 8, 0)
        params.addWidget(self.domain, 8, 1, 1, 3)

        action = QHBoxLayout()
        self.btn_run = QPushButton("开始")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_clear = QPushButton("清空日志")
        self.btn_help = QPushButton("帮助")
        self.btn_run.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_clear.clicked.connect(self.clear_output)
        self.btn_help.clicked.connect(self.show_help)
        action.addWidget(self.btn_run)
        action.addWidget(self.btn_stop)
        action.addWidget(self.btn_clear)
        action.addWidget(self.btn_help)

        train_menu = QGridLayout()
        self.train_ann_edit = QLineEdit("dataset_annotations")
        self.train_images_edit = QLineEdit()
        self.train_dataset_out_edit = QLineEdit("")
        self.train_dataset_out_edit.setPlaceholderText("train_dataset")
        self.train_model_edit = QLineEdit("raw\\yolo26l.pt")
        self.train_model_combo = NoWheelComboBox()
        self.train_model_combo.setMinimumWidth(260)
        self.train_model_info = QLabel("模型参数: -")
        self.train_model_info.setWordWrap(True)
        self.train_model_info.setObjectName("trainModelInfo")
        self.train_epochs = NoWheelSpinBox()
        self.train_epochs.setRange(1, 2000)
        self.train_epochs.setValue(50)
        self.train_batch = NoWheelSpinBox()
        self.train_batch.setRange(1, 256)
        self.train_batch.setValue(8)
        self.train_val_ratio = NoWheelDoubleSpinBox()
        self.train_val_ratio.setRange(0.05, 0.5)
        self.train_val_ratio.setDecimals(2)
        self.train_val_ratio.setSingleStep(0.05)
        self.train_val_ratio.setValue(0.2)
        self.train_max_ram = NoWheelDoubleSpinBox()
        self.train_max_ram.setRange(50.0, 98.0)
        self.train_max_ram.setDecimals(1)
        self.train_max_ram.setSingleStep(1.0)
        self.train_max_ram.setValue(88.0)
        self.train_max_vram = NoWheelDoubleSpinBox()
        self.train_max_vram.setRange(50.0, 98.0)
        self.train_max_vram.setDecimals(1)
        self.train_max_vram.setSingleStep(1.0)
        self.train_max_vram.setValue(92.0)
        self.train_memory_info = QLabel("本机内存: RAM - | VRAM -")
        self.train_memory_info.setObjectName("trainModelInfo")
        self.train_export_onnx = QCheckBox("训练后导出ONNX")
        self.train_export_onnx.setChecked(True)
        self.train_dataset_format = NoWheelComboBox()
        self.train_dataset_format.addItems([
            "YOLO + YAML（可直接训练）",
            "Pascal VOC XML（仅导出）",
        ])
        self.train_dataset_format.setCurrentIndex(0)
        self.btn_train = QPushButton("开始训练(二级菜单)")
        self.btn_train.clicked.connect(self.start_train)
        self.btn_train_pause = QPushButton("暂停训练")
        self.btn_train_pause.setEnabled(False)
        self.btn_train_pause.clicked.connect(self.toggle_train_pause)
        self.btn_train_stop = QPushButton("终止训练")
        self.btn_train_stop.setEnabled(False)
        self.btn_train_stop.clicked.connect(self.stop_train_task)

        self.btn_pick_ann = QPushButton("标注目录")
        self.btn_pick_img = QPushButton("图片目录")
        self.btn_pick_dataset_out = QPushButton("数据集输出")
        self.btn_pick_train_model = QPushButton("训练模型")
        for b in [self.btn_pick_ann, self.btn_pick_img, self.btn_pick_dataset_out, self.btn_pick_train_model]:
            b.setText("浏览")
            b.setObjectName("miniBtn")
            b.setMinimumWidth(72)
        self.btn_pick_ann.clicked.connect(self.pick_train_annotation_dir)
        self.btn_pick_img.clicked.connect(self.pick_train_images_dir)
        self.btn_pick_dataset_out.clicked.connect(self.pick_train_dataset_out)
        self.btn_pick_train_model.clicked.connect(self.pick_train_model)
        self.target_mode_combo.currentIndexChanged.connect(self._on_target_mode_changed)
        self._on_target_mode_changed(self.target_mode_combo.currentIndex())

        left_l.addLayout(form)
        left_l.addLayout(model_row)
        left_l.addWidget(self.model_list)
        left_l.addLayout(params)
        left_l.addLayout(action)

        right = QWidget()
        right.setObjectName("panelCard")
        right.setMinimumWidth(520)
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(12, 12, 12, 12)
        right_l.setSpacing(10)
        preview_toolbar = QHBoxLayout()
        self.btn_pause_preview = QPushButton("暂停识别")
        self.btn_pause_preview.clicked.connect(self.toggle_preview_pause)
        preview_toolbar.addWidget(self.btn_pause_preview)
        preview_toolbar.addStretch(1)
        self.preview = QLabel("预览区")
        self.preview.setObjectName("previewFrame")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(360)
        self.preview.setScaledContents(False)

        stats = QGridLayout()
        stats.setHorizontalSpacing(10)
        stats.setVerticalSpacing(8)
        stats.setColumnStretch(0, 6)
        stats.setColumnStretch(1, 3)
        stats.setColumnStretch(2, 3)
        stats.setRowStretch(0, 3)
        stats.setRowStretch(1, 1)

        chart_wrap = QWidget()
        chart_wrap_l = QVBoxLayout(chart_wrap)
        chart_wrap_l.setContentsMargins(0, 0, 0, 0)
        chart_wrap_l.setSpacing(4)
        chart_head = QHBoxLayout()
        chart_head.setContentsMargins(0, 0, 0, 0)
        chart_head.setSpacing(6)
        chart_title = QLabel("柱状图: 平均帧数FPS(蓝) / 平均分割数(橙)")
        chart_title.setObjectName("statsChartTitle")
        chart_head.addWidget(chart_title)
        chart_head.addStretch(1)
        self.lbl_chart_granularity_runtime = QLabel("综合粒度")
        chart_head.addWidget(self.lbl_chart_granularity_runtime)
        self.stats_window_combo = NoWheelComboBox()
        self.stats_window_combo.addItems(["每1张综合", "每5张综合", "每20张综合", "每100张综合"])
        self.stats_window_combo.setCurrentText("每1张综合")
        self.stats_window_combo.currentTextChanged.connect(lambda _t: self._update_runtime_stats_chart())
        chart_head.addWidget(self.stats_window_combo)
        chart_wrap_l.addLayout(chart_head)

        if HAS_MPL:
            self.stats_fig = Figure(figsize=(5.2, 2.3), tight_layout=True)
            self.stats_ax = self.stats_fig.add_subplot(111)
            self.stats_canvas = FigureCanvas(self.stats_fig)
            self.stats_canvas.setMinimumHeight(170)
            chart_wrap_l.addWidget(self.stats_canvas)
        else:
            self.stats_canvas = None
            self.stats_ax = None
            fb = QLabel("未检测到 matplotlib，无法显示统计柱状图")
            fb.setAlignment(Qt.AlignCenter)
            fb.setMinimumHeight(170)
            chart_wrap_l.addWidget(fb)

        seg_card = QWidget()
        seg_card.setObjectName("statsCard")
        seg_card_l = QVBoxLayout(seg_card)
        seg_card_l.setContentsMargins(8, 8, 8, 8)
        seg_card_l.setSpacing(4)
        seg_card_title = QLabel("总分割数")
        seg_card_title.setObjectName("segTotalTitle")
        self.s_seg_total = QLabel("0")
        self.s_seg_total.setObjectName("segTotalValue")
        self.s_seg_total.setAlignment(Qt.AlignCenter)
        self.s_avg_conf_now = QLabel("平均置信: 0.000")
        self.s_avg_conf_now.setObjectName("usageValue")
        self.s_avg_conf_now.setAlignment(Qt.AlignCenter)
        self.s_speed_now = QLabel("识别速度: 0.00 img/s")
        self.s_speed_now.setObjectName("usageValue")
        self.s_speed_now.setAlignment(Qt.AlignCenter)
        self.s_cpu_usage = QLabel("CPU: 0%")
        self.s_cpu_usage.setObjectName("usageValue")
        self.s_cpu_usage.setAlignment(Qt.AlignCenter)
        self.s_cuda_usage = QLabel("CUDA: 0%")
        self.s_cuda_usage.setObjectName("usageValue")
        self.s_cuda_usage.setAlignment(Qt.AlignCenter)
        seg_card_l.addWidget(seg_card_title, alignment=Qt.AlignHCenter)
        seg_card_l.addWidget(self.s_seg_total, alignment=Qt.AlignHCenter)
        seg_card_l.addWidget(self.s_avg_conf_now, alignment=Qt.AlignHCenter)
        seg_card_l.addWidget(self.s_speed_now, alignment=Qt.AlignHCenter)
        seg_card_l.addWidget(self.s_cpu_usage, alignment=Qt.AlignHCenter)
        seg_card_l.addWidget(self.s_cuda_usage, alignment=Qt.AlignHCenter)

        self.auto_reset_stats = QCheckBox("完成后清零统计")
        self.btn_reset_stats = QPushButton("清零统计")
        self.btn_reset_stats.setObjectName("miniBtn")
        self.btn_reset_stats.setMinimumWidth(84)
        self.btn_reset_stats.clicked.connect(self.reset_stats_view)
        stats.addWidget(chart_wrap, 0, 0, 2, 1)
        stats.addWidget(seg_card, 0, 1)
        self._refresh_runtime_usage_label(0.0, 0.0)

        mascot_wrap = QWidget()
        mascot_wrap.setObjectName("statsCard")
        mascot_l = QVBoxLayout(mascot_wrap)
        mascot_l.setContentsMargins(2, 8, 8, 8)
        mascot_l.setSpacing(4)
        self.mascot_label = QLabel("  o  \n /|\\ \n / \\")
        self.mascot_label.setObjectName("mascotValue")
        self.mascot_label.setAlignment(Qt.AlignCenter)
        self.mascot_step_label = QLabel("待机")
        self.mascot_step_label.setObjectName("usageValue")
        self.mascot_step_label.setAlignment(Qt.AlignCenter)
        mascot_l.addStretch(1)
        mascot_l.addWidget(self.mascot_label, alignment=Qt.AlignHCenter)
        mascot_l.addWidget(self.mascot_step_label, alignment=Qt.AlignHCenter)
        mascot_l.addStretch(1)
        stats.addWidget(mascot_wrap, 0, 2)

        stats_action_wrap = QWidget()
        stats_action_l = QHBoxLayout(stats_action_wrap)
        stats_action_l.setContentsMargins(0, 0, 0, 0)
        stats_action_l.setSpacing(8)
        stats_action_l.addWidget(self.auto_reset_stats)
        stats_action_l.addStretch(1)
        stats_action_l.addWidget(self.btn_reset_stats)
        stats.addWidget(stats_action_wrap, 1, 1, 1, 2)

        self.t_epoch = QLabel("-")
        self.t_loss = QLabel("-")
        self.t_map = QLabel("-")
        self.t_best = QLabel("-")
        self.t_onnx = QLabel("-")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.llm_progress_text = QLabel("扩词进度: 待启动")
        self.llm_progress_text.setObjectName("llmProgressText")
        self.llm_progress = QProgressBar()
        self.llm_progress.setRange(0, 100)
        self.llm_progress.setValue(0)
        self.llm_progress.setFormat("%p%")
        self.llm_terms_view = QPlainTextEdit()
        self.llm_terms_view.setObjectName("consoleBox")
        self.llm_terms_view.setReadOnly(True)
        self.llm_terms_view.setPlaceholderText("这里显示 LLM 扩展后的提示词")
        self.log = QPlainTextEdit()
        self.log.setObjectName("consoleBox")
        self.log.setReadOnly(True)

        right_l.addLayout(preview_toolbar)
        right_l.addWidget(self.preview)
        right_l.addLayout(stats)
        right_l.addWidget(self.bar)
        right_l.addWidget(self.llm_progress_text)
        right_l.addWidget(self.llm_progress)
        self.lbl_llm_result_title = QLabel("LLM扩词结果")
        right_l.addWidget(self.lbl_llm_result_title)
        right_l.addWidget(self.llm_terms_view)
        right_l.addWidget(self.log)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setWidget(left_content)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.NoFrame)
        right_scroll.setWidget(right)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 7)
        splitter.setSizes([560, 760])
        annotate_page_layout.addWidget(splitter)

        train_shell = QWidget()
        train_shell.setObjectName("panelCard")
        train_shell_l = QVBoxLayout(train_shell)
        train_shell_l.setContentsMargins(14, 14, 14, 14)
        train_shell_l.setSpacing(10)

        self.train_title_label = QLabel("快速训练")
        self.train_title_label.setObjectName("trainTitle")
        self.train_subtitle_label = QLabel("简化流程：填路径 -> 设轮次 -> 一键启动")
        self.train_subtitle_label.setObjectName("trainSubTitle")
        train_shell_l.addWidget(self.train_title_label)
        train_shell_l.addWidget(self.train_subtitle_label)

        top_hud = QVBoxLayout()
        top_hud.setSpacing(10)

        format_card = QWidget()
        format_card.setObjectName("trainTopCard")
        format_l = QGridLayout(format_card)
        format_l.setContentsMargins(10, 10, 10, 10)
        format_l.setHorizontalSpacing(6)
        format_l.setVerticalSpacing(6)
        self.lbl_train_format = QLabel("数据集生成格式")
        format_l.addWidget(self.lbl_train_format, 0, 0)
        format_l.addWidget(self.train_dataset_format, 0, 1)
        self.train_format_tip_label = QLabel("提示: 选择 XML 时会先生成 VOC 数据集，当前流程不执行训练")
        self.train_format_tip_label.setObjectName("trainModelInfo")
        format_l.addWidget(self.train_format_tip_label, 1, 0, 1, 2)

        dir_card = QWidget()
        dir_card.setObjectName("trainTopCard")
        dir_l = QGridLayout(dir_card)
        dir_l.setContentsMargins(10, 10, 10, 10)
        dir_l.setHorizontalSpacing(6)
        dir_l.setVerticalSpacing(6)

        self.lbl_train_ann_dir = QLabel("标注目录")
        dir_l.addWidget(self.lbl_train_ann_dir, 0, 0)
        dir_l.addWidget(self.train_ann_edit, 0, 1)
        dir_l.addWidget(self.btn_pick_ann, 0, 2)
        self.lbl_train_img_dir = QLabel("图片目录")
        dir_l.addWidget(self.lbl_train_img_dir, 1, 0)
        dir_l.addWidget(self.train_images_edit, 1, 1)
        dir_l.addWidget(self.btn_pick_img, 1, 2)
        self.lbl_train_data_out = QLabel("数据输出")
        dir_l.addWidget(self.lbl_train_data_out, 2, 0)
        dir_l.addWidget(self.train_dataset_out_edit, 2, 1)
        dir_l.addWidget(self.btn_pick_dataset_out, 2, 2)

        model_card = QWidget()
        model_card.setObjectName("trainTopCard")
        model_l = QGridLayout(model_card)
        model_l.setContentsMargins(10, 10, 10, 10)
        model_l.setHorizontalSpacing(6)
        model_l.setVerticalSpacing(6)

        self.btn_refresh_models = QPushButton("刷新")
        self.btn_refresh_models.setObjectName("miniBtn")
        self.btn_refresh_models.setMaximumWidth(68)
        self.btn_refresh_models.setMaximumHeight(26)

        self.lbl_train_model_raw = QLabel("训练模型(raw)")
        model_l.addWidget(self.lbl_train_model_raw, 0, 0)
        model_l.addWidget(self.train_model_combo, 0, 1)
        model_l.addWidget(self.btn_refresh_models, 0, 2)
        self.lbl_train_model_manual = QLabel("手动模型")
        model_l.addWidget(self.lbl_train_model_manual, 1, 0)
        model_l.addWidget(self.train_model_edit, 1, 1)
        model_l.addWidget(self.btn_pick_train_model, 1, 2)
        model_l.addWidget(self.train_model_info, 2, 0, 1, 3)

        top_hud.addWidget(format_card)
        top_hud.addWidget(dir_card)
        top_hud.addWidget(model_card)

        compact = QGridLayout()
        compact.setHorizontalSpacing(8)
        compact.setVerticalSpacing(6)
        self.lbl_train_epoch = QLabel("Epoch")
        compact.addWidget(self.lbl_train_epoch, 0, 0)
        compact.addWidget(self.train_epochs, 0, 1)
        self.lbl_train_batch = QLabel("Batch")
        compact.addWidget(self.lbl_train_batch, 0, 2)
        compact.addWidget(self.train_batch, 0, 3)
        self.lbl_train_val = QLabel("Val")
        compact.addWidget(self.lbl_train_val, 1, 0)
        compact.addWidget(self.train_val_ratio, 1, 1)
        compact.addWidget(self.train_export_onnx, 1, 2, 1, 2)
        self.lbl_train_max_ram = QLabel("最大RAM(%)")
        compact.addWidget(self.lbl_train_max_ram, 2, 0)
        compact.addWidget(self.train_max_ram, 2, 1)
        self.lbl_train_max_vram = QLabel("最大VRAM(%)")
        compact.addWidget(self.lbl_train_max_vram, 2, 2)
        compact.addWidget(self.train_max_vram, 2, 3)
        compact.addWidget(self.train_memory_info, 3, 0, 1, 4)

        self.btn_train.setText("开始训练")
        self.btn_train.setMinimumHeight(42)

        self.train_log = QPlainTextEdit()
        self.train_log.setObjectName("consoleBox")
        self.train_log.setReadOnly(True)
        self.train_log.setPlaceholderText("这里显示训练日志")
        self.train_log.setMinimumHeight(240)

        train_controls = QWidget()
        train_controls.setObjectName("trainControlBar")
        train_controls_l = QVBoxLayout(train_controls)
        train_controls_l.setContentsMargins(10, 10, 10, 10)
        train_controls_l.setSpacing(6)
        train_controls_l.addLayout(top_hud)
        train_controls_l.addLayout(compact)
        train_action = QHBoxLayout()
        train_action.setSpacing(8)
        train_action.addWidget(self.btn_train, 2)
        train_action.addWidget(self.btn_train_pause, 1)
        train_action.addWidget(self.btn_train_stop, 1)
        train_controls_l.addLayout(train_action)

        trend_panel = QWidget()
        trend_panel.setObjectName("trainTrendPanel")
        trend_l = QVBoxLayout(trend_panel)
        trend_l.setContentsMargins(10, 10, 10, 10)
        trend_l.setSpacing(6)
        self.train_trend_title_label = QLabel("训练趋势（4折线）")
        self.train_trend_title_label.setObjectName("trainTrendTitle")
        trend_l.addWidget(self.train_trend_title_label)

        if HAS_MPL:
            self.train_fig = Figure(figsize=(6.2, 3.8), tight_layout=True)
            self.train_canvas = FigureCanvas(self.train_fig)
            trend_l.addWidget(self.train_canvas, 1)
            self._update_train_trend_charts()
        else:
            self.train_canvas = None
            trend_l.addWidget(QLabel("未检测到 matplotlib，无法显示训练折线图"))

        self.train_top_splitter = QSplitter(Qt.Horizontal)
        self.train_top_splitter.setHandleWidth(8)
        self.train_top_splitter.setChildrenCollapsible(False)
        self.train_top_splitter.addWidget(train_controls)
        self.train_top_splitter.addWidget(trend_panel)
        self.train_top_splitter.setStretchFactor(0, 5)
        self.train_top_splitter.setStretchFactor(1, 7)
        self.train_top_splitter.setSizes([650, 750])
        train_shell_l.addWidget(self.train_top_splitter, 1)

        train_shell_l.addWidget(self.train_log)
        train_scroll = QScrollArea()
        train_scroll.setWidgetResizable(True)
        train_scroll.setFrameShape(QScrollArea.NoFrame)
        train_scroll.setWidget(train_shell)
        train_page_layout.addWidget(train_scroll)

        deploy_shell = QWidget()
        deploy_shell.setObjectName("panelCard")
        deploy_l = QVBoxLayout(deploy_shell)
        deploy_l.setContentsMargins(14, 14, 14, 14)
        deploy_l.setSpacing(10)

        self.deploy_title_label = QLabel("快速部署")
        self.deploy_title_label.setObjectName("trainTitle")
        self.deploy_subtitle_label = QLabel("摄像头实时识别组合提示词目标（默认启用成分组合）")
        self.deploy_subtitle_label.setObjectName("trainSubTitle")
        deploy_l.addWidget(self.deploy_title_label)
        deploy_l.addWidget(self.deploy_subtitle_label)

        self.deploy_model_edit = QLineEdit("yoloe-26x-seg.pt")
        self.deploy_camera_id = NoWheelSpinBox()
        self.deploy_camera_id.setRange(0, 9)
        self.deploy_camera_id.setValue(0)
        self.deploy_prompt_edit = QLineEdit()
        self.deploy_prompt_edit.setPlaceholderText("例如: 笔记本电脑, 电源适配器")
        self.deploy_negative_prompt_edit = QLineEdit()
        self.deploy_negative_prompt_edit.setPlaceholderText("例如: 手机, 平板")
        self.deploy_target_desc_edit = QPlainTextEdit()
        self.deploy_target_desc_edit.setMaximumHeight(82)
        self.deploy_target_desc_edit.setPlaceholderText("目标描述（可用于智能建议）")
        self.deploy_conf = NoWheelDoubleSpinBox()
        self.deploy_conf.setRange(0.001, 1.0)
        self.deploy_conf.setDecimals(3)
        self.deploy_conf.setValue(0.35)
        self.deploy_imgsz = NoWheelSpinBox()
        self.deploy_imgsz.setRange(320, 2048)
        self.deploy_imgsz.setValue(960)
        self.deploy_enable_llm = QCheckBox("启用扩词")
        self.deploy_enable_llm.setChecked(True)
        self.deploy_component_enable = QCheckBox("成分组合")
        self.deploy_component_enable.setChecked(True)
        self.deploy_component_enable.setEnabled(False)
        self.deploy_component_name = QLineEdit("目标")
        self.deploy_target_mode_combo = NoWheelComboBox()
        self.deploy_target_mode_combo.addItem("单目标", "single")
        self.deploy_target_mode_combo.addItem("多目标", "multi")
        self.deploy_target_mode_combo.setCurrentIndex(0)

        btn_pick_deploy_model = QPushButton("模型文件")
        btn_pick_deploy_model.setObjectName("miniBtn")
        btn_pick_deploy_model.clicked.connect(self.pick_deploy_model)
        btn_deploy_suggest = QPushButton("智能建议")
        btn_deploy_suggest.setObjectName("miniBtn")
        btn_deploy_suggest.clicked.connect(self.run_deploy_prompt_assistant)

        deploy_controls = QWidget()
        deploy_controls.setObjectName("trainControlBar")
        deploy_controls_l = QGridLayout(deploy_controls)
        deploy_controls_l.setContentsMargins(10, 10, 10, 10)
        deploy_controls_l.setHorizontalSpacing(12)
        deploy_controls_l.setVerticalSpacing(8)
        self.lbl_deploy_model = QLabel("模型")
        deploy_controls_l.addWidget(self.lbl_deploy_model, 0, 0)
        deploy_controls_l.addWidget(self.deploy_model_edit, 0, 1)
        deploy_controls_l.addWidget(btn_pick_deploy_model, 0, 2)
        self.lbl_deploy_camera = QLabel("摄像头")
        deploy_controls_l.addWidget(self.lbl_deploy_camera, 1, 0)
        deploy_controls_l.addWidget(self.deploy_camera_id, 1, 1)
        self.lbl_deploy_conf = QLabel("置信阈值")
        deploy_controls_l.addWidget(self.lbl_deploy_conf, 1, 2)
        deploy_controls_l.addWidget(self.deploy_conf, 1, 3)
        self.lbl_deploy_imgsz = QLabel("输入尺寸")
        deploy_controls_l.addWidget(self.lbl_deploy_imgsz, 2, 0)
        deploy_controls_l.addWidget(self.deploy_imgsz, 2, 1)
        deploy_controls_l.addWidget(self.deploy_enable_llm, 2, 2)
        deploy_controls_l.addWidget(self.deploy_component_enable, 2, 3)
        self.lbl_deploy_prompt = QLabel("提示词")
        deploy_controls_l.addWidget(self.lbl_deploy_prompt, 3, 0)
        deploy_controls_l.addWidget(self.deploy_prompt_edit, 3, 1, 1, 3)
        self.lbl_deploy_negative = QLabel("负提示词")
        deploy_controls_l.addWidget(self.lbl_deploy_negative, 4, 0)
        deploy_controls_l.addWidget(self.deploy_negative_prompt_edit, 4, 1, 1, 3)
        self.lbl_deploy_target = QLabel("目标描述")
        deploy_controls_l.addWidget(self.lbl_deploy_target, 5, 0)
        deploy_controls_l.addWidget(self.deploy_target_desc_edit, 5, 1, 1, 2)
        deploy_controls_l.addWidget(btn_deploy_suggest, 5, 3)
        self.lbl_deploy_component = QLabel("组合名称")
        deploy_controls_l.addWidget(self.lbl_deploy_component, 6, 0)
        deploy_controls_l.addWidget(self.deploy_component_name, 6, 1, 1, 3)
        self.lbl_deploy_target_mode = QLabel("目标模式")
        deploy_controls_l.addWidget(self.lbl_deploy_target_mode, 7, 0)
        deploy_controls_l.addWidget(self.deploy_target_mode_combo, 7, 1, 1, 3)
        deploy_controls_l.setColumnStretch(0, 0)
        deploy_controls_l.setColumnStretch(1, 4)
        deploy_controls_l.setColumnStretch(2, 0)
        deploy_controls_l.setColumnStretch(3, 4)

        deploy_chart_panel = QWidget()
        deploy_chart_panel.setObjectName("trainTrendPanel")
        deploy_chart_l = QVBoxLayout(deploy_chart_panel)
        deploy_chart_l.setContentsMargins(10, 10, 10, 10)
        deploy_chart_l.setSpacing(6)
        deploy_chart_head = QHBoxLayout()
        deploy_chart_head.setSpacing(6)
        deploy_chart_head.addWidget(QLabel("柱状图: 平均帧数FPS(蓝) / 平均分割数(橙)"))
        deploy_chart_head.addStretch(1)
        self.lbl_chart_granularity_deploy = QLabel("综合粒度")
        deploy_chart_head.addWidget(self.lbl_chart_granularity_deploy)
        self.deploy_stats_window_combo = NoWheelComboBox()
        self.deploy_stats_window_combo.addItems(["每1帧综合", "每5帧综合", "每20帧综合", "每100帧综合"])
        self.deploy_stats_window_combo.setCurrentText("每1帧综合")
        self.deploy_stats_window_combo.currentTextChanged.connect(lambda _t: self._update_deploy_stats_chart())
        deploy_chart_head.addWidget(self.deploy_stats_window_combo)
        deploy_chart_l.addLayout(deploy_chart_head)
        if HAS_MPL:
            self.deploy_stats_fig = Figure(figsize=(5.2, 2.1), tight_layout=True)
            self.deploy_stats_ax = self.deploy_stats_fig.add_subplot(111)
            self.deploy_stats_canvas = FigureCanvas(self.deploy_stats_fig)
            self.deploy_stats_canvas.setMinimumHeight(180)
            deploy_chart_l.addWidget(self.deploy_stats_canvas)
        else:
            self.deploy_stats_canvas = None
            self.deploy_stats_ax = None
            deploy_chart_l.addWidget(QLabel("未检测到 matplotlib，无法显示统计柱状图"))

        self.deploy_top_splitter = QSplitter(Qt.Horizontal)
        self.deploy_top_splitter.setHandleWidth(8)
        self.deploy_top_splitter.setChildrenCollapsible(False)
        self.deploy_top_splitter.addWidget(deploy_controls)
        self.deploy_top_splitter.addWidget(deploy_chart_panel)
        self.deploy_top_splitter.setStretchFactor(0, 7)
        self.deploy_top_splitter.setStretchFactor(1, 6)
        self.deploy_top_splitter.setSizes([720, 680])
        deploy_l.addWidget(self.deploy_top_splitter)

        deploy_state = QHBoxLayout()
        self.deploy_state_icon = QLabel("●")
        self.deploy_state_icon.setObjectName("usageValue")
        self.deploy_state_icon.setStyleSheet("color:#16a34a;")
        self.deploy_state_text = QLabel("待机")
        self.deploy_state_text.setObjectName("usageValue")
        self.btn_deploy_start = QPushButton("启动摄像头识别")
        self.btn_deploy_stop = QPushButton("停止部署")
        self.btn_deploy_stop.setEnabled(False)
        self.btn_deploy_start.clicked.connect(self.start_deploy)
        self.btn_deploy_stop.clicked.connect(self.stop)
        deploy_state.addWidget(self.deploy_state_icon)
        deploy_state.addWidget(self.deploy_state_text)
        deploy_state.addStretch(1)
        deploy_state.addWidget(self.btn_deploy_start)
        deploy_state.addWidget(self.btn_deploy_stop)
        deploy_l.addLayout(deploy_state)

        self.deploy_preview = QLabel("部署预览区")
        self.deploy_preview.setObjectName("previewFrame")
        self.deploy_preview.setAlignment(Qt.AlignCenter)
        self.deploy_preview.setMinimumHeight(420)
        deploy_l.addWidget(self.deploy_preview)

        self.deploy_log = QPlainTextEdit()
        self.deploy_log.setObjectName("consoleBox")
        self.deploy_log.setReadOnly(True)
        self.deploy_log.setPlaceholderText("这里显示快速部署日志")
        self.deploy_log.setMinimumHeight(130)
        deploy_l.addWidget(self.deploy_log)

        deploy_scroll = QScrollArea()
        deploy_scroll.setWidgetResizable(True)
        deploy_scroll.setFrameShape(QScrollArea.NoFrame)
        deploy_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        deploy_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        deploy_scroll.setWidget(deploy_shell)
        deploy_page_layout.addWidget(deploy_scroll)

        self.train_model_combo.currentTextChanged.connect(self.on_train_model_combo_changed)
        self.deploy_target_mode_combo.currentIndexChanged.connect(self._on_deploy_target_mode_changed)
        self._on_deploy_target_mode_changed(self.deploy_target_mode_combo.currentIndex())
        self.btn_refresh_models.clicked.connect(self.refresh_train_models_from_raw)
        self.refresh_train_models_from_raw()
        self._update_train_memory_info()
        self._apply_language()

    def _style(self):
        self.setStyleSheet(
            "QWidget#appRoot{font-size:13px;color:#0f172a;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            "stop:0 #f7fbff,stop:0.18 #f7fbff,stop:0.19 #eef4ff,stop:0.37 #eef4ff,"
            "stop:0.38 #f7fbff,stop:0.56 #f7fbff,stop:0.57 #eaf2ff,stop:0.76 #eaf2ff,"
            "stop:0.77 #f6fbff,stop:1 #f6fbff);}"
            "QLabel{color:#0f172a;background:transparent;font-weight:600;}"
            "QLabel#titleText{font-size:22px;font-weight:800;color:#0b1220;}"
            "QLabel#trainTitle{font-size:20px;font-weight:800;color:#0f172a;}"
            "QLabel#trainSubTitle{font-size:12px;color:#64748b;font-weight:600;}"
            "QWidget#trainControlBar{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #f8fcff,stop:1 #f3f8ff);"
            "border:1px solid #cfe0f5;border-radius:12px;}"
            "QWidget#trainTopCard{background:#ffffff;border:1px solid #cfdef2;border-radius:10px;}"
            "QWidget#trainTrendPanel{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #f8fcff,stop:1 #edf5ff);"
            "border:1px solid #d1e1f4;border-radius:12px;}"
            "QLabel#trainTrendTitle{font-size:12px;color:#334155;font-weight:700;}"
            "QWidget#trainControlBar QLabel{color:#334155;font-weight:700;}"
            "QWidget#trainControlBar QLineEdit,QWidget#trainControlBar QSpinBox,QWidget#trainControlBar QDoubleSpinBox,QWidget#trainControlBar QComboBox{"
            "background:#ffffff;color:#0f172a;border:1px solid #bfd2e8;border-radius:8px;padding:4px;}"
            "QWidget#trainControlBar QComboBox QAbstractItemView{"
            "background:#ffffff;color:#0f172a;selection-background-color:#d4ecff;selection-color:#0f172a;"
            "border:1px solid #bfd2e8;}"
            "QComboBox QAbstractItemView{"
            "background:#ffffff;color:#0f172a;selection-background-color:#d4ecff;selection-color:#0f172a;}"
            "QLabel#trainModelInfo{font-size:11px;color:#475569;background:#f7fbff;border:1px solid #d7e5f7;border-radius:8px;padding:6px;}"
            "QPushButton#miniBtn{background:#0f4c81;color:#f8fafc;border:none;border-radius:8px;padding:4px 8px;font-weight:700;}"
            "QPushButton#miniBtn:hover{background:#165d9c;}"
            "QLabel#metricChip{font-size:11px;color:#64748b;font-weight:700;background:transparent;}"
            "QLabel#metricValue{font-size:14px;color:#0f172a;font-weight:800;background:#ffffff;"
            "border:1px solid #ccdcf1;border-radius:8px;padding:2px 8px;}"
            "QLabel#statusText{color:#334155;font-weight:700;}"
            "QLabel#stepText{color:#475569;font-weight:600;}"
            "QLabel#llmProgressText{color:#0f172a;font-weight:700;}"
            "QLabel#statsChartTitle{color:#334155;font-weight:700;}"
            "QLabel#segTotalTitle{color:#334155;font-weight:700;}"
            "QLabel#segTotalValue{font-size:24px;color:#0f172a;font-weight:800;}"
            "QLabel#usageValue{font-size:12px;color:#334155;font-weight:700;}"
            "QLabel#mascotValue{font-family:Consolas,'Courier New',monospace;font-size:19px;color:#0f172a;font-weight:700;}"
            "QWidget#statsCard{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #f9fcff,stop:1 #f1f7ff);border:1px solid #d3e2f4;border-radius:10px;}"
            "QLabel#footerText{font-size:11px;color:#64748b;font-weight:500;padding:2px 6px;}"
            "QWidget#panelCard{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #ffffff,stop:1 #f8fbff);border:1px solid #d1deef;border-radius:14px;}"
            "QTabWidget::pane{border:1px solid #d1deef;border-radius:12px;background:#f7fbff;}"
            "QTabBar::tab{background:#e6eef8;color:#1e293b;border:1px solid #c7d6ea;border-bottom:none;"
            "padding:9px 16px;margin-right:6px;border-top-left-radius:10px;border-top-right-radius:10px;font-weight:700;}"
            "QTabBar::tab:selected{background:#ffffff;color:#0f172a;}"
            "QTabBar::tab:!selected{background:#e7f0fb;color:#475569;}"
            "QSplitter::handle{background:#d4e2f3;border-radius:4px;margin:6px 0px;}"
            "QSplitter::handle:hover{background:#b7d0ea;}"
            "QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QListWidget,QPlainTextEdit{"
            "background:#ffffff;color:#0f172a;border:1px solid #bccfe6;border-radius:10px;padding:7px;}"
            "QLineEdit[invalid='true'],QPlainTextEdit[invalid='true'],QComboBox[invalid='true'],QListWidget[invalid='true']{"
            "border:1px solid #ef4444;background:#fff1f2;}"
            "QLineEdit[invalidBlink='true'],QPlainTextEdit[invalidBlink='true'],QComboBox[invalidBlink='true'],QListWidget[invalidBlink='true']{"
            "border:2px solid #dc2626;background:#fee2e2;}"
            "QLineEdit:focus,QComboBox:focus,QSpinBox:focus,QDoubleSpinBox:focus,QPlainTextEdit:focus{"
            "border:1px solid #22a9d6;background:#fcfeff;}"
            "QLineEdit::placeholder{color:#94a3b8;}"
            "QPushButton{background:#0f4c81;color:#ffffff;border:none;border-radius:10px;padding:8px 14px;font-weight:700;}"
            "QPushButton:hover{background:#165d9c;}"
            "QPushButton:pressed{background:#0b406d;}"
            "QPushButton:disabled{background:#94a3b8;color:#f1f5f9;}"
            "QProgressBar{background:#f8fafc;color:#0f172a;border:1px solid #c5d1e1;border-radius:10px;text-align:center;}"
            "QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1597d4,stop:1 #34c2e4);border-radius:10px;}"
            "QCheckBox{color:#0f172a;background:transparent;}"
            "QCheckBox::indicator{width:14px;height:14px;}"
            "QToolButton#hintButton{background:#e2e8f0;color:#334155;border:1px solid #94a3b8;"
            "border-radius:9px;min-width:18px;min-height:18px;font-weight:700;}"
            "QToolButton#hintButton:hover{background:#cbd5e1;color:#0f172a;}"
            "QLabel#previewFrame{background:#0b1220;color:#cbd5e1;border:1px dashed #334155;border-radius:12px;}"
            "QPlainTextEdit#consoleBox{background:#f8fafc;border:1px solid #d5deea;border-radius:10px;}"
            "QLabel#statusBadge{background:#dcfce7;color:#166534;border-radius:12px;padding:4px 12px;font-weight:800;}"
            "QLabel[status='running']{background:#dbeafe;color:#1d4ed8;}"
            "QLabel[status='ready']{background:#dcfce7;color:#166534;}"
            "QLabel[status='error']{background:#fee2e2;color:#991b1b;}"
        )

    def set_status(self, state, text):
        self.status_label.setText(text)
        step_text = self._extract_step_text() or ("处理中" if self.current_language == "zh" else "Processing")
        if state == "running":
            self.status_badge.setText("RUNNING")
            self._set_mascot_mode("running", step_text)
        elif state == "error":
            self.status_badge.setText("ERROR")
            self._set_mascot_mode("error", step_text)
        else:
            self.status_badge.setText("READY")
            self._set_mascot_mode("idle", step_text)
        self.status_badge.setProperty("status", state)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)

    def append_log(self, text):
        self.log.appendPlainText(text)
        if hasattr(self, "train_log"):
            self.train_log.appendPlainText(text)
        if hasattr(self, "deploy_log"):
            self.deploy_log.appendPlainText(text)
        t = str(text or "")
        if t.startswith("STEP:"):
            self._update_mascot_from_step(t.split(":", 1)[1].strip())
        elif t.startswith("进度 "):
            self._update_mascot_from_step(t)
        self._update_llm_progress_from_log(text)
        self._update_train_info_from_log(text)

    def _ensure_llm_progress_timer(self):
        if self._llm_progress_timer is None:
            self._llm_progress_timer = QTimer(self)
            self._llm_progress_timer.setInterval(1000)
            self._llm_progress_timer.timeout.connect(self._tick_llm_progress)

    def _set_llm_progress(self, value, text=None):
        v = max(0, min(100, int(value)))
        self.llm_progress.setValue(v)
        if text:
            self.llm_progress_text.setText(f"扩词进度: {text}")

    def _set_llm_step(self, step, label):
        self._llm_step = max(0, min(self._llm_total_steps, int(step)))
        percent = int(round(100.0 * float(self._llm_step) / float(self._llm_total_steps)))
        self.llm_progress.setValue(percent)
        self.llm_progress_text.setText(f"扩词进度: 第{self._llm_step}/{self._llm_total_steps}步 - {label}")

    def _begin_llm_progress(self, mode):
        self._llm_progress_active = True
        self._llm_progress_mode = str(mode)
        self._llm_wait_seconds = 0
        self._set_llm_step(1, "初始化任务")
        self._ensure_llm_progress_timer()
        self._llm_progress_timer.start()

    def _tick_llm_progress(self):
        if not self._llm_progress_active:
            return
        if self._llm_step == 2:
            self._llm_wait_seconds += 1
            base = "等待模型响应"
            if self._llm_progress_mode == "assistant":
                base = "等待智能建议响应"
            self.llm_progress_text.setText(
                f"扩词进度: 第2/{self._llm_total_steps}步 - {base}（已等待 {self._llm_wait_seconds}s）"
            )

    def _llm_progress_mark(self, step, text=None):
        if not self._llm_progress_active:
            return
        if step <= self._llm_step:
            return
        self._set_llm_step(step, text or "处理中")

    def _finish_llm_progress(self, ok=True, text=None):
        if self._llm_progress_timer is not None:
            self._llm_progress_timer.stop()
        self._llm_progress_active = False
        self._llm_progress_mode = ""
        self._llm_wait_seconds = 0
        if ok:
            self._set_llm_step(self._llm_total_steps, text or "扩词完成")
        else:
            self._set_llm_step(self._llm_total_steps, text or "扩词失败")

    def _reset_llm_progress(self):
        if self._llm_progress_timer is not None:
            self._llm_progress_timer.stop()
        self._llm_progress_active = False
        self._llm_progress_mode = ""
        self._llm_wait_seconds = 0
        self._llm_step = 0
        self._set_llm_progress(0, "待启动")

    def _update_llm_progress_from_log(self, text):
        if not self._llm_progress_active:
            return
        t = str(text or "")
        if "开始LLM扩词" in t:
            self._llm_progress_mark(2, "等待模型响应")
        elif "提示词扩展模式" in t:
            self._llm_progress_mark(3, "解析扩词结果")
        elif "LLM_EXPANDED_PROMPTS_BEGIN" in t:
            self._llm_progress_mark(4, "整理扩词分组")
        elif "LLM_EXPANDED_PROMPTS_END" in t:
            self._llm_progress_mark(4, "扩词输出完成，等待主流程接管")
        elif t.startswith("进度 ") or t.startswith("STEP:"):
            if self._llm_step >= 4:
                self._finish_llm_progress(ok=True, text="扩词完成，已进入标注")
        elif "LLM 扩展异常" in t:
            self._finish_llm_progress(ok=False, text="扩词异常，已回退")

    def _update_train_trend_charts(self, epoch_now=None, epoch_total=None, loss=None, map50=None, map95=None):
        if not HAS_MPL or self.train_canvas is None:
            return
        if epoch_now is not None and epoch_total is not None:
            x = int(epoch_now)
            prog = 100.0 * float(epoch_now) / max(1.0, float(epoch_total))
            if self._train_hist_epoch and self._train_hist_epoch[-1] == x:
                self._train_hist_progress[-1] = prog
                self._train_hist_loss[-1] = loss
                self._train_hist_map50[-1] = map50
                self._train_hist_map95[-1] = map95
            else:
                self._train_hist_epoch.append(x)
                self._train_hist_progress.append(prog)
                self._train_hist_loss.append(loss)
                self._train_hist_map50.append(map50)
                self._train_hist_map95.append(map95)

        self.train_fig.clear()
        ax_prog, ax_loss, ax_map50, ax_map95 = self.train_fig.subplots(2, 2).flatten()

        def _plot(ax, ys, title, color, ylabel):
            xs = self._train_hist_epoch
            clean_y = [float(v) if v is not None else float("nan") for v in ys]
            if xs:
                ax.plot(xs, clean_y, color=color, linewidth=1.8, marker="o", markersize=3)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)

        _plot(ax_prog, self._train_hist_progress, "进度(%)", "#2563eb", "%")
        _plot(ax_loss, self._train_hist_loss, "Loss", "#dc2626", "Loss")
        _plot(ax_map50, self._train_hist_map50, "mAP50", "#16a34a", "mAP50")
        _plot(ax_map95, self._train_hist_map95, "mAP50-95", "#f59e0b", "mAP")

        self.train_fig.tight_layout(pad=1.2)
        self.train_canvas.draw_idle()

    def _update_train_info_from_log(self, text):
        m = re.search(
            r"TRAIN_METRIC:\s*epoch=(\d+)/(\d+)\s+loss=([^\s]+)\s+map50=([^\s]+)\s+map50_95=([^\s]+)",
            text,
        )
        if m:
            ep, total, loss, _map50, map95 = m.groups()
            self.t_epoch.setText(f"{ep}/{total}")
            self.t_loss.setText(loss)
            self.t_map.setText(map95)

            def _safe_float(v):
                try:
                    return float(v)
                except Exception:
                    return None

            self._update_train_trend_charts(
                epoch_now=int(ep),
                epoch_total=int(total),
                loss=_safe_float(loss),
                map50=_safe_float(_map50),
                map95=_safe_float(map95),
            )

        if text.startswith("TRAIN_BEST:"):
            p = text.split(":", 1)[1].strip()
            self.train_best_path = p
            self.t_best.setText(Path(p).name or p)

        if text.startswith("TRAIN_ONNX:"):
            p = text.split(":", 1)[1].strip()
            self.train_onnx_path = p
            self.t_onnx.setText(Path(p).name or p)

    def refresh_train_models_from_raw(self):
        raw_dir = Path(self.workspace) / "raw"
        exts = {".pt", ".onnx", ".engine"}
        items = []
        if raw_dir.exists() and raw_dir.is_dir():
            items = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
            items.sort()

        self.train_model_combo.blockSignals(True)
        self.train_model_combo.clear()
        if items:
            for p in items:
                self.train_model_combo.addItem(self._to_rel_display_path(p))
            preferred_idx = 0
            # 默认优先非分割的 26l 模型，其次再回退到 26l-seg。
            for i, p in enumerate(items):
                name = p.name.lower()
                if "26l" in name and not ("-seg" in name or "_seg" in name):
                    preferred_idx = i
                    break
            else:
                for i, p in enumerate(items):
                    name = p.name.lower()
                    if "26l" in name and ("-seg" in name or "_seg" in name):
                        preferred_idx = i
                        break
            self.train_model_combo.setCurrentIndex(preferred_idx)
            self.train_model_edit.setText(self.train_model_combo.currentText())
        else:
            self.train_model_combo.addItem("(raw 目录无可用模型)")
        self.train_model_combo.blockSignals(False)
        self.on_train_model_combo_changed(self.train_model_combo.currentText())

    def on_train_model_combo_changed(self, text):
        p = self._resolve_input_path(str(text).strip())
        if p.exists() and p.is_file():
            self.train_model_edit.setText(self._to_rel_display_path(p))
            try:
                st = p.stat()
                size_mb = st.st_size / (1024.0 * 1024.0)
                self.train_model_info.setText(
                    f"名称: {p.name} | 格式: {p.suffix.lower()} | 大小: {size_mb:.2f} MB"
                )
            except Exception:
                self.train_model_info.setText(f"名称: {p.name} | 格式: {p.suffix.lower()} | 大小: 未知")
        else:
            self.train_model_info.setText("模型参数: 未找到 raw 目录模型，可手动选择")

    def append_llm_prompt(self, text):
        self.llm_terms_view.appendPlainText(text)

    def show_feature_help(self, title, image_name, desc):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{title} - 功能说明")
        dlg.resize(640, 520)
        dlg.setStyleSheet(
            "QDialog{background:#ffffff;color:#0f172a;}"
            "QLabel{color:#0f172a;background:#ffffff;}"
            "QPushButton{background:#0f172a;color:#ffffff;border:none;border-radius:8px;padding:6px 12px;}"
            "QPushButton:hover{background:#1e293b;}"
        )

        lay = QVBoxLayout(dlg)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size:16px;font-weight:700;color:#0f172a;")
        lay.addWidget(title_label)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        img_path = Path(self.workspace) / "res" / image_name
        if img_path.exists():
            pix = QPixmap(str(img_path))
            if not pix.isNull():
                image_label.setPixmap(pix.scaled(600, 340, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                image_label.setText(f"无法读取示例图: {img_path.name}")
        else:
            image_label.setText(f"未找到示例图: {img_path.name}")
        lay.addWidget(image_label)

        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size:13px;color:#334155;background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;padding:10px;")
        lay.addWidget(desc_label)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dlg.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn_close)
        lay.addLayout(row)

        dlg.exec()

    def _on_force_cpu_toggled(self, checked):
        self.device.setEnabled(not checked)
        if checked:
            self.device.setCurrentText("cpu")
        self._update_train_memory_info()

    def _update_train_memory_info(self):
        ram_text = "-"
        vram_text = "-"
        try:
            if psutil is not None:
                total = float(psutil.virtual_memory().total) / (1024.0 ** 3)
                ram_text = f"{total:.1f} GB"
        except Exception:
            pass
        try:
            if torch is not None and torch.cuda.is_available():
                dev = torch.cuda.current_device()
                prop = torch.cuda.get_device_properties(dev)
                total_vram = float(getattr(prop, "total_memory", 0.0) or 0.0) / (1024.0 ** 3)
                vram_text = f"{total_vram:.1f} GB"
            else:
                vram_text = "N/A"
        except Exception:
            vram_text = "N/A"

        if getattr(self, "current_language", "zh") == "en":
            self.train_memory_info.setText(f"System Memory: RAM {ram_text} | VRAM {vram_text}")
        else:
            self.train_memory_info.setText(f"本机内存: RAM {ram_text} | VRAM {vram_text}")

    def clear_output(self):
        self.log.setPlainText("")
        if hasattr(self, "train_log"):
            self.train_log.setPlainText("")
        if hasattr(self, "deploy_log"):
            self.deploy_log.setPlainText("")
        self.llm_terms_view.setPlainText("[正向扩词分组]\n")
        self.t_epoch.setText("-")
        self.t_loss.setText("-")
        self.t_map.setText("-")
        self.t_best.setText("-")
        self.t_onnx.setText("-")
        self.reset_stats_view()
        self._reset_llm_progress()
        self.train_best_path = ""
        self.train_onnx_path = ""
        self._train_hist_epoch = []
        self._train_hist_progress = []
        self._train_hist_loss = []
        self._train_hist_map50 = []
        self._train_hist_map95 = []
        self._update_train_trend_charts()
        self._reset_deploy_stats_view()
        self.update_step(self._tr("step_wait"))

    def update_step(self, text):
        self.step_label.setText(f"{self._tr('step_prefix')}: {text}")
        self._update_mascot_from_step(text)

    def toggle_preview_pause(self):
        if self.worker is None:
            return
        if self.current_task != "annotate":
            self.append_log("STEP: 暂停识别仅在标注任务中可用")
            return
        if self._run_paused:
            self.worker.resume_after_pause()
            return
        if self._run_pause_requested:
            self.worker.cancel_pause_request()
            self._run_pause_requested = False
            self.btn_pause_preview.setText(self._tr("btn_pause"))
            self.append_log("STEP: 已取消暂停请求")
            return
        self._run_pause_requested = True
        self.btn_pause_preview.setText(self._tr("btn_wait_current"))
        self.worker.request_pause_after_current()
        self.append_log("STEP: 已请求暂停，将在当前图片完成后暂停识别")

    def update_preview(self, path):
        self.current_preview_path = path
        if self.current_task == "annotate" and self.preview_paused:
            return
        try:
            data = Path(path).read_bytes()
        except Exception:
            return
        pix = QPixmap()
        pix.loadFromData(data)
        if pix.isNull():
            return
        target = self.preview
        if self.current_task == "deploy" and hasattr(self, "deploy_preview"):
            target = self.deploy_preview
        target.setPixmap(pix.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _apply_splitter_ratio(self, splitter, left_ratio):
        if splitter is None:
            return
        if splitter.count() < 2:
            return
        total_w = max(240, splitter.size().width())
        left_w = int(total_w * float(left_ratio))
        right_w = max(120, total_w - left_w)
        splitter.setSizes([left_w, right_w])

    def _pick_layout_profile(self):
        w = max(1, self.width())
        h = max(1, self.height())
        ratio = w / float(h)
        # 16:9 更宽，16:10 更平衡；其余走紧凑模式
        if ratio >= 1.72:
            return {
                "annotate_left": 0.43,
                "train_left": 0.54,
                "deploy_left": 0.55,
                "annotate_preview_h": 380,
                "deploy_preview_h": 420,
            }
        if ratio >= 1.56:
            return {
                "annotate_left": 0.46,
                "train_left": 0.57,
                "deploy_left": 0.58,
                "annotate_preview_h": 360,
                "deploy_preview_h": 400,
            }
        return {
            "annotate_left": 0.50,
            "train_left": 0.60,
            "deploy_left": 0.62,
            "annotate_preview_h": 340,
            "deploy_preview_h": 360,
        }

    def _apply_aspect_layout(self):
        profile = self._pick_layout_profile()
        self._apply_splitter_ratio(self.annotate_splitter, profile["annotate_left"])
        self._apply_splitter_ratio(self.train_top_splitter, profile["train_left"])
        self._apply_splitter_ratio(self.deploy_top_splitter, profile["deploy_left"])
        if hasattr(self, "preview"):
            self.preview.setMinimumHeight(profile["annotate_preview_h"])
        if hasattr(self, "deploy_preview"):
            self.deploy_preview.setMinimumHeight(profile["deploy_preview_h"])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_aspect_layout()
        if self.current_preview_path:
            self.update_preview(self.current_preview_path)

    def update_model_progress(self, done, total):
        if total > 0:
            self.bar.setValue(int(done * 100 / total))

    def reset_stats_view(self):
        self.s_seg_total.setText("0")
        self.s_avg_conf_now.setText("平均置信: 0.000")
        self.s_speed_now.setText("识别速度: 0.00 img/s")
        self._refresh_runtime_usage_label(0.0, 0.0)
        self._stats_chart_start_ts = None
        self._stats_x_hist = []
        self._stats_fps_hist = []
        self._stats_segavg_hist = []
        self._update_runtime_stats_chart()

    def _get_runtime_device_usage(self):
        cpu_pct = 0.0
        cuda_pct = 0.0
        try:
            if psutil is not None:
                cpu_pct = float(psutil.cpu_percent(interval=None))
        except Exception:
            cpu_pct = 0.0

        try:
            if torch is not None and torch.cuda.is_available():
                dev = torch.cuda.current_device()
                prop = torch.cuda.get_device_properties(dev)
                total = float(getattr(prop, "total_memory", 0.0) or 0.0)
                used = float(torch.cuda.memory_allocated(dev))
                if total > 0:
                    cuda_pct = max(0.0, min(100.0, (used / total) * 100.0))
        except Exception:
            cuda_pct = 0.0

        return cpu_pct, cuda_pct

    def _refresh_runtime_usage_label(self, cpu_pct: float, cuda_pct: float):
        dev = "auto"
        try:
            dev = (self.device.currentText() or "auto").strip().lower()
        except Exception:
            dev = "auto"

        use_cuda = False
        if dev.startswith("cuda"):
            use_cuda = True
        elif dev.startswith("cpu"):
            use_cuda = False
        else:
            try:
                use_cuda = bool(torch is not None and torch.cuda.is_available())
            except Exception:
                use_cuda = False

        if use_cuda:
            self.s_cuda_usage.setVisible(True)
            self.s_cpu_usage.setVisible(False)
            self.s_cuda_usage.setText(f"CUDA: {cuda_pct:.1f}%")
        else:
            self.s_cpu_usage.setVisible(True)
            self.s_cuda_usage.setVisible(False)
            self.s_cpu_usage.setText(f"CPU: {cpu_pct:.1f}%")

    def _update_runtime_stats_chart(self):
        if not HAS_MPL or self.stats_canvas is None or self.stats_ax is None:
            return

        def _bucket_size():
            try:
                t = self.stats_window_combo.currentText().strip()
                return max(1, int(re.sub(r"[^0-9]", "", t) or "20"))
            except Exception:
                return 20

        bucket = _bucket_size()
        self.stats_fig.clear()
        self.stats_ax = self.stats_fig.add_subplot(111)
        self.stats_ax.clear()
        if not self._stats_x_hist:
            self.stats_ax.set_xlabel("已处理图片数")
            self.stats_ax.set_ylabel("平均帧数(FPS)")
            self.stats_ax.grid(alpha=0.2)
            self.stats_canvas.draw_idle()
            return

        ax1 = self.stats_ax
        ax2 = ax1.twinx()
        max_bars = 6
        x_src = self._stats_x_hist
        fps_src = self._stats_fps_hist
        segavg_src = self._stats_segavg_hist
        if bucket <= 1:
            x_raw = list(x_src)
            y_fps = list(fps_src)
            y_segavg = list(segavg_src)
        else:
            x_raw = []
            y_fps = []
            y_segavg = []
            n = len(x_src)
            for i in range(0, n, bucket):
                j = min(n, i + bucket)
                x_raw.append(int(x_src[j - 1]))
                y_fps.append(sum(fps_src[i:j]) / float(j - i))
                y_segavg.append(sum(segavg_src[i:j]) / float(j - i))

        if len(x_raw) > max_bars:
            merge_step = int(math.ceil(len(x_raw) / float(max_bars)))
            x_cap = []
            fps_cap = []
            segavg_cap = []
            for i in range(0, len(x_raw), merge_step):
                j = min(len(x_raw), i + merge_step)
                x_cap.append(int(x_raw[j - 1]))
                fps_cap.append(sum(y_fps[i:j]) / float(j - i))
                segavg_cap.append(sum(y_segavg[i:j]) / float(j - i))
            x_raw = x_cap
            y_fps = fps_cap
            y_segavg = segavg_cap
        x = list(range(len(x_raw)))

        width = 0.38
        l1 = ax1.bar([i - width / 2.0 for i in x], y_fps, width=width, color="#2563eb", alpha=0.9, label="平均帧数")
        l2 = ax2.bar([i + width / 2.0 for i in x], y_segavg, width=width, color="#f59e0b", alpha=0.9, label="平均分割数")
        if bucket <= 1:
            ax1.set_xlabel("已处理图片数")
        else:
            ax1.set_xlabel(f"已处理图片数(每{bucket}张综合)")
        ax1.set_ylabel("平均帧数(FPS)", color="#2563eb")
        ax2.set_ylabel("平均分割数", color="#f59e0b")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(v) for v in x_raw], fontsize=8)
        ax1.grid(alpha=0.22)
        lines = [l1, l2]
        labels = ["平均帧数", "平均分割数"]
        ax1.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.26),
            ncol=2,
            fontsize=8,
            frameon=False,
        )
        self.stats_fig.subplots_adjust(top=0.78, bottom=0.22, left=0.10, right=0.90)
        self.stats_canvas.draw_idle()

    def update_stats(self, done, total, det, seg, unknown, avg_conf):
        _ = unknown
        _ = det
        _ = total
        self.s_seg_total.setText(str(seg))

        cpu_pct, cuda_pct = self._get_runtime_device_usage()
        self._refresh_runtime_usage_label(cpu_pct, cuda_pct)

        if self._stats_chart_start_ts is None:
            self._stats_chart_start_ts = time.monotonic()

        elapsed = max(1e-6, time.monotonic() - self._stats_chart_start_ts)
        fps = float(done) / elapsed
        avg_seg = float(seg) / max(1.0, float(done))
        self.s_avg_conf_now.setText(f"平均置信: {float(avg_conf):.3f}")
        self.s_speed_now.setText(f"识别速度: {fps:.2f} img/s")

        if not self._stats_x_hist or int(done) != int(self._stats_x_hist[-1]):
            self._stats_x_hist.append(int(done))
            self._stats_fps_hist.append(float(fps))
            self._stats_segavg_hist.append(float(avg_seg))
        else:
            self._stats_fps_hist[-1] = float(fps)
            self._stats_segavg_hist[-1] = float(avg_seg)

        self._update_runtime_stats_chart()

    def _reset_deploy_stats_view(self):
        self._deploy_stats_start_ts = None
        self._deploy_stats_x_hist = []
        self._deploy_stats_fps_hist = []
        self._deploy_stats_segavg_hist = []
        self._update_deploy_stats_chart()

    def _update_deploy_stats_chart(self):
        if not HAS_MPL or not hasattr(self, "deploy_stats_canvas") or self.deploy_stats_canvas is None:
            return

        def _bucket_size():
            try:
                t = self.deploy_stats_window_combo.currentText().strip()
                return max(1, int(re.sub(r"[^0-9]", "", t) or "20"))
            except Exception:
                return 20

        bucket = _bucket_size()
        self.deploy_stats_fig.clear()
        ax1 = self.deploy_stats_fig.add_subplot(111)
        if not self._deploy_stats_x_hist:
            ax1.set_xlabel("处理帧数")
            ax1.set_ylabel("平均帧数(FPS)")
            ax1.grid(alpha=0.2)
            self.deploy_stats_canvas.draw_idle()
            return

        x_src = self._deploy_stats_x_hist
        fps_src = self._deploy_stats_fps_hist
        segavg_src = self._deploy_stats_segavg_hist
        if bucket <= 1:
            x_raw = list(x_src)
            y_fps = list(fps_src)
            y_segavg = list(segavg_src)
        else:
            x_raw, y_fps, y_segavg = [], [], []
            n = len(x_src)
            for i in range(0, n, bucket):
                j = min(n, i + bucket)
                x_raw.append(int(x_src[j - 1]))
                y_fps.append(sum(fps_src[i:j]) / float(j - i))
                y_segavg.append(sum(segavg_src[i:j]) / float(j - i))

        max_bars = 6
        if len(x_raw) > max_bars:
            merge_step = int(math.ceil(len(x_raw) / float(max_bars)))
            x_cap, fps_cap, segavg_cap = [], [], []
            for i in range(0, len(x_raw), merge_step):
                j = min(len(x_raw), i + merge_step)
                x_cap.append(int(x_raw[j - 1]))
                fps_cap.append(sum(y_fps[i:j]) / float(j - i))
                segavg_cap.append(sum(y_segavg[i:j]) / float(j - i))
            x_raw, y_fps, y_segavg = x_cap, fps_cap, segavg_cap

        x = list(range(len(x_raw)))
        ax2 = ax1.twinx()
        width = 0.38
        l1 = ax1.bar([i - width / 2.0 for i in x], y_fps, width=width, color="#2563eb", alpha=0.9, label="平均帧数")
        l2 = ax2.bar([i + width / 2.0 for i in x], y_segavg, width=width, color="#f59e0b", alpha=0.9, label="平均分割数")
        ax1.set_xlabel("处理帧数" if bucket <= 1 else f"处理帧数(每{bucket}帧综合)")
        ax1.set_ylabel("平均帧数(FPS)", color="#2563eb")
        ax2.set_ylabel("平均分割数", color="#f59e0b")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(v) for v in x_raw], fontsize=8)
        ax1.grid(alpha=0.22)
        ax1.legend([l1, l2], ["平均帧数", "平均分割数"], loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=2, fontsize=8, frameon=False)
        self.deploy_stats_fig.subplots_adjust(top=0.78, bottom=0.22, left=0.10, right=0.90)
        self.deploy_stats_canvas.draw_idle()

    def update_deploy_stats(self, done, total, det, seg, unknown, avg_conf):
        _ = total
        _ = det
        _ = unknown
        _ = avg_conf
        if self.current_task != "deploy":
            return
        if self._deploy_stats_start_ts is None:
            self._deploy_stats_start_ts = time.monotonic()
        elapsed = max(1e-6, time.monotonic() - self._deploy_stats_start_ts)
        fps = float(done) / elapsed
        avg_seg = float(seg) / max(1.0, float(done))
        if not self._deploy_stats_x_hist or int(done) != int(self._deploy_stats_x_hist[-1]):
            self._deploy_stats_x_hist.append(int(done))
            self._deploy_stats_fps_hist.append(float(fps))
            self._deploy_stats_segavg_hist.append(float(avg_seg))
        else:
            self._deploy_stats_fps_hist[-1] = float(fps)
            self._deploy_stats_segavg_hist[-1] = float(avg_seg)
        self._update_deploy_stats_chart()

    def show_efficiency_help(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("有效率说明")
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            "有效率 = 总分割数 / 未识别数。\n"
            "如果有效率偏低，常见原因包括:\n"
            "1) 数据集拍摄质量问题（模糊、遮挡、光照不稳定、目标太小）\n"
            "2) 提示词不够贴切（目标描述不完整或与场景不匹配）"
        )
        msg.setStyleSheet(self._message_box_stylesheet())
        msg.exec()

    def _start_motion_effects(self):
        self._mascot_timer = QTimer(self)
        self._mascot_timer.setInterval(240)
        self._mascot_timer.timeout.connect(self._tick_mascot)
        self._mascot_timer.start()

    def _tick_motion_effects(self):
        return

    def show_help(self):
        msg = (
            "快速使用说明\n"
            "1) 先选图片目录，至少添加一个模型。\n"
            "2) 提示词支持中文，例如: 青苹果、裂缝、异常细胞。\n"
            "3) 勾选API扩词并填写 API Key，可提升冷门目标召回。\n"
            "4) 运行后右侧会显示 LLM 扩词结果与实时日志。\n"
            "5) 若看起来卡住，请检查网络/API可用性，或调低 timeout。"
        )
        self._show_info("帮助", msg)

    def pick_images(self):
        p = QFileDialog.getExistingDirectory(self, "选择图片目录", self.workspace)
        if p:
            self.images_edit.setText(self._to_rel_display_path(p))

    def pick_output(self):
        p = QFileDialog.getExistingDirectory(self, "选择输出目录", self.workspace)
        if p:
            self.output_edit.setText(self._to_rel_display_path(p))

    def pick_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择模型文件", self.workspace, "Model Files (*.pt *.onnx *.engine)")
        if p:
            self.model_edit.setText(self._to_rel_display_path(p))

    def pick_train_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择训练模型", self.workspace, "Model Files (*.pt)")
        if p:
            self.train_model_edit.setText(self._to_rel_display_path(p))

    def pick_train_annotation_dir(self):
        p = QFileDialog.getExistingDirectory(self, "选择标注目录", self.workspace)
        if p:
            self.train_ann_edit.setText(self._to_rel_display_path(p))

    def pick_train_images_dir(self):
        p = QFileDialog.getExistingDirectory(self, "选择训练图片目录", self.workspace)
        if p:
            self.train_images_edit.setText(self._to_rel_display_path(p))

    def pick_train_dataset_out(self):
        p = QFileDialog.getExistingDirectory(self, "选择拆分数据集输出目录", self.workspace)
        if p:
            self.train_dataset_out_edit.setText(self._to_rel_display_path(p))

    def add_model(self):
        model = self.model_edit.text().strip()
        if not model:
            return
        for i in range(self.model_list.count()):
            if self.model_list.item(i).text() == model:
                return
        self.model_list.addItem(QListWidgetItem(model))
        self._set_invalid(self.model_list, invalid=False, blink=False)

    def remove_model(self):
        for item in self.model_list.selectedItems():
            self.model_list.takeItem(self.model_list.row(item))

    def validate(self):
        if not self.images_edit.text().strip():
            return ("请先选择图片目录", [self.images_edit])
        if self.model_list.count() == 0:
            return ("请至少添加一个模型", [self.model_list])
        if not self.prompt_text_edit.text().strip():
            return ("请填写提示词文本", [self.prompt_text_edit])
        if self.target_mode_combo.currentData() == "single" and (not self.composition_name_edit.text().strip()):
            return ("单目标模式下请填写统一类别名", [self.composition_name_edit])
        return (None, [])

    def _on_target_mode_changed(self, _index):
        is_single = self.target_mode_combo.currentData() == "single"
        self.composition_name_edit.setEnabled(is_single)
        if not is_single:
            self.composition_name_edit.setPlaceholderText("多目标模式下不需要统一类别名")
        else:
            self.composition_name_edit.setPlaceholderText("统一类别名，例如: 苹果")

    def _is_online(self):
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=0.6).close()
            return True
        except Exception:
            return False

    def _can_reach_llm_host(self):
        api_base = self.llm_base.text().strip()
        if not api_base:
            return True
        try:
            u = urlparse(api_base)
            host = u.hostname
            if not host:
                return True
            port = int(u.port or (443 if u.scheme == "https" else 80))
            socket.create_connection((host, port), timeout=0.9).close()
            return True
        except Exception:
            return False

    def _build_main_entry_command(self, subcommand):
        # In packaged mode, call the EXE itself; in source mode, call python main.py.
        if getattr(sys, "frozen", False):
            return [sys.executable, subcommand]
        return [sys.executable, "-u", "main.py", subcommand]

    def _precheck_annotate_inputs(self):
        image_dir_text = self.images_edit.text().strip()
        output_dir_text = self.output_edit.text().strip()

        if not image_dir_text:
            return ("图片目录错误", "请先选择图片目录", [self.images_edit])

        image_dir = self._resolve_input_path(image_dir_text)
        if not image_dir.exists() or not image_dir.is_dir():
            return ("图片目录错误", "图片地址不存在或不是有效目录", [self.images_edit])

        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        all_files = [p for p in image_dir.rglob("*") if p.is_file()]
        image_files = [p for p in all_files if p.suffix.lower() in supported_exts]

        if not image_files:
            if all_files:
                return (
                    "照片格式不支持",
                    "当前目录存在文件，但未发现受支持的图片格式。\n"
                    "支持格式: .jpg .jpeg .png .bmp .webp .tif .tiff",
                    [self.images_edit],
                )
            return ("图片目录为空", "图片地址内没有图片，请放入至少一张图片后重试", [self.images_edit])

        if not output_dir_text:
            return ("输出目录错误", "请先选择输出目录", [self.output_edit])

        output_dir = self._resolve_input_path(output_dir_text)
        if not output_dir.exists() or not output_dir.is_dir():
            return ("输出目录错误", "输出地址不存在，请先创建目录或重新选择", [self.output_edit])

        if self.enable_llm.isChecked() or self.force_llm.isChecked():
            if not self._is_online():
                return ("网络错误", "检测到当前未联网，无法使用 LLM 扩词。请联网后重试", [])
            if not self._can_reach_llm_host():
                return ("网络错误", "已联网但无法连接到 LLM 接口地址，请检查接口地址或网络代理设置", [self.llm_base])

        return None

    def run_prompt_assistant(self):
        desc = self.scene_desc_edit.toPlainText().strip()
        if not desc:
            self._flash_invalid_widgets([self.scene_desc_edit])
            self.scene_desc_edit.setFocus()
            self._show_info("提示", "请先填写目标描述，再点击智能建议")
            return
        if not self._is_online():
            self._show_warning("网络错误", "当前未联网，无法获取智能建议")
            return
        if not self._can_reach_llm_host():
            self._show_warning("网络错误", "无法连接到 LLM 接口地址，请检查网络或接口配置")
            return
        self.set_status("running", "生成建议中")
        self._begin_llm_progress("assistant")
        self._llm_progress_mark(2, "等待智能建议响应")
        self.append_log("STEP: 智能建议 - 正在请求LLM")
        self.btn_run.setEnabled(False)
        self.btn_train.setEnabled(False)

        self.assistant_worker = PromptAssistantWorker(
            description=desc,
            provider=self.llm_provider.currentText(),
            model=self.llm_model.text().strip() or "deepseek-chat",
            api_base=self.llm_base.text().strip(),
            api_key=self.llm_api_key.text().strip(),
            timeout=self.llm_timeout.value(),
        )
        self.assistant_worker.done.connect(self.on_prompt_assistant_done)
        self.assistant_worker.failed.connect(self.on_prompt_assistant_fail)
        self.assistant_worker.start()

    def on_prompt_assistant_done(self, data):
        self.assistant_worker = None
        self.btn_run.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.set_status("ready", "就绪")
        self._llm_progress_mark(3, "解析建议参数")
        self._llm_progress_mark(4, "应用建议到界面")
        self._finish_llm_progress(ok=True, text="智能建议完成")

        filtered_prompt = self._filter_overextended_prompt_text(
            str(data.get("prompt_text", "")).strip(),
            self.scene_desc_edit.toPlainText(),
        )
        self.prompt_text_edit.setText(filtered_prompt)
        self.negative_prompt_text_edit.setText(str(data.get("negative_prompt_text", "")).strip())

        need_more = bool(data.get("need_more_desc", False))
        advice = str(data.get("advice", "")).strip()
        self.llm_terms_view.appendPlainText(f"[智能建议] need_more_desc={need_more}")
        if advice:
            self.llm_terms_view.appendPlainText("[建议] " + advice)

        if need_more:
            self._show_info("描述建议", advice or "描述信息不足，请补充目标、背景、角度和排除对象")
        else:
            self._show_info("完成", "已生成并填入建议参数")

    def on_prompt_assistant_fail(self, msg):
        self.assistant_worker = None
        self.btn_run.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.set_status("error", "异常")
        self._finish_llm_progress(ok=False, text="智能建议失败")
        self._show_warning("智能建议失败", msg)

    def run_deploy_prompt_assistant(self):
        desc = self.deploy_target_desc_edit.toPlainText().strip()
        if not desc:
            self._flash_invalid_widgets([self.deploy_target_desc_edit])
            self.deploy_target_desc_edit.setFocus()
            self._show_info("提示", "请先填写目标描述，再点击智能建议")
            return
        if not self._is_online():
            self._show_warning("网络错误", "当前未联网，无法获取智能建议")
            return
        if not self._can_reach_llm_host():
            self._show_warning("网络错误", "无法连接到 LLM 接口地址，请检查网络或接口配置")
            return
        self.set_status("running", "生成建议中")
        self._begin_llm_progress("assistant")
        self.append_log("STEP: 快速部署智能建议 - 正在请求LLM")
        self.btn_run.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.btn_deploy_start.setEnabled(False)

        self.assistant_worker = PromptAssistantWorker(
            description=desc,
            provider=self.llm_provider.currentText(),
            model=self.llm_model.text().strip() or "deepseek-chat",
            api_base=self.llm_base.text().strip(),
            api_key=self.llm_api_key.text().strip(),
            timeout=self.llm_timeout.value(),
        )
        self.assistant_worker.done.connect(self.on_deploy_prompt_assistant_done)
        self.assistant_worker.failed.connect(self.on_deploy_prompt_assistant_fail)
        self.assistant_worker.start()

    def on_deploy_prompt_assistant_done(self, data):
        self.assistant_worker = None
        self.btn_run.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.btn_deploy_start.setEnabled(True)
        self.set_status("ready", "就绪")
        self._finish_llm_progress(ok=True, text="智能建议完成")
        filtered_prompt = self._filter_overextended_prompt_text(
            str(data.get("prompt_text", "")).strip(),
            self.deploy_target_desc_edit.toPlainText(),
        )
        self.deploy_prompt_edit.setText(filtered_prompt)
        self.deploy_negative_prompt_edit.setText(str(data.get("negative_prompt_text", "")).strip())
        self._show_info("完成", "已生成并填入快速部署建议参数")

    def on_deploy_prompt_assistant_fail(self, msg):
        self.assistant_worker = None
        self.btn_run.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.btn_deploy_start.setEnabled(True)
        self.set_status("error", "异常")
        self._finish_llm_progress(ok=False, text="智能建议失败")
        self._show_warning("智能建议失败", msg)

    def build_commands(self):
        models = [self.model_list.item(i).text().strip() for i in range(self.model_list.count())]
        out_base = self._resolve_input_path(self.output_edit.text().strip())
        cmds = []
        for model in models:
            model_name = Path(model).stem
            model_out = out_base / model_name
            preview_file = out_base / "_live_preview.jpg"
            try:
                preview_file.unlink(missing_ok=True)
            except Exception:
                pass
            cmd = self._build_main_entry_command("annotate") + [
                "--model",
                self._to_rel_display_path(self._resolve_input_path(model)),
                "--images",
                self._to_rel_display_path(self._resolve_input_path(self.images_edit.text().strip())),
                "--output",
                str(model_out),
                "--conf",
                str(self.conf.value()),
                "--min-box-conf",
                str(self.conf.value()),
                "--iou",
                str(self.iou.value()),
                "--imgsz",
                str(self.imgsz.value()),
                "--detect-max-terms",
                "36",
                "--device",
                "cpu" if self.force_cpu.isChecked() else self.device.currentText(),
                "--preview-file",
                str(preview_file),
            ]

            if self.target_mode_combo.currentData() == "single" and self.composition_name_edit.text().strip():
                cmd.append("--component-combine")
                cmd.extend(["--component-name", self.composition_name_edit.text().strip()])

            cmd.extend(["--target-mode", str(self.target_mode_combo.currentData() or "single")])
            cmd.extend(["--label-output-mode", str(self.label_output_mode_combo.currentData() or "both")])

            if self.pixel_refine.isChecked():
                cmd.append("--pixel-refine")

            if self.enable_box_expand.isChecked():
                cmd.append("--enable-box-expand")

            if self.hard_small_target.isChecked():
                cmd.append("--hard-small-target")

            if not self.enable_second_check.isChecked():
                cmd.append("--disable-second-check")

            prompt_text = self.prompt_text_edit.text().strip()
            negative_prompt_text = self.negative_prompt_text_edit.text().strip()
            if prompt_text:
                cmd.extend(["--prompt-text", prompt_text])

            if negative_prompt_text:
                cmd.extend(["--negative-prompt-text", negative_prompt_text])

            if self.enable_llm.isChecked() or self.force_llm.isChecked():
                cmd.append("--enable-llm-expand")
                cmd.extend(["--llm-provider", self.llm_provider.currentText()])
                if self.llm_model.text().strip():
                    cmd.extend(["--llm-model", self.llm_model.text().strip()])
                if self.llm_base.text().strip():
                    cmd.extend(["--llm-api-base", self.llm_base.text().strip()])
                if self.llm_api_key.text().strip():
                    cmd.extend(["--llm-api-key", self.llm_api_key.text().strip()])
                cmd.extend(["--llm-timeout", str(self.llm_timeout.value())])
                if self.domain.text().strip():
                    cmd.extend(["--domain-hint", self.domain.text().strip()])

            if self.force_llm.isChecked():
                cmd.append("--force-llm-expand")

            cmds.append(cmd)
        return cmds

    def pick_deploy_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择部署模型", self.workspace, "Model Files (*.pt *.onnx *.engine)")
        if p:
            self.deploy_model_edit.setText(self._to_rel_display_path(p))

    def validate_deploy(self):
        model = self.deploy_model_edit.text().strip()
        prompt = self.deploy_prompt_edit.text().strip()
        deploy_mode = str(self.deploy_target_mode_combo.currentData() or "single")
        if not model:
            return ("请填写部署模型路径", [self.deploy_model_edit])
        model_path = self._resolve_input_path(model)
        if not model_path.exists() or not model_path.is_file():
            return ("部署模型不存在，请重新选择", [self.deploy_model_edit])
        if not prompt:
            return ("请填写组合提示词", [self.deploy_prompt_edit])
        if deploy_mode == "single" and (not self.deploy_component_name.text().strip()):
            return ("单目标模式下请填写组合名称", [self.deploy_component_name])
        if self.deploy_enable_llm.isChecked() and (not self._is_online()):
            return ("网络错误：当前未联网，无法扩词", [])
        if self.deploy_enable_llm.isChecked() and (not self._can_reach_llm_host()):
            return ("网络错误：无法连接 LLM 接口地址", [self.llm_base])
        return (None, [])

    def _on_deploy_target_mode_changed(self, _index):
        is_single = str(self.deploy_target_mode_combo.currentData() or "single") == "single"
        self.deploy_component_enable.setChecked(is_single)
        self.deploy_component_name.setEnabled(is_single)
        if is_single:
            self.deploy_component_name.setPlaceholderText("统一类别名，例如: 目标")
        else:
            self.deploy_component_name.setPlaceholderText("多目标模式下不需要组合名称")

    def build_deploy_command(self):
        preview_path = self._resolve_input_path("dataset_annotations/_deploy_preview.jpg")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        self._deploy_preview_file = str(preview_path)
        cmd = self._build_main_entry_command("camera") + [
            "--model",
            self._to_rel_display_path(self._resolve_input_path(self.deploy_model_edit.text().strip())),
            "--camera",
            str(self.deploy_camera_id.value()),
            "--prompt-text",
            self.deploy_prompt_edit.text().strip(),
            "--negative-prompt-text",
            self.deploy_negative_prompt_edit.text().strip(),
            "--conf",
            str(self.deploy_conf.value()),
            "--imgsz",
            str(self.deploy_imgsz.value()),
            "--device",
            "cpu" if self.force_cpu.isChecked() else self.device.currentText(),
            "--preview-file",
            str(preview_path),
            "--no-window",
        ]
        if self.deploy_enable_llm.isChecked():
            cmd.append("--enable-llm-expand")
            cmd.extend(["--llm-provider", self.llm_provider.currentText()])
            if self.llm_model.text().strip():
                cmd.extend(["--llm-model", self.llm_model.text().strip()])
            if self.llm_base.text().strip():
                cmd.extend(["--llm-api-base", self.llm_base.text().strip()])
            if self.llm_api_key.text().strip():
                cmd.extend(["--llm-api-key", self.llm_api_key.text().strip()])
            cmd.extend(["--llm-timeout", str(self.llm_timeout.value())])
            if self.domain.text().strip():
                cmd.extend(["--domain-hint", self.domain.text().strip()])

        cmd.extend(["--target-mode", str(self.deploy_target_mode_combo.currentData() or "single")])

        if self.deploy_target_mode_combo.currentData() == "single" and self.deploy_component_name.text().strip():
            cmd.append("--component-combine")
            cmd.extend(["--component-name", self.deploy_component_name.text().strip()])
        return cmd

    def _ensure_deploy_preview_timer(self):
        if self._deploy_preview_timer is None:
            self._deploy_preview_timer = QTimer(self)
            self._deploy_preview_timer.setInterval(140)
            self._deploy_preview_timer.timeout.connect(self._tick_deploy_preview)

    def _tick_deploy_preview(self):
        if self.current_task != "deploy":
            return
        if not self._deploy_preview_file:
            return
        try:
            mt = Path(self._deploy_preview_file).stat().st_mtime
        except Exception:
            return
        if mt <= self._deploy_preview_mtime:
            return
        self._deploy_preview_mtime = mt
        self.update_preview(self._deploy_preview_file)

    def start_deploy(self):
        self._clear_invalid_marks()
        err, fields = self.validate_deploy()
        if err:
            self._flash_invalid_widgets(fields)
            self._show_warning("参数错误", err)
            return

        self.current_task = "deploy"
        self.tabs.setCurrentIndex(2)
        self.clear_output()
        self._reset_deploy_stats_view()
        self._deploy_preview_mtime = -1.0
        self.deploy_preview.clear()
        self.deploy_preview.setText("部署预览区")
        self.deploy_state_text.setText("运行中")
        self.deploy_state_icon.setStyleSheet("color:#2563eb;")
        self.set_status("running", "快速部署中")
        self.update_step("准备启动摄像头部署")

        command = self.build_deploy_command()
        self.worker = AnnotationWorker([command], self.workspace)
        self.worker.log.connect(self.append_log)
        self.worker.step.connect(self.update_step)
        self.worker.stats.connect(self.update_deploy_stats)
        self.worker.preview.connect(self.update_preview)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)

        self.btn_run.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.btn_deploy_start.setEnabled(False)
        self.btn_deploy_stop.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self._ensure_deploy_preview_timer()
        self._deploy_preview_timer.start()
        self.worker.start()

    def validate_train(self):
        ann = self.train_ann_edit.text().strip() or self.output_edit.text().strip()
        imgs = self.train_images_edit.text().strip() or self.images_edit.text().strip()
        model = self.train_model_edit.text().strip() or self.model_edit.text().strip()
        if not ann:
            return ("请填写标注目录", [self.train_ann_edit])
        if not imgs:
            return ("请填写训练图片目录", [self.train_images_edit])
        if not model:
            return ("请填写训练模型路径", [self.train_model_edit])

        ann_dir = self._resolve_input_path(ann)
        if not ann_dir.exists() or not ann_dir.is_dir():
            return ("标注目录不存在或不是有效文件夹", [self.train_ann_edit])

        img_dir = self._resolve_input_path(imgs)
        if not img_dir.exists() or not img_dir.is_dir():
            return ("训练图片目录不存在或不是有效文件夹", [self.train_images_edit])

        model_path = self._resolve_input_path(model)
        if not model_path.exists() or not model_path.is_file():
            return ("训练模型不存在，请重新选择", [self.train_model_edit])

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        has_images = any(p.is_file() and p.suffix.lower() in image_exts for p in img_dir.rglob("*"))
        if not has_images:
            return ("训练图片目录内未发现可用图片", [self.train_images_edit])

        def _resolve_ann_root(base_dir: Path):
            # 兼容直接传根目录或上级目录的场景。
            candidates = [base_dir]
            try:
                candidates.extend([p for p in base_dir.iterdir() if p.is_dir()])
            except Exception:
                pass
            for c in candidates:
                if (c / "classes.txt").exists() and ((c / "labels_det").exists() or (c / "labels_seg").exists()):
                    return c
            return base_dir

        def _is_seg_model(model_file: Path):
            name = model_file.name.lower()
            return ("-seg" in name) or ("_seg" in name)

        ann_root = _resolve_ann_root(ann_dir)
        expect_label_dir = ann_root / ("labels_seg" if _is_seg_model(model_path) else "labels_det")

        if not expect_label_dir.exists() or not expect_label_dir.is_dir():
            kind = "分割(labels_seg)" if _is_seg_model(model_path) else "检测(labels_det)"
            return (
                f"目标文件夹内未找到对应类别标注目录: {kind}，请检查数据后重试",
                [self.train_ann_edit, self.train_model_edit],
            )

        label_files = [
            p for p in expect_label_dir.glob("*.txt")
            if p.is_file() and p.name.lower() != "classes.txt"
        ]
        if not label_files:
            kind = "分割(labels_seg)" if _is_seg_model(model_path) else "检测(labels_det)"
            return (
                f"目标文件夹内没有对应类别的标注文件({kind})，请检查标注是否生成",
                [self.train_ann_edit, self.train_model_edit],
            )
        return (None, [])

    def build_train_command(self):
        ann = self.train_ann_edit.text().strip() or self.output_edit.text().strip()
        imgs = self.train_images_edit.text().strip() or self.images_edit.text().strip()
        model = self.train_model_edit.text().strip() or self.model_edit.text().strip()
        dataset_out = self.train_dataset_out_edit.text().strip() or "train_dataset"

        cmd = self._build_main_entry_command("train") + [
            "--annotation-dir",
            self._to_rel_display_path(self._resolve_input_path(ann)),
            "--images",
            self._to_rel_display_path(self._resolve_input_path(imgs)),
            "--dataset-out",
            self._to_rel_display_path(self._resolve_input_path(dataset_out)),
            "--rebuild-dataset",
            "--model",
            self._to_rel_display_path(self._resolve_input_path(model)),
            "--epochs",
            str(self.train_epochs.value()),
            "--batch",
            str(self.train_batch.value()),
            "--imgsz",
            str(self.imgsz.value()),
            "--val-ratio",
            str(self.train_val_ratio.value()),
            "--max-ram-percent",
            str(self.train_max_ram.value()),
            "--max-vram-percent",
            str(self.train_max_vram.value()),
            "--device",
            "cpu" if self.force_cpu.isChecked() else self.device.currentText(),
        ]

        fmt_text = self.train_dataset_format.currentText()
        if "XML" in fmt_text:
            cmd.extend(["--dataset-format", "voc_xml"])
        else:
            cmd.extend(["--dataset-format", "yolo_yaml"])

        if self.train_export_onnx.isChecked():
            cmd.append("--export-onnx")
        return cmd

    def start_train(self):
        self._clear_invalid_marks()
        err, fields = self.validate_train()
        if err:
            self._flash_invalid_widgets(fields)
            self._show_warning("参数错误", err)
            return

        self.current_task = "train"
        self.tabs.setCurrentIndex(1)
        self.clear_output()
        self._run_paused = False
        self.update_step("准备启动训练")
        self.set_status("running", "训练中")
        self.bar.setValue(0)

        command = self.build_train_command()
        self.worker = AnnotationWorker([command], self.workspace)
        self.worker.log.connect(self.append_log)
        self.worker.step.connect(self.update_step)
        self.worker.progress.connect(self.update_model_progress)
        self.worker.stats.connect(self.update_stats)
        self.worker.preview.connect(self.update_preview)
        self.worker.llm_prompt.connect(self.append_llm_prompt)
        self.worker.paused.connect(self.on_run_paused)
        self.worker.resumed.connect(self.on_run_resumed)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)
        self.btn_run.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_train_pause.setEnabled(True)
        self.btn_train_pause.setText(self._tr("train_pause"))
        self.btn_train_stop.setEnabled(True)
        self.worker.start()

    def toggle_train_pause(self):
        if self.current_task != "train" or not self.worker:
            self.append_log("STEP: 训练任务未运行，无法暂停/继续")
            return
        if self._run_paused:
            if self.worker.resume_now():
                self._run_paused = False
                self.btn_train_pause.setText(self._tr("train_pause"))
            else:
                self.append_log("STEP: 继续失败（缺少进程恢复能力）")
        else:
            if self.worker.pause_now():
                self._run_paused = True
                self.btn_train_pause.setText(self._tr("train_resume"))
            else:
                self.append_log("STEP: 暂停失败（缺少进程挂起能力）")

    def stop_train_task(self):
        if self.current_task != "train":
            self.append_log("STEP: 当前不是训练任务")
            return
        if self.worker:
            self.worker.stop()
            self.append_log("STEP: 已请求终止训练")

    def start(self):
        self._clear_invalid_marks()
        self.btn_run.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.set_status("running", "准备中")
        self.update_step("检查输入参数")
        QApplication.processEvents()

        issue = self._precheck_annotate_inputs()
        if issue is not None:
            title, message, fields = issue
            self._flash_invalid_widgets(fields)
            self.btn_run.setEnabled(True)
            self.btn_train.setEnabled(True)
            self.set_status("ready", "就绪")
            self._show_warning(title, message)
            return

        err, fields = self.validate()
        if err:
            self._flash_invalid_widgets(fields)
            self.btn_run.setEnabled(True)
            self.btn_train.setEnabled(True)
            self.set_status("ready", "就绪")
            self._show_warning("参数错误", err)
            return

        self.current_task = "annotate"
        self.tabs.setCurrentIndex(0)
        self.clear_output()
        self.preview_paused = False
        self._run_pause_requested = False
        self._run_paused = False
        self.btn_pause_preview.setText(self._tr("btn_pause"))
        self.preview.clear()
        self.preview.setText("预览区")
        self.update_step("准备启动标注")
        self.set_status("running", "运行中")
        self.bar.setValue(0)
        self._stats_chart_start_ts = time.monotonic()
        if self.enable_llm.isChecked() or self.force_llm.isChecked():
            self._begin_llm_progress("annotate")
            self._llm_progress_mark(1, "初始化扩词任务")
        else:
            self._set_llm_progress(100, "未启用扩词")

        commands = self.build_commands()
        self.worker = AnnotationWorker(commands, self.workspace)
        self.worker.log.connect(self.append_log)
        self.worker.step.connect(self.update_step)
        self.worker.progress.connect(self.update_model_progress)
        self.worker.stats.connect(self.update_stats)
        self.worker.preview.connect(self.update_preview)
        self.worker.llm_prompt.connect(self.append_llm_prompt)
        self.worker.paused.connect(self.on_run_paused)
        self.worker.resumed.connect(self.on_run_resumed)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)
        self.btn_run.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
        if self._deploy_preview_timer is not None:
            self._deploy_preview_timer.stop()

    def on_run_paused(self, msg):
        self._run_pause_requested = False
        self._run_paused = True
        if self.current_task == "train":
            self.btn_train_pause.setText(self._tr("train_resume"))
        else:
            self.btn_pause_preview.setText("继续识别")
        self.set_status("ready", "已暂停")
        self.update_step("暂停中")
        self.append_log("STEP: " + str(msg or "任务已暂停"))

    def on_run_resumed(self, msg):
        self._run_paused = False
        self._run_pause_requested = False
        if self.current_task == "train":
            self.btn_train_pause.setText(self._tr("train_pause"))
        else:
            self.btn_pause_preview.setText(self._tr("btn_pause"))
        self.set_status("running", "运行中")
        self.update_step("继续运行")
        self.append_log("STEP: " + str(msg or "继续运行"))

    def on_done(self):
        self._run_pause_requested = False
        self._run_paused = False
        self.btn_pause_preview.setText(self._tr("btn_pause"))
        self._set_mascot_mode("done", "完成")
        self.set_status("ready", "就绪")
        self.update_step("已完成")
        self.btn_run.setEnabled(True)
        self.btn_train.setEnabled(True)
        if hasattr(self, "btn_deploy_start"):
            self.btn_deploy_start.setEnabled(True)
        if hasattr(self, "btn_deploy_stop"):
            self.btn_deploy_stop.setEnabled(False)
        self.btn_stop.setEnabled(False)
        if hasattr(self, "btn_train_pause"):
            self.btn_train_pause.setEnabled(False)
            self.btn_train_pause.setText(self._tr("train_pause"))
        if hasattr(self, "btn_train_stop"):
            self.btn_train_stop.setEnabled(False)
        self.bar.setValue(100)
        if self._deploy_preview_timer is not None:
            self._deploy_preview_timer.stop()
        if self._llm_progress_active:
            self._finish_llm_progress(ok=True, text="扩词流程结束")
        if self.current_task == "train":
            detail = "训练任务已完成"
            if self.train_best_path:
                detail += f"\nBEST: {self.train_best_path}"
            if self.train_onnx_path:
                detail += f"\nONNX: {self.train_onnx_path}"
            self._show_info("完成", detail)
        else:
            if self.current_task == "deploy":
                if hasattr(self, "deploy_state_text"):
                    self.deploy_state_text.setText("已停止")
                if hasattr(self, "deploy_state_icon"):
                    self.deploy_state_icon.setStyleSheet("color:#16a34a;")
                self._show_info("完成", "快速部署任务已结束")
            else:
                self._show_info("完成", "全部标注任务已完成")
                if self.auto_reset_stats.isChecked():
                    self.reset_stats_view()

    def on_fail(self, msg):
        self._run_pause_requested = False
        self._run_paused = False
        self.btn_pause_preview.setText(self._tr("btn_pause"))
        self._set_mascot_mode("error", "异常")
        self.set_status("error", "异常")
        self.update_step("已中断")
        self.btn_run.setEnabled(True)
        self.btn_train.setEnabled(True)
        if hasattr(self, "btn_deploy_start"):
            self.btn_deploy_start.setEnabled(True)
        if hasattr(self, "btn_deploy_stop"):
            self.btn_deploy_stop.setEnabled(False)
        self.btn_stop.setEnabled(False)
        if hasattr(self, "btn_train_pause"):
            self.btn_train_pause.setEnabled(False)
            self.btn_train_pause.setText(self._tr("train_pause"))
        if hasattr(self, "btn_train_stop"):
            self.btn_train_stop.setEnabled(False)
        if self._deploy_preview_timer is not None:
            self._deploy_preview_timer.stop()
        if self.current_task == "deploy":
            if hasattr(self, "deploy_state_text"):
                self.deploy_state_text.setText("异常")
            if hasattr(self, "deploy_state_icon"):
                self.deploy_state_icon.setStyleSheet("color:#dc2626;")
        if self._llm_progress_active:
            self._finish_llm_progress(ok=False, text="扩词/任务中断")
        self._show_warning("任务结束", msg)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
