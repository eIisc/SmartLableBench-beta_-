from ultralytics import YOLO
import argparse
import cv2
import json
import numpy as np
import os
import re
from pathlib import Path
from prompt_expander import expand_prompts
import multiprocessing
from urllib import request, error

try:
    import torch
except Exception:
    torch = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 注意部位 HSV 表：用于在像素延展阶段提供颜色先验（H:0-179, S/V:0-255）。
ATTENTION_PART_HSV_TABLE = {
    "green_fruit_skin": [
        {"lower": [30, 35, 35], "upper": [95, 255, 255]},
    ],
    "yellow_spot": [
        {"lower": [16, 35, 35], "upper": [38, 255, 255]},
    ],
    "dark_crack": [
        {"lower": [0, 0, 0], "upper": [179, 140, 85]},
    ],
    "metal_gray": [
        {"lower": [0, 0, 45], "upper": [179, 55, 225]},
    ],
}

SUV_FEATURE_TERMS = {
    "宽大轮胎", "车身厚重", "离地间隙大", "接近角大", "离去角大", "侧窗方正", "车头厚重",
    "方形轮拱", "车顶纵轨", "车顶横杆", "侧踏板", "外挂备胎", "后门侧开", "后风挡可开",
    "大面积进气格栅", "高腰线", "方形尾灯", "高位刹车灯", "车顶扰流板", "方形后视镜",
    "宽体车身", "轮眉外扩", "车身防擦条", "越野轮胎花纹",
    "wide off-road tires", "heavy body", "large approach angle", "large departure angle",
    "square side windows", "thick front fascia", "square wheel arches", "roof rails", "roof cross bars",
    "side steps", "external spare tire", "side-opening rear door", "openable rear window", "large front grille",
    "high beltline", "square tail lamps", "high-mounted brake light", "roof spoiler", "square side mirrors",
    "wide-body", "flared fenders", "side cladding", "off-road tire tread",
}

SEDAN_FEATURE_TERMS = {
    "three-box car", "lower ground clearance", "lower body", "longer trunk section",
    "smooth roofline", "smaller wheel arches", "long wheelbase look", "low stance",
    "longer trunk than suv", "smaller wheel arch than suv",
    "front-view sedan", "frontal sedan", "front-facing sedan", "sedan front fascia",
    "sedan grille", "sedan headlights", "sedan hood line",
    "三厢车身", "低离地间隙", "较低车身", "较长后备箱段", "平顺车顶线条", "小轮拱",
    "轿车正面", "前视轿车", "正面小轿车", "轿车前脸", "轿车进气格栅", "轿车前大灯",
    "轿车引擎盖线条", "三厢车前视轮廓",
}

GROUP_SYNONYMS = {
    "suv": {"suv", "suv车", "sport utility vehicle", "越野车", "SUV车"},
    "sedan": {"sedan", "saloon", "three-box car", "轿车", "三厢车"},
    "parrot": {"parrot", "鹦鹉"},
    "mpv": {"mpv", "minivan", "multi purpose vehicle", "面包车", "商务车"},
    "truck": {"truck", "lorry", "卡车", "货车"},
}

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
    "苹果": "apple",
    "青苹果": "green apple",
    "披萨": "pizza",
    "比萨": "pizza",
    "鹦鹉": "parrot",
    "匕首": "dagger",
    "短剑": "dagger",
    "轿车": "sedan",
    "轿车正面": "front-view sedan",
    "正面轿车": "front-view sedan",
    "前视轿车": "front-view sedan",
    "小轿车": "sedan",
    "俯视轿车": "top-view sedan",
    "俯视小轿车": "top-view sedan",
    "顶视轿车": "top-view sedan",
    "航拍轿车": "aerial-view sedan",
    "suv车": "suv",
}

CN_EN_TRANSLATIONS = {
    "披萨": "pizza",
    "比萨": "pizza",
    "鹦鹉": "parrot",
    "匕首": "dagger",
    "短剑": "dagger",
    "suv车": "suv",
    "越野车": "suv",
    "轿车": "sedan",
    "轿车正面": "front-view sedan",
    "正面轿车": "front-view sedan",
    "前视轿车": "front-view sedan",
    "小轿车": "sedan",
    "俯视轿车": "top-view sedan",
    "俯视小轿车": "top-view sedan",
    "顶视轿车": "top-view sedan",
    "航拍轿车": "aerial-view sedan",
    "三厢车": "three-box car",
    "宽大轮胎": "wide off-road tires",
    "车身厚重": "heavy body",
    "离地间隙大": "large clearance",
    "接近角大": "large approach angle",
    "离去角大": "large departure angle",
    "侧窗方正": "square side windows",
    "车头厚重": "thick front fascia",
    "方形轮拱": "square wheel arches",
    "车顶纵轨": "roof rails",
    "车顶横杆": "roof cross bars",
    "侧踏板": "side steps",
    "外挂备胎": "external spare tire",
    "后门侧开": "side-opening rear door",
    "后风挡可开": "openable rear window",
    "大面积进气格栅": "large front grille",
    "高腰线": "high beltline",
    "方形尾灯": "square tail lamps",
    "高位刹车灯": "high-mounted brake light",
    "车顶扰流板": "roof spoiler",
    "方形后视镜": "square side mirrors",
    "宽体车身": "wide-body",
    "轮眉外扩": "flared fenders",
    "车身防擦条": "side cladding",
    "越野轮胎花纹": "off-road tire tread",
}

AMBIGUOUS_TERM_HINTS = {
    "formula one": "这是一个有异议词，建议改为: race car / formula race car / open-wheel race car",
    "f1": "这是一个有异议词，建议改为: race car / formula race car / open-wheel race car",
}

AMBIGUOUS_TERM_FALLBACKS = {
    "formula one": {
        "suggest": "open-wheel race car",
        "highlights": [
            "large exposed tires",
            "front wing",
            "rear wing",
            "low single-seater body",
            "open cockpit",
        ],
    },
    "f1": {
        "suggest": "open-wheel race car",
        "highlights": [
            "large exposed tires",
            "front wing",
            "rear wing",
            "low single-seater body",
            "open cockpit",
        ],
    },
}


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def has_chinese(text):
    return bool(re.search(r"[\u4e00-\u9fff]", str(text)))


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


def tokenize_for_match(text):
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalize_text(text))


def translate_to_english(term):
    t = normalize_text(term)
    return CN_EN_TRANSLATIONS.get(t, term)


def translate_prompt_list_to_english(items):
    translated = []
    for item in items:
        mapped = PROMPT_ALIASES.get(normalize_text(item), item)
        mapped = translate_to_english(mapped)
        translated.append(mapped)
    return dedupe_keep_order(translated)


def warn_ambiguous_terms(items, prefix="提示词"):
    for item in items:
        key = normalize_text(item)
        if key in AMBIGUOUS_TERM_HINTS:
            print(f"{prefix}提醒: {item} -> {AMBIGUOUS_TERM_HINTS[key]}")


def llm_force_translate_terms(terms, args):
    pending = [t for t in dedupe_keep_order(terms) if has_chinese(t)]
    if not pending:
        return {}

    provider = (args.llm_provider or "openai-compatible").strip().lower()
    model = args.llm_model
    timeout = max(8, int(args.llm_timeout or 30))

    def parse_json_map(text):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
        except Exception:
            return {}
        return {}

    system_prompt = (
        "你是检测提示词翻译器。"
        "将中文目标词翻译为最短、最标准的英文检测词。"
        "仅输出 JSON 对象，不要解释。"
        "键是原词，值是英文词组。"
    )
    user_prompt = (
        "请翻译以下词为英文检测提示词，严格JSON输出:\n"
        + "\n".join(pending)
    )

    if provider == "openai-compatible":
        api_base = (args.llm_api_base or "https://api.deepseek.com").rstrip("/")
        api_key = args.llm_api_key
        if not api_key:
            return {}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        req = request.Request(
            url=f"{api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            obj = json.loads(body)
            content = obj["choices"][0]["message"]["content"]
            return parse_json_map(content)
        except Exception:
            return {}

    if provider == "ollama":
        api_base = (args.llm_api_base or "http://127.0.0.1:11434").rstrip("/")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        req = request.Request(
            url=f"{api_base}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=max(timeout, 12)) as resp:
                body = resp.read().decode("utf-8")
            obj = json.loads(body)
            content = obj["message"]["content"]
            return parse_json_map(content)
        except Exception:
            return {}

    return {}


def force_translate_prompt_list(items, args, prefix="提示词"):
    current = dedupe_keep_order(items)
    pending = [x for x in current if has_chinese(x)]
    if not pending:
        return current

    translated_map = llm_force_translate_terms(pending, args)
    out = []
    for term in current:
        if not has_chinese(term):
            out.append(term)
            continue
        t = translated_map.get(term, "").strip()
        if t and not has_chinese(t):
            out.append(t)
            print(f"{prefix}强制翻译: {term} -> {t}")
        else:
            # LLM翻译失败时保留词典翻译结果，保证流程可继续
            fallback = translate_to_english(term)
            out.append(fallback)
            if has_chinese(fallback):
                print(f"{prefix}强制翻译失败(保留原词): {term}")
            else:
                print(f"{prefix}词典翻译: {term} -> {fallback}")
    return dedupe_keep_order(out)


def llm_term_gate_for_yolo(terms, args, prefix="提示词"):
    terms = dedupe_keep_order(terms)
    if not terms:
        return terms

    provider = (args.llm_provider or "openai-compatible").strip().lower()
    model = args.llm_model
    timeout = max(8, int(args.llm_timeout or 30))

    # 仅在可调用LLM时执行门控
    if provider == "openai-compatible" and not args.llm_api_key:
        return terms

    system_prompt = (
        "你是YOLO开放词表提示词审核器。"
        "判断每个词是否适合直接作为目标检测类别词。"
        "若不适合，给出一个更可检测的替代词（英文优先），并给出关键亮点元素词组用于拟合检测。"
        "必须输出JSON数组，每项格式: "
        "{\"term\":\"原词\",\"ok\":true/false,\"reason\":\"简短原因\",\"suggest\":\"替代词或空\",\"highlights\":[\"元素词1\",\"元素词2\"]}。"
        "highlights 要尽量是可见部件或可见结构，不要抽象概念。"
        "不要输出任何额外说明。"
    )
    user_prompt = "待审核词: " + ", ".join(terms)

    def generate_highlights_for_term(base_term):
        base_term = str(base_term).strip()
        if not base_term:
            return []

        h_system = (
            "你是YOLO开放词表关键亮点生成器。"
            "给定一个目标词，输出可见且可检测的关键亮点元素词组。"
            "仅输出JSON数组，如 [\"element1\",\"element2\"]。"
            "不要输出object/item/target/outline/contour等空泛词。"
            "优先输出部件、结构、外形、典型可见特征。"
        )
        h_user = f"目标词: {base_term}"

        def parse_arr(text):
            try:
                arr = json.loads(text)
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                pass
            m = re.search(r"\[[\s\S]*\]", text)
            if not m:
                return []
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                return []
            return []

        if provider == "openai-compatible":
            api_base = (args.llm_api_base or "https://api.deepseek.com").rstrip("/")
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": h_system},
                    {"role": "user", "content": h_user},
                ],
                "temperature": 0.0,
            }
            req = request.Request(
                url=f"{api_base}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {args.llm_api_key}",
                },
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                obj = json.loads(body)
                content = obj["choices"][0]["message"]["content"]
                return parse_arr(content)[:8]
            except Exception:
                return []

        if provider == "ollama":
            api_base = (args.llm_api_base or "http://127.0.0.1:11434").rstrip("/")
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": h_system},
                    {"role": "user", "content": h_user},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            }
            req = request.Request(
                url=f"{api_base}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=max(timeout, 12)) as resp:
                    body = resp.read().decode("utf-8")
                obj = json.loads(body)
                content = obj["message"]["content"]
                return parse_arr(content)[:8]
            except Exception:
                return []

        return []

    def parse_gate_result(content):
        try:
            arr = json.loads(content)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
        m = re.search(r"\[[\s\S]*\]", content)
        if not m:
            return []
        try:
            arr = json.loads(m.group(0))
            return arr if isinstance(arr, list) else []
        except Exception:
            return []

    gate_rows = []
    if provider == "openai-compatible":
        api_base = (args.llm_api_base or "https://api.deepseek.com").rstrip("/")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        req = request.Request(
            url=f"{api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {args.llm_api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            obj = json.loads(body)
            content = obj["choices"][0]["message"]["content"]
            gate_rows = parse_gate_result(content)
        except Exception:
            return terms
    elif provider == "ollama":
        api_base = (args.llm_api_base or "http://127.0.0.1:11434").rstrip("/")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        req = request.Request(
            url=f"{api_base}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=max(timeout, 12)) as resp:
                body = resp.read().decode("utf-8")
            obj = json.loads(body)
            content = obj["message"]["content"]
            gate_rows = parse_gate_result(content)
        except Exception:
            return terms
    else:
        return terms

    if not gate_rows:
        return terms

    by_term = {}
    for row in gate_rows:
        if not isinstance(row, dict):
            continue
        term = str(row.get("term", "")).strip()
        if not term:
            continue
        by_term[normalize_text(term)] = row

    def anchor_highlights(base_term, highlights):
        base = normalize_text(base_term)
        base_tokens = set(tokenize_for_match(base))
        anchored = []
        for h in highlights:
            hs = str(h).strip()
            if not hs:
                continue
            ht = normalize_text(hs)
            if base_tokens and not any(tok in ht for tok in base_tokens):
                hs = f"{base_term} {hs}"
            anchored.append(hs)
        return dedupe_keep_order(anchored)[:8]

    out = []
    for term in terms:
        row = by_term.get(normalize_text(term))
        if not row:
            out.append(term)
            continue
        ok = bool(row.get("ok", True))
        reason = str(row.get("reason", "")).strip()
        suggest = str(row.get("suggest", "")).strip()
        highlights = row.get("highlights", [])
        if not isinstance(highlights, list):
            highlights = []

        fallback = AMBIGUOUS_TERM_FALLBACKS.get(normalize_text(term), None)
        if not suggest and fallback:
            suggest = str(fallback.get("suggest", "")).strip()
        if not highlights and fallback:
            highlights = list(fallback.get("highlights", []))

        if ok:
            out.append(term)
            continue
        if suggest:
            if not highlights:
                highlights = generate_highlights_for_term(suggest)
            out.append(suggest)
            for h in anchor_highlights(suggest, highlights):
                out.append(h)
            print(f"{prefix}门控替换: {term} -> {suggest}" + (f" | 原因: {reason}" if reason else ""))
        else:
            base_term = term
            if not highlights:
                highlights = generate_highlights_for_term(base_term)
            out.append(base_term)
            for h in anchor_highlights(base_term, highlights):
                out.append(h)
            print(f"{prefix}门控拟合: {term} -> {base_term}+highlights" + (f" | 原因: {reason}" if reason else ""))

    out = dedupe_keep_order(out)
    return out if out else terms


def parse_inline_prompts(prompt_text):
    if not prompt_text:
        return []
    # 仅按显式分隔符切分，保留短语内部空格，如 "front-view sedan"
    tokens = [t.strip() for t in re.split(r"[，,、;；\n\r|]+", prompt_text) if t.strip()]
    return translate_prompt_list_to_english(tokens)


def build_open_vocab_class_id_map(detect_prompts, seed_prompts):
    seeds = dedupe_keep_order(seed_prompts)
    if not seeds:
        seeds = ["object"]

    normalized_synonyms = {
        k: {normalize_text(x) for x in v}
        for k, v in GROUP_SYNONYMS.items()
    }

    def to_group_key(text):
        t = normalize_text(text)
        for key, vocab in normalized_synonyms.items():
            if t in vocab:
                return key
        return t

    def related(seed, term):
        s_norm = normalize_text(seed)
        t_norm = normalize_text(term)
        if s_norm == t_norm:
            return True
        if s_norm and (s_norm in t_norm or t_norm in s_norm):
            return True
        if to_group_key(seed) == to_group_key(term):
            return True
        s_tokens = set(tokenize_for_match(seed))
        t_tokens = set(tokenize_for_match(term))
        return len(s_tokens & t_tokens) > 0

    index_to_seed_id = {}
    for idx, term in enumerate(detect_prompts):
        mapped = None
        for sid, seed in enumerate(seeds):
            if related(seed, term):
                mapped = sid
                break
        # 非提示词相关扩展项不建立映射，避免被错误写入到任一种子类别。
        if mapped is not None:
            index_to_seed_id[idx] = mapped

    return index_to_seed_id, seeds


def load_prompts_from_file(prompt_file):
    items = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if normalize_text(line) in {"all", "*", "全部"}:
                continue
            items.append(line)
    return translate_prompt_list_to_english(items)


def resolve_runtime_device(args):
    req = (args.device or "auto").strip().lower()

    if req == "auto":
        if args.cpu_threads > 0:
            return "cpu"
        if torch is not None and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    if req.startswith("cuda"):
        if torch is None or not torch.cuda.is_available():
            print("CUDA不可用，自动回退到CPU")
            return "cpu"
        return req

    if req == "cpu":
        return "cpu"

    return "cpu"


def load_prompt_list(args):
    if args.prompts_file:
        if not os.path.exists(args.prompts_file):
            raise FileNotFoundError(f"未找到提示词文件: {args.prompts_file}")
        prompts = load_prompts_from_file(args.prompts_file)
        prompts = force_translate_prompt_list(prompts, args, prefix="提示词")
        prompts = llm_term_gate_for_yolo(prompts, args, prefix="提示词")
        warn_ambiguous_terms(prompts, prefix="提示词")
        return prompts, f"文件 {args.prompts_file}"

    if args.prompt_text and args.prompt_text.strip():
        prompts = parse_inline_prompts(args.prompt_text)
        prompts = force_translate_prompt_list(prompts, args, prefix="提示词")
        prompts = llm_term_gate_for_yolo(prompts, args, prefix="提示词")
        warn_ambiguous_terms(prompts, prefix="提示词")
        return prompts, "命令行/GUI 文本输入"

    auto_file = "your_4000_prompts.txt"
    if os.path.exists(auto_file):
        prompts = load_prompts_from_file(auto_file)
        prompts = force_translate_prompt_list(prompts, args, prefix="提示词")
        prompts = llm_term_gate_for_yolo(prompts, args, prefix="提示词")
        warn_ambiguous_terms(prompts, prefix="提示词")
        return prompts, f"自动文件 {auto_file}"

    return [], "无"


def load_negative_prompt_list(args):
    if args.negative_prompts_file:
        if not os.path.exists(args.negative_prompts_file):
            raise FileNotFoundError(f"未找到负提示词文件: {args.negative_prompts_file}")
        negatives = load_prompts_from_file(args.negative_prompts_file)
        negatives = force_translate_prompt_list(negatives, args, prefix="负提示词")
        warn_ambiguous_terms(negatives, prefix="负提示词")
        return negatives, f"文件 {args.negative_prompts_file}"

    if args.negative_prompt_text and args.negative_prompt_text.strip():
        negatives = parse_inline_prompts(args.negative_prompt_text)
        negatives = force_translate_prompt_list(negatives, args, prefix="负提示词")
        warn_ambiguous_terms(negatives, prefix="负提示词")
        return negatives, "命令行/GUI 文本输入"

    return [], "无"


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


def strict_negative_post_filter(seed_negatives, expanded_negatives, max_terms=24):
    seeds = dedupe_keep_order(seed_negatives)
    expanded = dedupe_keep_order(expanded_negatives)

    anchors = set()
    for s in seeds:
        for tk in tokenize_for_match(s):
            if len(tk) >= 2:
                anchors.add(tk)

    # 针对车辆类负提示词给出更稳的同义词白名单
    allow_extra = set()
    s_norm = [normalize_text(x) for x in seeds]
    if any(("sedan" in x) or ("轿车" in x) for x in s_norm):
        allow_extra.update({"sedan", "saloon", "three-box car", "轿车", "三厢车"})
    if any(("suv" in x) or ("越野" in x) for x in s_norm):
        allow_extra.update({"suv", "sport utility vehicle", "越野车", "SUV车"})

    # 负提示词严格模式：过滤仅属性词，尽量保留实体相关词
    generic_attr = {
        "green", "red", "black", "white", "blue", "gray", "silver",
        "球形", "方形", "高", "宽", "窄", "光滑", "纹理", "斑点",
        "spherical", "boxy", "smooth", "texture", "speckles",
    }

    out = []
    out.extend(seeds)
    for term in expanded:
        t = normalize_text(term)
        tks = tokenize_for_match(term)
        if not t:
            continue
        if t in generic_attr:
            continue
        if any((tk in anchors) for tk in tks):
            out.append(term)
            continue
        if t in {normalize_text(x) for x in allow_extra}:
            out.append(term)

    out = dedupe_keep_order(out)
    return out[:max_terms]


def expand_negative_prompts_strict(seed_negatives, args):
    negatives = dedupe_keep_order(seed_negatives)
    if not negatives:
        return negatives, {"mode": "none", "error": None}

    if not args.enable_llm_expand:
        return negatives, {"mode": "seed_only", "error": None}

    expanded, info = expand_prompts(
        seed_prompts=negatives,
        enable_llm=True,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
        llm_max_terms=min(60, args.llm_max_terms),
        llm_timeout=args.llm_timeout,
        domain_hint=(args.domain_hint + " negative target strict").strip(),
    )
    strict = strict_negative_post_filter(negatives, expanded, max_terms=24)
    return strict, {"mode": f"strict_{info.get('mode', 'unknown')}", "error": info.get("error")}


def build_detection_prompt_subset(expanded_prompts, seed_prompts, max_terms=48):
    expanded = dedupe_keep_order(expanded_prompts)
    seeds = dedupe_keep_order(seed_prompts)
    if not expanded:
        return seeds

    anchors = set()
    has_suv_seed = False
    has_sedan_seed = False
    has_pizza_seed = False
    for s in seeds:
        sn = normalize_text(s)
        if "suv" in sn or "越野车" in sn or "suv车" in sn:
            has_suv_seed = True
        if "sedan" in sn or "轿车" in sn or "三厢车" in sn:
            has_sedan_seed = True
        if "pizza" in sn or "披萨" in sn or "比萨" in sn:
            has_pizza_seed = True
        for tk in tokenize_for_match(s):
            if len(tk) >= 2:
                anchors.add(tk)

    selected = []
    selected.extend(seeds)

    noisy_tokens = {
        "posture", "scene", "shadow", "lighting", "appearance", "morphology", "design",
        "grip", "holding", "wielding", "gesture", "握法", "持握", "姿态"
    }
    part_tokens = {
        "head", "body", "tail", "wing", "beak", "blade", "handle", "hilt",
        "头部", "身体", "尾巴", "翅膀", "喙", "刀身", "刀把", "刀柄",
    }
    whole_tokens = {"whole", "full", "entire", "overall", "完整", "整体", "全身"}
    hard_block_terms = {
        "smooth roofline",
        "long wheelbase look",
        "lower body",
        "low stance",
        "longer trunk section",
        "smaller wheel arches",
        "平顺车顶线条",
        "较低车身",
        "较长后备箱段",
        "小轮拱",
    }
    pizza_appliance_terms = {
        "oven", "microwave", "toaster", "stove", "kitchen oven",
        "烤箱", "微波炉", "烤炉", "灶台",
    }

    entity_tokens = set()
    for s in seeds:
        s_norm = normalize_text(s)
        entity_tokens.update(tk for tk in tokenize_for_match(s_norm) if len(tk) >= 2)
        for group_key, vocab in GROUP_SYNONYMS.items():
            if s_norm in {normalize_text(x) for x in vocab}:
                for alias in vocab:
                    entity_tokens.update(tk for tk in tokenize_for_match(alias) if len(tk) >= 2)

    def is_noisy_term(term):
        t = normalize_text(term)
        if t in {"wheel design", "ground clearance shadow", "city road posture", "匕首握法", "dagger grip"}:
            return True
        if has_pizza_seed and any(x in t for x in pizza_appliance_terms):
            return True
        if t in hard_block_terms:
            return True
        tks = set(tokenize_for_match(term))
        return len(tks & noisy_tokens) > 0

    def has_entity_token(term):
        t = normalize_text(term)
        return any(tok in t for tok in entity_tokens)

    def is_part_only_term(term):
        tks = set(tokenize_for_match(term))
        if not (tks & part_tokens):
            return False
        if tks & whole_tokens:
            return False
        return True

    # 先保留短描述词，避免长句影响开放词表匹配
    for term in expanded:
        t = normalize_text(term)
        tks = tokenize_for_match(term)
        if is_noisy_term(term):
            continue
        if is_part_only_term(term):
            continue
        if len(tks) > 6:
            continue
        if has_suv_seed and term in SUV_FEATURE_TERMS:
            selected.append(term)
            continue
        if has_sedan_seed and term in SEDAN_FEATURE_TERMS:
            selected.append(term)
            continue
        if anchors and not any(a in t for a in anchors):
            continue
        if not has_entity_token(term):
            continue
        selected.append(term)

    # 再补充其余锚点相关词
    for term in expanded:
        t = normalize_text(term)
        if is_noisy_term(term):
            continue
        if is_part_only_term(term):
            continue
        if anchors and any(a in t for a in anchors) and has_entity_token(term):
            selected.append(term)

    selected = dedupe_keep_order(selected)

    selected = dedupe_keep_order(selected)
    return selected[:max_terms] if selected else seeds


def prune_overlapping_prompts_for_multi(seed_prompts, expanded_prompts, max_terms=300):
    """多目标模式下，过滤重叠/近重复提示词，避免目标语义互相覆盖。"""
    seeds = dedupe_keep_order(seed_prompts)
    expanded = dedupe_keep_order(expanded_prompts)

    kept = []
    kept_norm = []

    color_alias = {
        "red": "red",
        "green": "green",
        "blue": "blue",
        "black": "black",
        "white": "white",
        "silver": "silver",
        "gray": "gray",
        "grey": "gray",
        "yellow": "yellow",
        "orange": "orange",
        "purple": "purple",
        "pink": "pink",
        "brown": "brown",
        "gold": "gold",
        "golden": "gold",
        "铜": "copper",
        "copper": "copper",
        "红": "red",
        "绿色": "green",
        "绿": "green",
        "蓝": "blue",
        "黑": "black",
        "白": "white",
        "银": "silver",
        "灰": "gray",
        "黄": "yellow",
        "橙": "orange",
        "紫": "purple",
        "粉": "pink",
        "棕": "brown",
        "金": "gold",
    }
    non_anchor_tokens = {
        "with", "on", "in", "at", "of", "for", "to", "from", "by", "and", "or",
        "covered", "obscured", "mirrored", "blueprint", "texture", "wide", "angle", "view", "shape", "struct",
        "painted", "tiled", "roof",
        "with", "on", "覆盖", "遮挡", "镜像", "纹理", "结构", "视角", "屋顶", "涂层", "彩绘", "瓷砖",
    }

    def colors_of(text):
        out = set()
        for tk in tokenize_for_match(text):
            c = color_alias.get(normalize_text(tk))
            if c:
                out.add(c)
        return out

    def anchor_tokens_of(text):
        tks = [normalize_text(x) for x in tokenize_for_match(text)]
        out = []
        for tk in tks:
            if len(tk) < 3:
                continue
            if tk in non_anchor_tokens:
                continue
            if tk in color_alias:
                continue
            out.append(tk)
        return out

    # 基于种子词建立“锚点 -> 允许颜色”约束，防止多目标扩词出现颜色互斥重叠。
    anchor_allowed_colors = {}
    for s in seeds:
        s_colors = colors_of(s)
        if not s_colors:
            continue
        for a in anchor_tokens_of(s):
            anchor_allowed_colors.setdefault(a, set()).update(s_colors)

    def canon(text):
        t = normalize_text(text)
        if t.endswith("s") and len(t) > 4:
            t = t[:-1]
        return t

    def overlapped(n):
        for ex in kept_norm:
            if n == ex:
                return True
            # 避免 broad/narrow 互相重叠，例如 laptop 与 gaming laptop。
            if len(n) >= 5 and len(ex) >= 5 and (n in ex or ex in n):
                return True
        return False

    def violates_color_constraint(term):
        t_colors = colors_of(term)
        if not t_colors:
            return False
        for a in anchor_tokens_of(term):
            allow = anchor_allowed_colors.get(a)
            if not allow:
                continue
            # 该锚点存在颜色白名单时，发现完全不在白名单内的颜色则丢弃。
            if t_colors.isdisjoint(allow):
                return True
        return False

    for term in seeds:
        n = canon(term)
        if not n or overlapped(n):
            continue
        kept.append(term)
        kept_norm.append(n)

    for term in expanded:
        n = canon(term)
        if not n or overlapped(n):
            continue
        if violates_color_constraint(term):
            continue
        kept.append(term)
        kept_norm.append(n)
        if len(kept) >= max(1, int(max_terms)):
            break

    return kept


def build_prompt_groups_for_display(seed_prompts, expanded_prompts, max_each=24):
    seeds = dedupe_keep_order(seed_prompts)
    expanded = dedupe_keep_order(expanded_prompts)
    groups = {}

    normalized_synonyms = {
        k: {normalize_text(x) for x in v}
        for k, v in GROUP_SYNONYMS.items()
    }

    def to_group_key(text):
        t = normalize_text(text)
        for key, vocab in normalized_synonyms.items():
            if t in vocab:
                return key
        return t

    def related(seed, term):
        s_norm = normalize_text(seed)
        t_norm = normalize_text(term)
        if s_norm == t_norm:
            return True
        if s_norm and (s_norm in t_norm or t_norm in s_norm):
            return True
        if to_group_key(seed) == to_group_key(term):
            return True
        s_tokens = set(tokenize_for_match(seed))
        t_tokens = set(tokenize_for_match(term))
        return len(s_tokens & t_tokens) > 0

    for seed in seeds:
        s_norm = normalize_text(seed)
        anchors = set(tokenize_for_match(seed))
        has_suv_seed = ("suv" in s_norm) or ("越野车" in s_norm) or ("suv车" in s_norm)
        has_sedan_seed = ("sedan" in s_norm) or ("轿车" in s_norm) or ("三厢车" in s_norm)
        row = []
        for term in expanded:
            t_norm = normalize_text(term)
            if t_norm == s_norm:
                continue
            if has_suv_seed and term in SUV_FEATURE_TERMS:
                row.append(term)
                continue
            if has_sedan_seed and term in SEDAN_FEATURE_TERMS:
                row.append(term)
                continue
            if related(seed, term):
                row.append(term)
                continue
            if anchors and any(a in t_norm for a in anchors):
                row.append(term)
        row = dedupe_keep_order(row)
        if row:
            groups[seed] = row[:max_each]

    if not groups and expanded:
        groups[seeds[0] if seeds else "group"] = dedupe_keep_order(expanded)[:max_each]
    return groups


def collect_images(images_dir):
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def extract_name_list(names_obj):
    if isinstance(names_obj, dict):
        return [str(names_obj[k]) for k in sorted(names_obj.keys())]
    return [str(x) for x in names_obj]


def select_builtin_classes(builtin_names, prompts):
    if not prompts:
        return builtin_names

    builtin_norm = {normalize_text(n): n for n in builtin_names}
    selected = []

    for prompt in prompts:
        p = normalize_text(prompt)
        if p in builtin_norm:
            selected.append(builtin_norm[p])
            continue

        contains = [
            raw
            for raw in builtin_names
            if (p in normalize_text(raw)) or (normalize_text(raw) in p)
        ]
        if contains:
            selected.extend(contains)

    if not selected:
        # 精准回退，避免无匹配时退回“全部类别”带来的明显误检。
        fallback_map = {
            "parrot": ["bird"],
            "鹦鹉": ["bird"],
            "green apple": ["apple"],
            "青苹果": ["apple"],
            "sedan": ["car"],
            "suv": ["car"],
            "dagger": ["knife"],
            "短剑": ["knife"],
            "匕首": ["knife"],
        }
        for prompt in prompts:
            p = normalize_text(prompt)
            for key, vals in fallback_map.items():
                if key in p:
                    for v in vals:
                        vn = normalize_text(v)
                        if vn in builtin_norm:
                            selected.append(builtin_norm[vn])

    selected = dedupe_keep_order(selected)
    return selected if selected else builtin_names


def _is_line_like_polygon(poly_xyn):
    arr = np.asarray(poly_xyn, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
        return True
    poly_area = _polygon_area_ratio_xyn(arr)
    box = _polygon_bbox_xywhn(arr)
    if box is None:
        return True
    bw = float(box[2])
    bh = float(box[3])
    box_area = max(1e-8, bw * bh)
    fill_ratio = poly_area / box_area
    aspect = max(bw, bh) / max(1e-8, min(bw, bh))
    if fill_ratio < 0.10 and aspect > 6.0:
        return True
    if poly_area < 0.00015 and aspect > 10.0:
        return True
    return False


def _mask_bbox_density(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    area = int(len(xs))
    if area <= 0:
        return 0.0
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    return float(area) / float(bw * bh)


def _keep_anchor_component_without_thin_bridges(mask_u8, anchor_mask_u8, thin_diameter=3.0):
    src = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
    if int(np.count_nonzero(src)) <= 0:
        return src

    anchor = np.where(anchor_mask_u8 > 0, 255, 0).astype(np.uint8)
    if int(np.count_nonzero(cv2.bitwise_and(src, anchor))) <= 0:
        return src

    fg_bin = np.where(src > 0, 1, 0).astype(np.uint8)
    dist = cv2.distanceTransform(fg_bin, cv2.DIST_L2, 3)
    thin = np.where((src > 0) & ((dist * 2.0) <= float(thin_diameter)), 255, 0).astype(np.uint8)
    pruned = cv2.bitwise_and(src, cv2.bitwise_not(thin))
    if int(np.count_nonzero(pruned)) <= 0:
        return src

    n, labels, stats, _ = cv2.connectedComponentsWithStats(pruned, connectivity=8)
    if n <= 1:
        return src

    # 细桥切断后，仅保留最贴近锚区的主连通域，避免“脖子细线牵到手”。
    best_id = -1
    best_score = -1.0
    anchor_dilate = cv2.dilate(anchor, np.ones((3, 3), np.uint8), iterations=1)
    for lid in range(1, n):
        comp = np.where(labels == lid, 255, 0).astype(np.uint8)
        area = float(stats[lid, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        overlap = float(np.count_nonzero(cv2.bitwise_and(comp, anchor_dilate)))
        score = overlap * 3.0 + area
        if score > best_score:
            best_score = score
            best_id = lid

    if best_id <= 0:
        return src

    keep = np.where(labels == best_id, 255, 0).astype(np.uint8)
    grow = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=2)
    out = cv2.bitwise_and(src, grow)
    if int(np.count_nonzero(cv2.bitwise_and(out, anchor))) <= 0:
        return src
    return out


def _drop_overdense_outer_components(mask_u8, anchor_mask_u8, density_margin=0.06, min_outer_area=24):
    src = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
    if int(np.count_nonzero(src)) <= 0:
        return src

    anchor = np.where(anchor_mask_u8 > 0, 255, 0).astype(np.uint8)
    anchor_part = cv2.bitwise_and(src, anchor)
    anchor_density = _mask_bbox_density(anchor_part)
    if anchor_density <= 0.0:
        return src

    outer = cv2.bitwise_and(src, cv2.bitwise_not(anchor))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(outer, connectivity=8)
    if n <= 1:
        return src

    out = src.copy()
    for lid in range(1, n):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < int(min_outer_area):
            continue
        comp = np.where(labels == lid, 255, 0).astype(np.uint8)
        dens = _mask_bbox_density(comp)
        if dens > float(anchor_density + density_margin):
            out[labels == lid] = 0

    if int(np.count_nonzero(out)) <= 0:
        return src
    return out


def ensure_dirs(output_dir, save_vis):
    output_root = Path(output_dir)
    labels_det = output_root / "labels_det"
    labels_seg = output_root / "labels_seg"
    vis_dir = output_root / "vis"

    output_root.mkdir(parents=True, exist_ok=True)
    labels_det.mkdir(parents=True, exist_ok=True)
    labels_seg.mkdir(parents=True, exist_ok=True)
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    return output_root, labels_det, labels_seg, vis_dir


def write_classes_file(output_root, classes):
    with open(output_root / "classes.txt", "w", encoding="utf-8") as f:
        for cls in classes:
            f.write(f"{cls}\n")


def yolo_det_line_norm(cls_id, xywhn):
    xc, yc, bw, bh = xywhn
    xc = max(0.0, min(1.0, float(xc)))
    yc = max(0.0, min(1.0, float(yc)))
    bw = max(0.0, min(1.0, float(bw)))
    bh = max(0.0, min(1.0, float(bh)))
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def _parse_det_line_norm(line):
    try:
        parts = str(line).strip().split()
        if len(parts) != 5:
            return None
        cls_id = int(parts[0])
        xc, yc, bw, bh = [float(v) for v in parts[1:]]
        return cls_id, [xc, yc, bw, bh]
    except Exception:
        return None


def _polygon_area_ratio_xyn(poly_xyn):
    try:
        arr = np.asarray(poly_xyn, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
            return 0.0
        x = np.clip(arr[:, 0], 0.0, 1.0)
        y = np.clip(arr[:, 1], 0.0, 1.0)
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return float(area)
    except Exception:
        return 0.0


def _to_global_xywhn_from_roi(local_xywhn, roi_xyxy, image_w, image_h):
    xc, yc, bw, bh = [float(v) for v in local_xywhn]
    rx1, ry1, rx2, ry2 = [int(v) for v in roi_xyxy]
    rw = max(1.0, float(rx2 - rx1))
    rh = max(1.0, float(ry2 - ry1))
    gx = (float(rx1) + xc * rw) / float(image_w)
    gy = (float(ry1) + yc * rh) / float(image_h)
    gw = (bw * rw) / float(image_w)
    gh = (bh * rh) / float(image_h)
    return [
        max(0.0, min(1.0, gx)),
        max(0.0, min(1.0, gy)),
        max(0.0, min(1.0, gw)),
        max(0.0, min(1.0, gh)),
    ]


def _to_global_poly_xyn_from_roi(local_poly_xyn, roi_xyxy, image_w, image_h):
    rx1, ry1, rx2, ry2 = [int(v) for v in roi_xyxy]
    rw = max(1.0, float(rx2 - rx1))
    rh = max(1.0, float(ry2 - ry1))
    out = []
    for x, y in np.asarray(local_poly_xyn, dtype=np.float32):
        gx = (float(rx1) + float(x) * rw) / float(image_w)
        gy = (float(ry1) + float(y) * rh) / float(image_h)
        out.append((max(0.0, min(1.0, gx)), max(0.0, min(1.0, gy))))
    return out


def _append_det_line_if_new(det_lines, cls_id, xywhn, iou_thr=0.85):
    candidate = tuple(float(v) for v in xywhn)
    cxy = _xywhn_to_xyxy(candidate, 10000, 10000)
    for line in det_lines:
        parsed = _parse_det_line_norm(line)
        if not parsed:
            continue
        cid, ex = parsed
        if int(cid) != int(cls_id):
            continue
        exxy = _xywhn_to_xyxy(ex, 10000, 10000)
        if _iou_xyxy(cxy, exxy) >= float(iou_thr):
            return False
    det_lines.append(yolo_det_line_norm(cls_id, candidate))
    return True


def _polygon_bbox_xywhn(poly_xyn):
    try:
        arr = np.asarray(poly_xyn, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
            return None
        x = np.clip(arr[:, 0], 0.0, 1.0)
        y = np.clip(arr[:, 1], 0.0, 1.0)
        x1, x2 = float(np.min(x)), float(np.max(x))
        y1, y2 = float(np.min(y)), float(np.max(y))
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 1e-6 or bh <= 1e-6:
            return None
        xc = (x1 + x2) * 0.5
        yc = (y1 + y2) * 0.5
        return [xc, yc, bw, bh]
    except Exception:
        return None


def _union_xywhn(a, b):
    axc, ayc, abw, abh = [float(v) for v in a]
    bxc, byc, bbw, bbh = [float(v) for v in b]

    ax1, ay1 = axc - abw * 0.5, ayc - abh * 0.5
    ax2, ay2 = axc + abw * 0.5, ayc + abh * 0.5
    bx1, by1 = bxc - bbw * 0.5, byc - bbh * 0.5
    bx2, by2 = bxc + bbw * 0.5, byc + bbh * 0.5

    x1 = max(0.0, min(1.0, min(ax1, bx1)))
    y1 = max(0.0, min(1.0, min(ay1, by1)))
    x2 = max(0.0, min(1.0, max(ax2, bx2)))
    y2 = max(0.0, min(1.0, max(ay2, by2)))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = (x1 + x2) * 0.5
    yc = (y1 + y2) * 0.5
    return [xc, yc, bw, bh]


def _should_union_mask_box(det_xywhn, mask_xywhn):
    dxc, dyc, dbw, dbh = [float(v) for v in det_xywhn]
    mxc, myc, mbw, mbh = [float(v) for v in mask_xywhn]

    d_area = max(1e-8, dbw * dbh)
    m_area = max(1e-8, mbw * mbh)

    d_box = (dxc - dbw * 0.5, dyc - dbh * 0.5, dxc + dbw * 0.5, dyc + dbh * 0.5)
    m_box = (mxc - mbw * 0.5, myc - mbh * 0.5, mxc + mbw * 0.5, myc + mbh * 0.5)

    iou = _iou_xyxy(d_box, m_box)
    area_ratio = m_area / d_area
    side_expand_x = max(0.0, (mbw - dbw) / max(1e-8, dbw))
    side_expand_y = max(0.0, (mbh - dbh) / max(1e-8, dbh))
    center_dx = abs(mxc - dxc) / max(1e-8, dbw)
    center_dy = abs(myc - dyc) / max(1e-8, dbh)

    # 防止细线把远处无关物体串联进来导致外接框暴涨。
    if area_ratio > 1.6:
        return False
    if iou < 0.35:
        return False
    if side_expand_x > 0.6 or side_expand_y > 0.6:
        return False
    if center_dx > 0.45 or center_dy > 0.45:
        return False
    return True


def _xywhn_to_xyxy_norm(xywhn):
    xc, yc, bw, bh = [float(v) for v in xywhn]
    x1 = xc - bw * 0.5
    y1 = yc - bh * 0.5
    x2 = xc + bw * 0.5
    y2 = yc + bh * 0.5
    return (
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
        max(0.0, min(1.0, x2)),
        max(0.0, min(1.0, y2)),
    )


def _collect_secondary_support_boxes(secondary_result, class_id_mode, class_name_to_id, min_conf):
    support = {}
    if secondary_result is None or secondary_result.boxes is None or len(secondary_result.boxes) <= 0:
        return support

    names = secondary_result.names if isinstance(secondary_result.names, dict) else {
        i: n for i, n in enumerate(secondary_result.names)
    }
    xywhn = secondary_result.boxes.xywhn.cpu().numpy()
    cls_ids = secondary_result.boxes.cls.cpu().numpy().astype(int)
    confs = secondary_result.boxes.conf.cpu().numpy()

    for i, box in enumerate(xywhn):
        if float(confs[i]) < float(min_conf):
            continue
        model_cls = int(cls_ids[i])
        if class_id_mode == "index" and model_cls in class_name_to_id:
            cls_id = class_name_to_id[model_cls]
        else:
            name = str(names.get(model_cls, model_cls))
            cls_id = class_name_to_id.get(normalize_text(name), None)
        if cls_id is None:
            continue
        support.setdefault(int(cls_id), []).append([float(v) for v in box])
    return support


def _has_secondary_support(candidate_xywhn, cls_id, support_boxes):
    boxes = support_boxes.get(int(cls_id), [])
    if not boxes:
        return False
    cand = _xywhn_to_xyxy_norm(candidate_xywhn)
    best_iou = 0.0
    for b in boxes:
        biou = _iou_xyxy(cand, _xywhn_to_xyxy_norm(b))
        if biou > best_iou:
            best_iou = biou
    return best_iou >= 0.18


def _is_bird_like_name(name_text):
    t = normalize_text(name_text)
    return ("parrot" in t) or ("bird" in t) or ("鹦鹉" in t) or ("鸟" in t)


def _class_color(cls_id):
    cid = int(cls_id)
    vivid_palette = [
        (0, 255, 255),   # bright yellow
        (0, 140, 255),   # orange
        (255, 255, 0),   # cyan
        (255, 80, 255),  # magenta
        (0, 255, 0),     # green
        (0, 0, 255),     # red
    ]
    return vivid_palette[cid % len(vivid_palette)]


def _pick_contrast_color(local_bgr, seed=0):
    palette = [
        (0, 255, 255),
        (0, 140, 255),
        (255, 255, 0),
        (255, 80, 255),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
        (64, 64, 64),
    ]
    if local_bgr is None:
        return palette[int(seed) % len(palette)]

    local_patch = np.asarray(local_bgr, dtype=np.uint8).reshape(1, 1, 3)
    local_lab = cv2.cvtColor(local_patch, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)

    best = None
    best_score = -1.0
    for i, color in enumerate(palette):
        c_patch = np.asarray(color, dtype=np.uint8).reshape(1, 1, 3)
        c_lab = cv2.cvtColor(c_patch, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
        dist = float(np.linalg.norm(c_lab - local_lab))
        tie = 0.01 * ((i + int(seed)) % len(palette))
        score = dist + tie
        if score > best_score:
            best_score = score
            best = color
    return best


def _draw_pretty_box(canvas, x1, y1, x2, y2, color, label):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    edge = max(10, int(min(w, h) * 0.2))

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # 角标样式，视觉更干净且更易定位。
    cv2.line(canvas, (x1, y1), (x1 + edge, y1), color, 3, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1), (x1, y1 + edge), color, 3, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2 - edge, y1), color, 3, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2, y1 + edge), color, 3, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1 + edge, y2), color, 3, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1, y2 - edge), color, 3, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2 - edge, y2), color, 3, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2, y2 - edge), color, 3, cv2.LINE_AA)

    text = str(label)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    pad = 5
    bx1 = max(0, x1)
    by2 = max(th + 2 * pad + 2, y1)
    by1 = max(0, by2 - (th + 2 * pad + 2))
    bx2 = min(canvas.shape[1] - 1, bx1 + tw + 2 * pad + 2)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, -1, cv2.LINE_AA)

    text_color = (20, 20, 20)
    luma = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    if luma < 140:
        text_color = (255, 255, 255)
    cv2.putText(canvas, text, (bx1 + pad, by2 - pad - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2, cv2.LINE_AA)


def render_preview_from_filtered_labels(image, det_lines, seg_items):
    canvas = image.copy()
    h, w = canvas.shape[:2]
    seg_color_by_cls = {}

    if seg_items:
        overlay = canvas.copy()
        for item in seg_items:
            cls_id = int(item.get("cls_id", 0))
            poly = np.asarray(item.get("poly", []), dtype=np.float32)
            if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
                continue
            px = np.clip(poly[:, 0] * float(w), 0, float(w - 1)).astype(np.int32)
            py = np.clip(poly[:, 1] * float(h), 0, float(h - 1)).astype(np.int32)
            pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [pts], 255)
            mean_bgr = cv2.mean(image, mask=poly_mask)[:3]
            color = _pick_contrast_color(mean_bgr, seed=cls_id)
            seg_color_by_cls[cls_id] = color
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, color, 2)
        canvas = cv2.addWeighted(overlay, 0.52, canvas, 0.48, 0.0)

    for line in det_lines:
        parsed = _parse_det_line_norm(line)
        if not parsed:
            continue
        cls_id, xywhn = parsed
        x1, y1, x2, y2 = _xywhn_to_xyxy(xywhn, w, h)
        if cls_id in seg_color_by_cls:
            color = seg_color_by_cls[cls_id]
        else:
            x1c = max(0, min(w - 1, x1))
            y1c = max(0, min(h - 1, y1))
            x2c = max(x1c + 1, min(w, x2))
            y2c = max(y1c + 1, min(h, y2))
            local = image[y1c:y2c, x1c:x2c]
            local_mean = tuple(float(v) for v in cv2.mean(local)[:3]) if local.size > 0 else None
            color = _pick_contrast_color(local_mean, seed=cls_id)
        _draw_pretty_box(canvas, x1, y1, x2, y2, color, cls_id)
    return canvas


def expand_xywhn(xywhn, scale=1.15):
    xc, yc, bw, bh = [float(v) for v in xywhn]
    bw *= float(scale)
    bh *= float(scale)
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return [xc, yc, bw, bh]


def _xywhn_to_xyxy(xywhn, width, height):
    xc, yc, bw, bh = [float(v) for v in xywhn]
    x1 = int(round((xc - bw / 2.0) * width))
    y1 = int(round((yc - bh / 2.0) * height))
    x2 = int(round((xc + bw / 2.0) * width))
    y2 = int(round((yc + bh / 2.0) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _xyxy_to_xywhn(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    bw = float(x2 - x1) / float(width)
    bh = float(y2 - y1) / float(height)
    xc = float(x1 + x2) / 2.0 / float(width)
    yc = float(y1 + y2) / 2.0 / float(height)
    return [xc, yc, bw, bh]


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _build_attention_hsv_mask(hsv_roi):
    mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    for ranges in ATTENTION_PART_HSV_TABLE.values():
        for item in ranges:
            lower = np.array(item["lower"], dtype=np.uint8)
            upper = np.array(item["upper"], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_roi, lower, upper))
    return mask


def refine_box_with_pixel_correlation(
    image,
    xywhn,
    max_expand=1.25,
    min_iou=0.55,
    max_area_ratio=1.8,
):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _xywhn_to_xyxy(xywhn, w, h)
    bw = max(2, x2 - x1)
    bh = max(2, y2 - y1)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    ex = int(round(bw * (max_expand - 1.0) / 2.0))
    ey = int(round(bh * (max_expand - 1.0) / 2.0))
    rx1 = max(0, x1 - ex)
    ry1 = max(0, y1 - ey)
    rx2 = min(w, x2 + ex)
    ry2 = min(h, y2 + ey)
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return xywhn

    roi = image[ry1:ry2, rx1:rx2].copy()
    rect = (x1 - rx1, y1 - ry1, max(1, x2 - x1), max(1, y2 - y1))
    mask = cv2.GC_PR_BGD * np.ones(roi.shape[:2], dtype=np.uint8)

    # 使用原框中心区域作为高置信前景锚点，避免像素扩展脱离目标主体。
    core_ratio = 0.45
    core_w = max(2, int(round((x2 - x1) * core_ratio)))
    core_h = max(2, int(round((y2 - y1) * core_ratio)))
    core_x1 = max(rx1, cx - core_w // 2)
    core_y1 = max(ry1, cy - core_h // 2)
    core_x2 = min(rx2, core_x1 + core_w)
    core_y2 = min(ry2, core_y1 + core_h)

    if core_x2 > core_x1 and core_y2 > core_y1:
        mask[core_y1 - ry1 : core_y2 - ry1, core_x1 - rx1 : core_x2 - rx1] = cv2.GC_FGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(roi, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return xywhn

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if fg_mask.sum() <= 0:
        return xywhn

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    core_local_x1 = max(0, core_x1 - rx1)
    core_local_y1 = max(0, core_y1 - ry1)
    core_local_x2 = min(fg_mask.shape[1], core_x2 - rx1)
    core_local_y2 = min(fg_mask.shape[0], core_y2 - ry1)
    core_mask = np.zeros(fg_mask.shape, dtype=np.uint8)
    if core_local_x2 > core_local_x1 and core_local_y2 > core_local_y1:
        core_mask[core_local_y1:core_local_y2, core_local_x1:core_local_x2] = 255

    # 使用 LAB 色彩距离约束“同区域像素延展”，减少跨材质误扩展。
    if core_mask.any():
        core_pixels = lab_roi[core_mask > 0].astype(np.float32)
        if core_pixels.size > 0:
            core_mean = np.mean(core_pixels, axis=0)
            core_dist = np.linalg.norm(core_pixels - core_mean, axis=1)
            core_sigma = float(np.percentile(core_dist, 90)) if core_dist.size > 0 else 0.0
            lab_thr = float(np.clip(core_sigma + 18.0, 10.0, 42.0))
            lab_dist_map = np.linalg.norm(lab_roi.astype(np.float32) - core_mean.reshape(1, 1, 3), axis=2)
            lab_mask = (lab_dist_map <= lab_thr).astype(np.uint8) * 255
            fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_or(lab_mask, core_mask))

    # HSV 注意部位先验作为弱约束：有命中时优先保留命中区域及核心区。
    hsv_attention = _build_attention_hsv_mask(hsv_roi)
    if int(np.count_nonzero(hsv_attention)) > 0:
        keep_mask = cv2.bitwise_or(hsv_attention, core_mask)
        fg_mask = cv2.bitwise_and(fg_mask, keep_mask)

    if fg_mask.sum() <= 0:
        return xywhn

    # 严格模式：先裁掉离核心过远的前景，避免细线把远处物体牵进来。
    core_inv = np.where(core_mask > 0, 0, 255).astype(np.uint8)
    core_dist_map = cv2.distanceTransform(core_inv, cv2.DIST_L2, 3)
    max_core_dist = max(6.0, 0.50 * float(max(bw, bh)))
    fg_mask = np.where((fg_mask > 0) & (core_dist_map <= max_core_dist), 255, 0).astype(np.uint8)

    if fg_mask.sum() <= 0:
        return xywhn

    # 抑制单线桥接：非核心附近必须具有足够厚度。
    fg_bin = np.where(fg_mask > 0, 1, 0).astype(np.uint8)
    fg_dist_map = cv2.distanceTransform(fg_bin, cv2.DIST_L2, 3)
    thick_keep = np.where((fg_dist_map * 2.0) >= 3.2, 255, 0).astype(np.uint8)
    near_core_keep = np.where(core_dist_map <= max(3.0, 0.15 * float(max(bw, bh))), 255, 0).astype(np.uint8)
    fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_or(thick_keep, near_core_keep))

    if fg_mask.sum() <= 0:
        return xywhn

    anchor_box_mask = np.zeros(fg_mask.shape, dtype=np.uint8)
    ox1 = max(0, min(fg_mask.shape[1], x1 - rx1))
    oy1 = max(0, min(fg_mask.shape[0], y1 - ry1))
    ox2 = max(0, min(fg_mask.shape[1], x2 - rx1))
    oy2 = max(0, min(fg_mask.shape[0], y2 - ry1))
    if ox2 > ox1 and oy2 > oy1:
        anchor_box_mask[oy1:oy2, ox1:ox2] = 255

    fg_mask = _keep_anchor_component_without_thin_bridges(fg_mask, anchor_box_mask, thin_diameter=3.2)
    fg_mask = _drop_overdense_outer_components(fg_mask, anchor_box_mask, density_margin=0.06, min_outer_area=20)

    if fg_mask.sum() <= 0:
        return xywhn

    # 先开运算去细线桥接，再闭运算补小洞，减少“牵线串到无关物体”。
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    if num_labels <= 1:
        return xywhn

    best_id = -1
    best_score = -1
    seed_x = max(0, min(labels.shape[1] - 1, cx - rx1))
    seed_y = max(0, min(labels.shape[0] - 1, cy - ry1))
    center_label = int(labels[seed_y, seed_x])

    anchor_labels = set()
    if core_local_x2 > core_local_x1 and core_local_y2 > core_local_y1:
        anchor_labels = set(np.unique(labels[core_local_y1:core_local_y2, core_local_x1:core_local_x2]).tolist())
        anchor_labels.discard(0)
    if center_label > 0:
        anchor_labels.add(center_label)

    for lid in range(1, num_labels):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        if anchor_labels and lid not in anchor_labels:
            continue

        score = area
        if lid == center_label:
            score += area
        if score > best_score:
            best_score = score
            best_id = lid

    if best_id <= 0:
        return xywhn

    lx = int(stats[best_id, cv2.CC_STAT_LEFT])
    ly = int(stats[best_id, cv2.CC_STAT_TOP])
    lw = int(stats[best_id, cv2.CC_STAT_WIDTH])
    lh = int(stats[best_id, cv2.CC_STAT_HEIGHT])

    comp_area = int(stats[best_id, cv2.CC_STAT_AREA])
    comp_bbox_area = max(1, lw * lh)
    fill_ratio = float(comp_area) / float(comp_bbox_area)
    aspect = float(max(lw, lh)) / float(max(1, min(lw, lh)))

    comp_mask = (labels == best_id).astype(np.uint8) * 255
    comp_mask = _keep_anchor_component_without_thin_bridges(comp_mask, anchor_box_mask, thin_diameter=3.2)
    comp_mask = _drop_overdense_outer_components(comp_mask, anchor_box_mask, density_margin=0.06, min_outer_area=20)
    if int(np.count_nonzero(comp_mask)) <= 0:
        return xywhn

    ys, xs = np.where(comp_mask > 0)
    lx = int(xs.min())
    ly = int(ys.min())
    lw = int(xs.max() - xs.min() + 1)
    lh = int(ys.max() - ys.min() + 1)

    comp_area = int(np.count_nonzero(comp_mask))
    comp_bbox_area = max(1, lw * lh)
    fill_ratio = float(comp_area) / float(comp_bbox_area)
    aspect = float(max(lw, lh)) / float(max(1, min(lw, lh)))

    opened = cv2.morphologyEx(comp_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    opened_keep_ratio = float(np.count_nonzero(opened)) / float(max(1, np.count_nonzero(comp_mask)))

    ox1 = max(0, min(comp_mask.shape[1], x1 - rx1))
    oy1 = max(0, min(comp_mask.shape[0], y1 - ry1))
    ox2 = max(0, min(comp_mask.shape[1], x2 - rx1))
    oy2 = max(0, min(comp_mask.shape[0], y2 - ry1))
    anchor_overlap = 0.0
    if ox2 > ox1 and oy2 > oy1:
        anchor_overlap = float(np.count_nonzero(comp_mask[oy1:oy2, ox1:ox2])) / float(max(1, (ox2 - ox1) * (oy2 - oy1)))

    nx1 = rx1 + lx
    ny1 = ry1 + ly
    nx2 = nx1 + lw
    ny2 = ny1 + lh

    # 框修正只允许外扩，不允许收缩，避免出现“框不全”。
    nx1 = min(nx1, x1)
    ny1 = min(ny1, y1)
    nx2 = max(nx2, x2)
    ny2 = max(ny2, y2)

    old_box = (x1, y1, x2, y2)
    new_box = (nx1, ny1, nx2, ny2)
    iou = _iou_xyxy(old_box, new_box)
    old_area = max(1, (x2 - x1) * (y2 - y1))
    new_area = max(1, (nx2 - nx1) * (ny2 - ny1))
    side_limit_x = int(round((max_expand - 1.0) * bw / 2.0))
    side_limit_y = int(round((max_expand - 1.0) * bh / 2.0))

    old_cx = 0.5 * (x1 + x2)
    old_cy = 0.5 * (y1 + y2)
    new_cx = 0.5 * (nx1 + nx2)
    new_cy = 0.5 * (ny1 + ny2)
    drift_x = abs(new_cx - old_cx) / float(max(1, bw))
    drift_y = abs(new_cy - old_cy) / float(max(1, bh))

    if iou < float(min_iou):
        return xywhn

    # 严格禁止单线桥接式扩张和远离高置信锚点的漂移扩张。
    if fill_ratio < 0.30 and aspect > 2.6:
        return xywhn
    if opened_keep_ratio < 0.65:
        return xywhn
    if anchor_overlap < 0.18:
        return xywhn
    if drift_x > 0.28 or drift_y > 0.28:
        return xywhn

    # 扩张颜色门控：若新增区域与高置信主题颜色差异过大，则放弃扩张。
    if core_mask.any():
        ox1, oy1, ox2, oy2 = x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1
        nx1r, ny1r, nx2r, ny2r = nx1 - rx1, ny1 - ry1, nx2 - rx1, ny2 - ry1
        ox1 = max(0, min(fg_mask.shape[1], int(ox1)))
        oy1 = max(0, min(fg_mask.shape[0], int(oy1)))
        ox2 = max(0, min(fg_mask.shape[1], int(ox2)))
        oy2 = max(0, min(fg_mask.shape[0], int(oy2)))
        nx1r = max(0, min(fg_mask.shape[1], int(nx1r)))
        ny1r = max(0, min(fg_mask.shape[0], int(ny1r)))
        nx2r = max(0, min(fg_mask.shape[1], int(nx2r)))
        ny2r = max(0, min(fg_mask.shape[0], int(ny2r)))

        ring_mask = np.zeros(fg_mask.shape, dtype=np.uint8)
        if nx2r > nx1r and ny2r > ny1r:
            ring_mask[ny1r:ny2r, nx1r:nx2r] = 255
        if ox2 > ox1 and oy2 > oy1:
            ring_mask[oy1:oy2, ox1:ox2] = 0

        ring_count = int(np.count_nonzero(ring_mask))
        if ring_count > 12:
            core_lab = lab_roi[core_mask > 0].astype(np.float32)
            ring_lab = lab_roi[ring_mask > 0].astype(np.float32)
            core_hsv = hsv_roi[core_mask > 0].astype(np.float32)
            ring_hsv = hsv_roi[ring_mask > 0].astype(np.float32)
            if core_lab.size > 0 and ring_lab.size > 0 and core_hsv.size > 0 and ring_hsv.size > 0:
                core_lab_mean = np.mean(core_lab, axis=0)
                ring_lab_dist = np.linalg.norm(ring_lab - core_lab_mean, axis=1)
                lab_shift = float(np.percentile(ring_lab_dist, 75))

                core_hsv_mean = np.mean(core_hsv, axis=0)
                ring_hsv_mean = np.mean(ring_hsv, axis=0)
                hue_diff = abs(float(ring_hsv_mean[0]) - float(core_hsv_mean[0]))
                hue_diff = min(hue_diff, 180.0 - hue_diff)
                sv_diff = abs(float(ring_hsv_mean[1]) - float(core_hsv_mean[1])) + 0.5 * abs(
                    float(ring_hsv_mean[2]) - float(core_hsv_mean[2])
                )

                att_overlap = 0.0
                if int(np.count_nonzero(hsv_attention)) > 0:
                    att_overlap = float(np.count_nonzero(cv2.bitwise_and(hsv_attention, ring_mask))) / float(ring_count)

                if lab_shift > 34.0 and hue_diff > 24.0 and sv_diff > 90.0 and att_overlap < 0.05:
                    return xywhn

    if float(new_area) / float(old_area) > float(max_area_ratio):
        return xywhn
    if (x1 - nx1) > side_limit_x or (nx2 - x2) > side_limit_x:
        return xywhn
    if (y1 - ny1) > side_limit_y or (ny2 - y2) > side_limit_y:
        return xywhn

    return _xyxy_to_xywhn(nx1, ny1, nx2, ny2, w, h)


def yolo_seg_line_norm(cls_id, polygon_xyn):
    coords = []
    for x, y in polygon_xyn:
        xn = max(0.0, min(1.0, float(x)))
        yn = max(0.0, min(1.0, float(y)))
        coords.append(f"{xn:.6f}")
        coords.append(f"{yn:.6f}")
    if len(coords) < 6:
        return None
    return f"{cls_id} " + " ".join(coords)


def _sanitize_seg_polygon_xyn(poly_xyn, anchor_xywhn, image_shape, gate_expand=1.18):
    h, w = image_shape[:2]
    poly = np.asarray(poly_xyn, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        return None

    px = np.clip(poly[:, 0] * float(w), 0, float(w - 1)).astype(np.int32)
    py = np.clip(poly[:, 1] * float(h), 0, float(h - 1)).astype(np.int32)
    pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))

    seg_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(seg_mask, [pts], 255)
    if int(np.count_nonzero(seg_mask)) <= 0:
        return None

    # 锚框门控：限制分割只在目标主体邻域内，切断远端牵线。
    ax = expand_xywhn(anchor_xywhn, scale=gate_expand)
    x1, y1, x2, y2 = _xywhn_to_xyxy(ax, w, h)
    gate_mask = np.zeros((h, w), dtype=np.uint8)
    gate_mask[y1:y2, x1:x2] = 255
    gated = cv2.bitwise_and(seg_mask, gate_mask)
    if int(np.count_nonzero(gated)) <= 0:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gated, connectivity=8)
    if num_labels <= 1:
        return None

    # 优先保留靠近锚框中心且面积最大的连通域。
    acx = int(round(float(anchor_xywhn[0]) * w))
    acy = int(round(float(anchor_xywhn[1]) * h))
    acx = max(0, min(w - 1, acx))
    acy = max(0, min(h - 1, acy))
    center_label = int(labels[acy, acx])

    best_id = -1
    best_score = -1.0
    for lid in range(1, num_labels):
        area = float(stats[lid, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        score = area
        if lid == center_label:
            score += area
        if score > best_score:
            best_score = score
            best_id = lid

    if best_id <= 0:
        return None

    keep = np.where(labels == best_id, 255, 0).astype(np.uint8)

    # 主体锚区使用原始锚框（不扩张），外扩区域聚集度不能高于主体。
    ax0, ay0, ax1, ay1 = _xywhn_to_xyxy(anchor_xywhn, w, h)
    anchor_mask = np.zeros((h, w), dtype=np.uint8)
    anchor_mask[ay0:ay1, ax0:ax1] = 255

    keep = _keep_anchor_component_without_thin_bridges(keep, anchor_mask, thin_diameter=3.0)
    keep = _drop_overdense_outer_components(keep, anchor_mask, density_margin=0.06, min_outer_area=16)
    if int(np.count_nonzero(keep)) <= 0:
        return None

    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 8:
        return None

    eps = max(1.0, 0.002 * cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, eps, True)
    if approx is None or len(approx) < 3:
        approx = cnt

    out = []
    for p in approx.reshape(-1, 2):
        xn = max(0.0, min(1.0, float(p[0]) / float(w)))
        yn = max(0.0, min(1.0, float(p[1]) / float(h)))
        out.append((xn, yn))
    if len(out) < 3:
        return None
    return out


def write_label_files(stem, det_lines, seg_lines, labels_det, labels_seg, label_output_mode="both"):
    mode = str(label_output_mode or "both").strip().lower()
    det_file = labels_det / f"{stem}.txt"
    seg_file = labels_seg / f"{stem}.txt"

    if mode in {"both", "det"}:
        with open(det_file, "w", encoding="utf-8") as f:
            f.write("\n".join(det_lines) + ("\n" if det_lines else ""))
    elif det_file.exists():
        det_file.unlink()

    if mode in {"both", "seg"}:
        with open(seg_file, "w", encoding="utf-8") as f:
            f.write("\n".join(seg_lines) + ("\n" if seg_lines else ""))
    elif seg_file.exists():
        seg_file.unlink()


def remove_label_files_if_exists(stem, labels_det, labels_seg):
    det_file = labels_det / f"{stem}.txt"
    seg_file = labels_seg / f"{stem}.txt"
    if det_file.exists():
        det_file.unlink()
    if seg_file.exists():
        seg_file.unlink()


def process_one_image(
    model,
    image_path,
    class_name_to_id,
    negative_prompts,
    labels_det,
    labels_seg,
    vis_dir,
    save_vis,
    conf,
    min_box_conf,
    iou,
    imgsz,
    device,
    class_id_mode="name",
    show_preview=False,
    preview_scale=0.7,
    preview_file=None,
    enable_box_expand=False,
    box_expand=1.15,
    pixel_refine=False,
    pixel_refine_max_expand=1.25,
    pixel_refine_iou_min=0.55,
    pixel_refine_area_max=1.8,
    hard_small_target=False,
    hard_max_box_area_ratio=0.12,
    hard_large_area_ratio=0.30,
    hard_rescue_model=None,
    hard_rescue_conf=0.20,
    secondary_model=None,
    secondary_conf=0.30,
    secondary_iou=0.55,
    force_single_class=False,
    label_output_mode="both",
    image_index=None,
    image_total=None,
):
    step_prefix = ""
    if image_index is not None and image_total is not None:
        step_prefix = f"[{image_index}/{image_total}] "
    print(f"STEP: {step_prefix}{Path(image_path).name} - 读取图像")

    image = cv2.imread(str(image_path))
    if image is None:
        return {"image": str(image_path), "ok": False, "reason": "read_failed", "det": 0, "seg": 0, "stop": False}

    print(f"STEP: {step_prefix}{Path(image_path).name} - 模型推理")
    results = model.predict(image, conf=float(conf), iou=float(iou), imgsz=imgsz, device=device, verbose=False)
    if not results:
        remove_label_files_if_exists(image_path.stem, labels_det, labels_seg)
        return {"image": str(image_path), "ok": True, "reason": "no_result", "det": 0, "seg": 0, "stop": False}

    r = results[0]
    names = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}

    det_lines = []
    seg_lines = []
    seg_preview_items = []
    conf_sum = 0.0
    conf_count = 0
    neg_filtered = 0
    effective_min_box_conf = float(min_box_conf)
    effective_box_expand = float(box_expand)
    effective_pixel_refine_area_max = float(pixel_refine_area_max)
    effective_pixel_refine_iou_min = float(pixel_refine_iou_min)
    effective_hard_max_area = float(hard_max_box_area_ratio)
    effective_hard_large_area = float(hard_large_area_ratio)

    if hard_small_target:
        # 高难模式改为后处理过滤：保持原始识别召回，不在推理阶段收紧阈值。
        effective_hard_max_area = min(max(effective_hard_max_area, 0.005), 0.20)
        effective_hard_large_area = min(max(effective_hard_large_area, 0.10), 0.95)

    # 扩张开启时再收严，关闭时保持基础框不扩张。
    if enable_box_expand:
        effective_box_expand = min(effective_box_expand, 1.08)
        effective_refine_expand = min(float(pixel_refine_max_expand), 1.15)
    else:
        effective_refine_expand = float(pixel_refine_max_expand)

    secondary_support = {}
    if secondary_model is not None:
        try:
            print(f"STEP: {step_prefix}{Path(image_path).name} - 二次特性检查")
            secondary_results = secondary_model.predict(
                image,
                conf=float(secondary_conf),
                iou=float(secondary_iou),
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            if secondary_results:
                secondary_support = _collect_secondary_support_boxes(
                    secondary_results[0],
                    class_id_mode=class_id_mode,
                    class_name_to_id=class_name_to_id,
                    min_conf=secondary_conf,
                )
        except Exception as _e:
            print(f"二次检查失败，已跳过: {_e}")

    det_anchor_by_idx = {}
    det_candidates = []

    if r.boxes is not None and len(r.boxes) > 0:
        xywhn = r.boxes.xywhn.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        mask_xyn = r.masks.xyn if (r.masks is not None and r.masks.xyn is not None) else None

        for idx, box in enumerate(xywhn):
            if float(confs[idx]) < effective_min_box_conf:
                continue
            model_cls = int(cls_ids[idx])
            name = str(names.get(model_cls, model_cls))
            if should_reject_by_negative(name, negative_prompts):
                neg_filtered += 1
                continue
            if class_id_mode == "index" and model_cls in class_name_to_id:
                cls_id = class_name_to_id[model_cls]
            else:
                cls_id = class_name_to_id.get(normalize_text(name), None)
            if cls_id is None:
                continue
            if force_single_class:
                cls_id = 0
            fixed_box = box

            if pixel_refine:
                fixed_box = refine_box_with_pixel_correlation(
                    image=image,
                    xywhn=fixed_box,
                    max_expand=effective_refine_expand,
                    min_iou=effective_pixel_refine_iou_min,
                    max_area_ratio=effective_pixel_refine_area_max,
                )

            # 兜底策略：若分割可用，用分割外接框与检测框并集，减少仅框住目标局部的问题。
            if mask_xyn is not None and idx < len(mask_xyn):
                mask_box = _polygon_bbox_xywhn(mask_xyn[idx])
                if mask_box is not None:
                    if _should_union_mask_box(fixed_box, mask_box):
                        fixed_box = _union_xywhn(fixed_box, mask_box)

            if enable_box_expand:
                pre_expand_box = [float(v) for v in fixed_box]
                class_expand = effective_box_expand
                if _is_bird_like_name(name):
                    class_expand = min(max(class_expand, 1.12), 1.16)
                fixed_box = expand_xywhn(fixed_box, scale=class_expand)

                # 二次模型未支持扩张结果时，回退到扩张前框，避免串联误扩张。
                if secondary_support and not _has_secondary_support(fixed_box, cls_id, secondary_support):
                    fixed_box = pre_expand_box

            det_anchor_by_idx[idx] = [float(v) for v in fixed_box]
            det_lines.append(yolo_det_line_norm(cls_id, fixed_box))
            conf_sum += float(confs[idx])
            conf_count += 1
            det_candidates.append(
                {
                    "idx": int(idx),
                    "cls_id": int(cls_id),
                    "xywhn": [float(v) for v in fixed_box],
                }
            )

    if r.masks is not None and r.masks.xyn is not None:
        model_cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
        model_confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        model_xywhn = r.boxes.xywhn.cpu().numpy() if r.boxes is not None else []
        for idx, poly in enumerate(r.masks.xyn):
            if idx >= len(model_cls_ids):
                continue
            if idx < len(model_confs) and float(model_confs[idx]) < effective_min_box_conf:
                continue
            if _is_line_like_polygon(poly):
                continue
            model_cls = int(model_cls_ids[idx])
            name = str(names.get(model_cls, model_cls))
            if should_reject_by_negative(name, negative_prompts):
                continue
            if class_id_mode == "index" and model_cls in class_name_to_id:
                cls_id = class_name_to_id[model_cls]
            else:
                cls_id = class_name_to_id.get(normalize_text(name), None)
            if cls_id is None:
                continue
            if force_single_class:
                cls_id = 0
            anchor = det_anchor_by_idx.get(idx)
            if anchor is None and idx < len(model_xywhn):
                anchor = [float(v) for v in model_xywhn[idx]]
            clean_poly = _sanitize_seg_polygon_xyn(poly, anchor, image.shape) if anchor is not None else poly
            if clean_poly is None:
                continue
            line = yolo_seg_line_norm(cls_id, clean_poly)
            if line:
                seg_lines.append(line)
                seg_preview_items.append({"cls_id": int(cls_id), "poly": clean_poly})

    if hard_small_target and det_lines:
        has_small_det = False
        for line in det_lines:
            parsed = _parse_det_line_norm(line)
            if not parsed:
                continue
            _cid, xywhn = parsed
            if float(xywhn[2]) * float(xywhn[3]) <= effective_hard_max_area:
                has_small_det = True
                break

        has_small_seg = any(
            float(_polygon_area_ratio_xyn(it.get("poly", []))) <= effective_hard_max_area for it in seg_preview_items
        )
        has_valid_small_target = bool(has_small_det or has_small_seg)

        # 若本次已有有效小目标，仅删除超过总面积30%的大区域。
        if has_valid_small_target:
            kept_det = []
            for line in det_lines:
                parsed = _parse_det_line_norm(line)
                if not parsed:
                    continue
                _cid, xywhn = parsed
                if float(xywhn[2]) * float(xywhn[3]) <= effective_hard_large_area:
                    kept_det.append(line)
            det_lines = kept_det

            kept_seg_lines = []
            kept_seg_items = []
            for line, item in zip(seg_lines, seg_preview_items):
                area_ratio = float(_polygon_area_ratio_xyn(item.get("poly", [])))
                if area_ratio <= effective_hard_large_area:
                    kept_seg_lines.append(line)
                    kept_seg_items.append(item)
            seg_lines = kept_seg_lines
            seg_preview_items = kept_seg_items
        else:
            # 没有小目标时，在现有框选区域内用 yolo26l 再检一次，优先取小框。
            det_lines = []
            seg_lines = []
            seg_preview_items = []

            if hard_rescue_model is not None and det_candidates:
                h, w = image.shape[:2]
                for cand in det_candidates:
                    cx, cy, bw, bh = [float(v) for v in cand["xywhn"]]
                    if bw <= 0 or bh <= 0:
                        continue
                    rx1, ry1, rx2, ry2 = _xywhn_to_xyxy([cx, cy, bw, bh], w, h)
                    if rx2 - rx1 < 6 or ry2 - ry1 < 6:
                        continue

                    roi = image[ry1:ry2, rx1:rx2]
                    if roi is None or roi.size == 0:
                        continue

                    try:
                        rs = hard_rescue_model.predict(
                            roi,
                            conf=float(hard_rescue_conf),
                            iou=float(iou),
                            imgsz=imgsz,
                            device=device,
                            verbose=False,
                        )
                    except Exception:
                        continue

                    if not rs:
                        continue

                    rr = rs[0]
                    rnames = rr.names if isinstance(rr.names, dict) else {i: n for i, n in enumerate(rr.names)}
                    if rr.boxes is None or len(rr.boxes) <= 0:
                        continue

                    r_xywhn = rr.boxes.xywhn.cpu().numpy()
                    r_cls_ids = rr.boxes.cls.cpu().numpy().astype(int)
                    r_confs = rr.boxes.conf.cpu().numpy()
                    r_masks = rr.masks.xyn if (rr.masks is not None and rr.masks.xyn is not None) else None

                    for ridx, rbox in enumerate(r_xywhn):
                        if float(r_confs[ridx]) < float(hard_rescue_conf):
                            continue
                        model_cls = int(r_cls_ids[ridx])
                        name = str(rnames.get(model_cls, model_cls))
                        if should_reject_by_negative(name, negative_prompts):
                            continue
                        if class_id_mode == "index" and model_cls in class_name_to_id:
                            cls_id = class_name_to_id[model_cls]
                        else:
                            cls_id = class_name_to_id.get(normalize_text(name), None)
                        if cls_id is None:
                            continue
                        if force_single_class:
                            cls_id = 0

                        gbox = _to_global_xywhn_from_roi(rbox, (rx1, ry1, rx2, ry2), w, h)
                        area_ratio = float(gbox[2]) * float(gbox[3])
                        if area_ratio > effective_hard_max_area:
                            continue
                        if area_ratio > effective_hard_large_area:
                            continue
                        inserted = _append_det_line_if_new(det_lines, int(cls_id), gbox, iou_thr=0.80)
                        if not inserted:
                            continue

                        if r_masks is not None and ridx < len(r_masks):
                            gpoly = _to_global_poly_xyn_from_roi(r_masks[ridx], (rx1, ry1, rx2, ry2), w, h)
                            if _is_line_like_polygon(gpoly):
                                continue
                            clean_poly = _sanitize_seg_polygon_xyn(gpoly, gbox, image.shape)
                            if clean_poly is None:
                                continue
                            if float(_polygon_area_ratio_xyn(clean_poly)) > effective_hard_large_area:
                                continue
                            sline = yolo_seg_line_norm(int(cls_id), clean_poly)
                            if sline:
                                seg_lines.append(sline)
                                seg_preview_items.append({"cls_id": int(cls_id), "poly": clean_poly})

    print(f"STEP: {step_prefix}{Path(image_path).name} - 写入标注")
    if det_lines or seg_lines:
        write_label_files(
            image_path.stem,
            det_lines,
            seg_lines,
            labels_det,
            labels_seg,
            label_output_mode=label_output_mode,
        )
    else:
        remove_label_files_if_exists(image_path.stem, labels_det, labels_seg)
        return {
            "image": str(image_path),
            "ok": True,
            "reason": "no_result",
            "det": 0,
            "seg": 0,
            "conf_sum": conf_sum,
            "conf_count": conf_count,
            "neg_filtered": neg_filtered,
            "stop": False,
        }

    vis_save = None
    if save_vis:
        vis_save = r.plot()

    vis_preview = None
    if show_preview or preview_file:
        print(f"STEP: {step_prefix}{Path(image_path).name} - 渲染预览")
        vis_preview = render_preview_from_filtered_labels(image, det_lines, seg_preview_items)

    if save_vis and vis_save is not None:
        cv2.imwrite(str(vis_dir / image_path.name), vis_save)

    if preview_file and vis_preview is not None:
        cv2.imwrite(str(preview_file), vis_preview)
        print(f"PREVIEW_FILE: {preview_file}")

    if show_preview and vis_preview is not None:
        preview = vis_preview
        if preview_scale != 1.0:
            h0, w0 = preview.shape[:2]
            nw = max(320, int(w0 * preview_scale))
            nh = max(240, int(h0 * preview_scale))
            preview = cv2.resize(preview, (nw, nh))
        cv2.imshow("标注实时预览(按Q停止)", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return {
                "image": str(image_path),
                "ok": False,
                "reason": "user_stop",
                "det": len(det_lines),
                "seg": len(seg_lines),
                "conf_sum": conf_sum,
                "conf_count": conf_count,
                "stop": True,
            }

    return {
        "image": str(image_path),
        "ok": True,
        "reason": "done",
        "det": len(det_lines),
        "seg": len(seg_lines),
        "conf_sum": conf_sum,
        "conf_count": conf_count,
        "neg_filtered": neg_filtered,
        "stop": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="YOLOE 基于提示词的数据集自动标注工具")
    parser.add_argument("--model", default="yoloe-26x-seg.pt", help="模型路径")
    parser.add_argument("--images", required=True, help="图片目录（会递归扫描）")
    parser.add_argument("--output", default="dataset_annotations", help="输出目录")
    parser.add_argument("--prompts-file", default=None, help="提示词文件（每行一个）")
    parser.add_argument("--prompt-text", default="", help="直接输入提示词，逗号/空格分隔")
    parser.add_argument("--negative-prompts-file", default=None, help="负提示词文件（每行一个）")
    parser.add_argument("--negative-prompt-text", default="", help="负提示词，逗号/空格分隔")
    parser.add_argument("--component-combine", action="store_true", help="将输出标签类别统一为固定词")
    parser.add_argument("--component-name", default="", help="成分组合固定类别名")
    parser.add_argument("--target-mode", choices=["auto", "single", "multi"], default="auto", help="目标模式")
    parser.add_argument("--label-output-mode", choices=["both", "det", "seg"], default="both", help="输出标签类型")
    parser.add_argument("--open-vocab", choices=["auto", "on", "off"], default="auto", help="开放词表模式: auto自动/on强制/off关闭")
    parser.add_argument("--enable-llm-expand", action="store_true", help="启用 LLM 提示词扩展")
    parser.add_argument("--force-llm-expand", action="store_true", help="强制启用LLM扩词，失败即退出")
    parser.add_argument("--llm-provider", default="openai-compatible", help="LLM 提供方(openai-compatible/ollama)")
    parser.add_argument("--llm-model", default="deepseek-chat", help="LLM 模型名称")
    parser.add_argument("--llm-api-base", default="https://api.deepseek.com", help="LLM 接口基础地址")
    parser.add_argument("--llm-api-key", default=None, help="LLM 接口密钥")
    parser.add_argument("--llm-max-terms", type=int, default=300, help="扩展后最大提示词数")
    parser.add_argument("--llm-timeout", type=int, default=80, help="LLM 请求超时秒数")
    parser.add_argument("--domain-hint", default="", help="领域提示，如: abnormal cell / wall crack")
    parser.add_argument("--detect-max-terms", type=int, default=36, help="参与检测的最大提示词数（过大易降召回）")
    parser.add_argument("--conf", type=float, default=0.25, help="推理置信度阈值")
    parser.add_argument("--min-box-conf", type=float, default=0.20, help="标注写入最小置信度")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS 重叠阈值")
    parser.add_argument("--imgsz", type=int, default=1024, help="推理输入尺寸")
    parser.add_argument("--enable-box-expand", action="store_true", help="启用框扩张补齐边缘部位")
    parser.add_argument("--box-expand", type=float, default=1.15, help="检测框写入前外扩倍率，缓解漏标尾部/把手")
    parser.add_argument("--pixel-refine", action="store_true", help="启用像素相关性修正框，提升目标完整性")
    parser.add_argument("--pixel-refine-max-expand", type=float, default=1.25, help="像素修正最大外扩倍率")
    parser.add_argument("--pixel-refine-iou-min", type=float, default=0.55, help="像素修正最小IoU约束")
    parser.add_argument("--pixel-refine-area-max", type=float, default=1.8, help="像素修正最大面积倍率")
    parser.add_argument("--hard-small-target", action="store_true", help="高难小目标模式：限制大框并提高精度")
    parser.add_argument("--hard-max-box-area-ratio", type=float, default=0.12, help="高难模式单框最大面积占比")
    parser.add_argument("--hard-large-area-ratio", type=float, default=0.30, help="高难模式删除大区域阈值(面积占比)")
    parser.add_argument("--hard-rescue-model", default="yoloe-26l-seg.pt", help="高难模式二次补检模型路径")
    parser.add_argument("--hard-rescue-conf", type=float, default=0.20, help="高难模式二次补检置信度")
    parser.add_argument("--second-check-model", default="yoloe-26l-seg.pt", help="二次特性检查模型路径")
    parser.add_argument("--disable-second-check", action="store_true", help="关闭二次特性检查")
    parser.add_argument("--second-check-conf", type=float, default=0.30, help="二次检查置信度阈值")
    parser.add_argument("--second-check-iou", type=float, default=0.55, help="二次检查NMS阈值")
    parser.add_argument("--device", default="auto", help="推理设备: auto/cpu/cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=0, help="CPU线程数(0=自动)")
    parser.add_argument("--save-vis", action="store_true", help="保存可视化结果")
    parser.add_argument("--show-preview", action="store_true", help="显示每张图像的实时标注小窗")
    parser.add_argument("--preview-scale", type=float, default=0.7, help="预览窗口缩放比例")
    parser.add_argument("--preview-file", default=None, help="将实时预览帧写入该文件路径（供GUI内嵌显示）")
    args = parser.parse_args(argv)

    if args.force_llm_expand:
        args.enable_llm_expand = True

    target_mode = args.target_mode
    if target_mode == "auto":
        target_mode = "single" if args.component_combine else "multi"

    if target_mode == "single" and (not str(args.component_name).strip()):
        print("单目标模式需要提供 --component-name")
        return

    if not os.path.exists(args.model):
        print(f"未找到模型文件: {args.model}")
        return

    prompts, prompt_source = load_prompt_list(args)
    negative_prompts, negative_source = load_negative_prompt_list(args)
    original_prompts = list(prompts)
    print(f"提示词来源: {prompt_source}")
    print(f"负提示词来源: {negative_source}")
    if prompts:
        print(f"提示词(英文优先): {', '.join(prompts[:20])}")
    if negative_prompts:
        print(f"负提示词(英文优先): {', '.join(negative_prompts[:20])}")
    negative_expand_info = {"mode": "none", "error": None}
    if negative_prompts:
        negative_prompts, negative_expand_info = expand_negative_prompts_strict(negative_prompts, args)
        print(f"负提示词扩展模式: {negative_expand_info.get('mode')}")
        if negative_expand_info.get("error"):
            print(f"负提示词扩展异常: {negative_expand_info.get('error')}")
        print(f"负提示词数量: {len(negative_prompts)} | {', '.join(negative_prompts[:20])}")
        print("NEG_LLM_EXPANDED_PROMPTS_BEGIN")
        for p in negative_prompts:
            print(f"NEG_LLM_PROMPT: {p}")
        neg_groups = build_prompt_groups_for_display(negative_prompts, negative_prompts)
        for seed, terms in neg_groups.items():
            print(f"NEG_LLM_GROUP: {seed} => {' | '.join(terms)}")
        print("NEG_LLM_EXPANDED_PROMPTS_END")

    if prompts:
        if args.enable_llm_expand:
            print(
                f"开始LLM扩词: provider={args.llm_provider} model={args.llm_model} timeout={args.llm_timeout}s"
            )
        prompts, expand_info = expand_prompts(
            seed_prompts=prompts,
            enable_llm=args.enable_llm_expand,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_api_base=args.llm_api_base,
            llm_api_key=args.llm_api_key,
            llm_max_terms=args.llm_max_terms,
            llm_timeout=args.llm_timeout,
            domain_hint=args.domain_hint,
        )
        print(f"提示词扩展模式: {expand_info['mode']}")
        print(f"原始提示词: {len(original_prompts)} | 扩展后: {len(prompts)}")
        if target_mode == "multi":
            before_cnt = len(prompts)
            prompts = prune_overlapping_prompts_for_multi(
                seed_prompts=original_prompts or prompts,
                expanded_prompts=prompts,
                max_terms=args.llm_max_terms,
            )
            print(f"多目标防重叠过滤: {before_cnt} -> {len(prompts)}")
        if args.enable_llm_expand:
            print("LLM_EXPANDED_PROMPTS_BEGIN")
            for p in prompts:
                print(f"LLM_PROMPT: {p}")
            groups = build_prompt_groups_for_display(original_prompts or prompts, prompts)
            for seed, terms in groups.items():
                print(f"LLM_GROUP: {seed} => {' | '.join(terms)}")
            print("LLM_EXPANDED_PROMPTS_END")
        if expand_info.get("error"):
            print(f"LLM 扩展异常，已回退: {expand_info['error']}")
            if args.force_llm_expand:
                print("你启用了强制LLM扩词，程序退出")
                return
    else:
        expand_info = {"mode": "none", "added": 0, "error": None}

    images = collect_images(args.images)
    if not images:
        print(f"目录无可用图片: {args.images}")
        return

    output_root, labels_det, labels_seg, vis_dir = ensure_dirs(args.output, args.save_vis)

    runtime_device = resolve_runtime_device(args)
    print(f"推理设备: 请求={args.device} 实际={runtime_device}")

    if runtime_device == "cpu":
        threads = args.cpu_threads if args.cpu_threads > 0 else max(1, multiprocessing.cpu_count() - 1)
        try:
            cv2.setNumThreads(threads)
        except Exception:
            pass
        if torch is not None:
            try:
                torch.set_num_threads(threads)
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(max(1, threads // 2))
            except Exception:
                pass
        print(f"CPU模式已启用，线程数: {threads}")
    elif args.cpu_threads > 0:
        print("当前为GPU模式，CPU线程设置将被忽略")

    detect_prompts = build_detection_prompt_subset(
        expanded_prompts=prompts,
        seed_prompts=original_prompts or prompts,
        max_terms=args.detect_max_terms,
    )
    print(f"检测词表收敛: 扩展词 {len(prompts)} -> 检测词 {len(detect_prompts)}")

    with open(output_root / "prompts_used.txt", "w", encoding="utf-8") as f:
        for p in detect_prompts:
            f.write(p + "\n")

    model = YOLO(args.model)
    if runtime_device == "cpu":
        try:
            model.to("cpu")
        except Exception:
            pass
    print(f"模型加载成功: {args.model}")
    secondary_model = None
    hard_rescue_model = None
    if not args.disable_second_check:
        second_path = Path(args.second_check_model)
        if second_path.exists():
            try:
                secondary_model = YOLO(str(second_path))
                if runtime_device == "cpu":
                    try:
                        secondary_model.to("cpu")
                    except Exception:
                        pass
                print(f"二次检查模型加载成功: {second_path}")
            except Exception as e:
                print(f"二次检查模型加载失败，已跳过: {e}")
        else:
            print(f"未找到二次检查模型，已跳过: {second_path}")

    if args.hard_small_target:
        rescue_path = Path(args.hard_rescue_model)
        if (secondary_model is not None) and (str(rescue_path) == str(Path(args.second_check_model))):
            hard_rescue_model = secondary_model
            print(f"高难补检模型复用二次检查模型: {rescue_path}")
        elif rescue_path.exists():
            try:
                hard_rescue_model = YOLO(str(rescue_path))
                if runtime_device == "cpu":
                    try:
                        hard_rescue_model.to("cpu")
                    except Exception:
                        pass
                print(f"高难补检模型加载成功: {rescue_path}")
            except Exception as e:
                print(f"高难补检模型加载失败，已跳过: {e}")
        else:
            print(f"未找到高难补检模型，已跳过: {rescue_path}")
    builtin_names = extract_name_list(model.names)
    active_names = []
    mode_used = "builtin"
    class_id_mode = "name"

    if detect_prompts:
        print(f"检测提示词数量: {len(detect_prompts)}")

        should_open_vocab = args.open_vocab in {"auto", "on"}
        if should_open_vocab:
            print("尝试设置开放词表...")
            try:
                model.set_classes(detect_prompts)
                active_names = detect_prompts
                mode_used = "open-vocab"
                class_id_mode = "index"
                print("开放词表设置成功")
                if secondary_model is not None:
                    try:
                        secondary_model.set_classes(detect_prompts)
                        print("二次检查模型开放词表设置成功")
                    except Exception as _e:
                        print(f"二次检查模型开放词表设置失败，回退内置类别: {_e}")
                if hard_rescue_model is not None and hard_rescue_model is not secondary_model:
                    try:
                        hard_rescue_model.set_classes(detect_prompts)
                        print("高难补检模型开放词表设置成功")
                    except Exception as _e:
                        print(f"高难补检模型开放词表设置失败，回退内置类别: {_e}")
            except Exception as e:
                if args.open_vocab == "on":
                    print("set_classes 失败，且你指定了 --open-vocab on，程序退出")
                    print("请确认 mobileclip2_b.ts 在项目目录中可用")
                    print(f"错误: {e}")
                    return
                print("开放词表设置失败，自动回退到固定类别匹配")
                print(f"错误: {e}")

        if not active_names:
            active_names = select_builtin_classes(builtin_names, detect_prompts)
            mode_used = "builtin-filtered"
    else:
        active_names = builtin_names
        mode_used = "builtin-all"

    print(f"标注类别模式: {mode_used}")
    print(f"最终参与标注类别数: {len(active_names)}")
    if args.hard_small_target:
        print(
            "高难小目标模式已启用 | "
            f"max_box_area={args.hard_max_box_area_ratio} | "
            f"drop_large_area={args.hard_large_area_ratio}"
        )

    force_single_class = bool(target_mode == "single")

    if class_id_mode == "index":
        class_name_to_id, output_classes = build_open_vocab_class_id_map(
            detect_prompts=active_names,
            seed_prompts=original_prompts or prompts,
        )
    else:
        class_name_to_id = {normalize_text(name): i for i, name in enumerate(active_names)}
        output_classes = active_names

    if force_single_class:
        output_classes = [str(args.component_name).strip()]

    write_classes_file(output_root, output_classes)

    total_det = 0
    total_seg = 0
    total_conf_sum = 0.0
    total_conf_count = 0
    total_neg_filtered = 0
    no_result_count = 0
    ok_count = 0
    failed = []

    print(f"开始标注，共 {len(images)} 张图片")
    for i, image_path in enumerate(images, start=1):
        result = process_one_image(
            model=model,
            image_path=image_path,
            class_name_to_id=class_name_to_id,
            negative_prompts=negative_prompts,
            labels_det=labels_det,
            labels_seg=labels_seg,
            vis_dir=vis_dir,
            save_vis=args.save_vis,
            conf=args.conf,
            min_box_conf=args.min_box_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=runtime_device,
            class_id_mode=class_id_mode,
            show_preview=args.show_preview,
            preview_scale=args.preview_scale,
            preview_file=args.preview_file,
            enable_box_expand=args.enable_box_expand,
            box_expand=args.box_expand,
            pixel_refine=args.pixel_refine,
            pixel_refine_max_expand=args.pixel_refine_max_expand,
            pixel_refine_iou_min=args.pixel_refine_iou_min,
            pixel_refine_area_max=args.pixel_refine_area_max,
            hard_small_target=args.hard_small_target,
            hard_max_box_area_ratio=args.hard_max_box_area_ratio,
            hard_large_area_ratio=args.hard_large_area_ratio,
            hard_rescue_model=hard_rescue_model,
            hard_rescue_conf=args.hard_rescue_conf,
            secondary_model=secondary_model,
            secondary_conf=args.second_check_conf,
            secondary_iou=args.second_check_iou,
            force_single_class=force_single_class,
            label_output_mode=args.label_output_mode,
            image_index=i,
            image_total=len(images),
        )

        if result.get("stop"):
            print("收到用户停止指令，结束标注")
            failed.append(result)
            break

        if result["ok"]:
            ok_count += 1
            total_det += result["det"]
            total_seg += result["seg"]
            total_conf_sum += float(result.get("conf_sum", 0.0))
            total_conf_count += int(result.get("conf_count", 0))
            total_neg_filtered += int(result.get("neg_filtered", 0))
            if result.get("reason") == "no_result" or (result.get("det", 0) == 0 and result.get("seg", 0) == 0):
                no_result_count += 1
        else:
            failed.append(result)

        avg_conf = (total_conf_sum / float(total_conf_count)) if total_conf_count > 0 else 0.0
        print(
            f"进度 {i}/{len(images)} | det {total_det} | seg {total_seg} | 未识别 {no_result_count} | 失败 {len(failed)} | 平均置信 {avg_conf:.3f}"
        )

    summary = {
        "model": args.model,
        "images_dir": args.images,
        "output_dir": str(output_root),
        "images_total": len(images),
        "images_ok": ok_count,
        "images_failed": len(failed),
        "objects_det": total_det,
        "objects_seg": total_seg,
        "avg_confidence": round((total_conf_sum / float(total_conf_count)), 6) if total_conf_count > 0 else 0.0,
        "objects_neg_filtered": total_neg_filtered,
        "images_no_result": no_result_count,
        "prompts_count": len(prompts),
        "detect_prompts_count": len(detect_prompts),
        "prompts_original_count": len(original_prompts),
        "prompt_expand_mode": expand_info.get("mode"),
        "prompt_expand_added": expand_info.get("added"),
        "prompt_expand_error": expand_info.get("error"),
        "active_classes_count": len(output_classes),
        "mode_used": mode_used,
        "target_mode": target_mode,
        "label_output_mode": args.label_output_mode,
        "prompts_file": args.prompts_file,
        "negative_prompts_file": args.negative_prompts_file,
        "negative_prompt_count": len(negative_prompts),
        "negative_prompt_expand_mode": negative_expand_info.get("mode"),
        "negative_prompt_expand_error": negative_expand_info.get("error"),
        "enable_llm_expand": args.enable_llm_expand,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "llm_timeout": args.llm_timeout,
        "detect_max_terms": args.detect_max_terms,
        "force_llm_expand": args.force_llm_expand,
        "domain_hint": args.domain_hint,
        "open_vocab": args.open_vocab,
        "conf": args.conf,
        "min_box_conf": args.min_box_conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "enable_box_expand": args.enable_box_expand,
        "box_expand": args.box_expand,
        "pixel_refine": args.pixel_refine,
        "pixel_refine_max_expand": args.pixel_refine_max_expand,
        "pixel_refine_iou_min": args.pixel_refine_iou_min,
        "pixel_refine_area_max": args.pixel_refine_area_max,
        "hard_small_target": args.hard_small_target,
        "hard_max_box_area_ratio": args.hard_max_box_area_ratio,
        "second_check_enabled": (not args.disable_second_check),
        "second_check_model": args.second_check_model,
        "second_check_conf": args.second_check_conf,
        "second_check_iou": args.second_check_iou,
        "attention_hsv_parts": list(ATTENTION_PART_HSV_TABLE.keys()),
        "device_requested": args.device,
        "device_resolved": runtime_device,
        "cpu_threads": args.cpu_threads,
        "component_combine": bool(args.component_combine),
        "component_name": str(args.component_name),
        "show_preview": args.show_preview,
        "preview_scale": args.preview_scale,
        "failed_samples": failed[:50],
    }

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("标注完成")
    print(f"检测标签目录: {labels_det}")
    print(f"分割标签目录: {labels_seg}")
    if args.save_vis:
        print(f"可视化目录: {vis_dir}")
    print(f"摘要文件: {output_root / 'summary.json'}")

    if args.show_preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
