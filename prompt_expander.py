import json
import os
import re
from urllib import request, error


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        key = normalize_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def split_terms(text):
    parts = re.split(r"[\n,，;；、|]+", text)
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text):
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalize_text(text))


def _color_conflict_filter(seed_prompts, terms):
    seed_norm = [normalize_text(x) for x in seed_prompts]
    has_green_apple = any(("green apple" in s) or ("青苹果" in s) for s in seed_norm)
    has_red_apple_seed = any(("red apple" in s) or ("红苹果" in s) for s in seed_norm)

    if not has_green_apple or has_red_apple_seed:
        return terms

    blocked = ("red apple", "红苹果", "ripe red apple", "red fruit")
    filtered = []
    for term in terms:
        t = normalize_text(term)
        if any(b in t for b in blocked):
            continue
        filtered.append(term)
    return filtered


def _vehicle_type_conflict_filter(seed_prompts, terms):
    seed_norm = [normalize_text(x) for x in seed_prompts]
    has_suv = any(("suv" in s) or ("越野车" in s) for s in seed_norm)
    has_sedan = any(("sedan" in s) or ("轿车" in s) for s in seed_norm)

    if has_suv and not has_sedan:
        blocked = (
            "sedan",
            "轿车",
            "三厢",
            "low ground clearance",
            "低离地间隙",
        )
        return [t for t in terms if not any(b in normalize_text(t) for b in blocked)]

    if has_sedan and not has_suv:
        blocked = (
            "suv",
            "越野",
            "high ground clearance",
            "高离地间隙",
            "roof rack",
            "车顶行李架",
        )
        return [t for t in terms if not any(b in normalize_text(t) for b in blocked)]

    return terms


def _sedan_roof_overexpansion_filter(seed_prompts, terms):
    seed_norm = [normalize_text(x) for x in seed_prompts]
    has_sedan = any(("sedan" in s) or ("轿车" in s) for s in seed_norm)
    if not has_sedan:
        return terms

    material_tokens = {
        "metal", "fabric", "vinyl", "leather", "cloth", "plastic", "carbon", "fiber",
        "aluminum", "steel", "iron", "copper", "brass", "bronze", "gold", "silver",
        "platinum", "titanium", "magnesium", "zinc", "tin", "lead", "nickel",
    }

    filtered = []
    roof_count = 0
    roof_cap = 3
    for term in terms:
        t = normalize_text(term)
        tks = set(_tokenize(t))
        is_roof = "roof" in t or "车顶" in t
        if is_roof and (tks & material_tokens):
            continue
        if is_roof:
            if roof_count >= roof_cap:
                continue
            roof_count += 1
        filtered.append(term)
    return filtered


def _limit_dominant_token_family(seed_prompts, terms):
    if len(terms) < 10:
        return terms

    seed_tokens = set()
    for s in seed_prompts:
        seed_tokens.update(_tokenize(s))

    stop_tokens = {
        "object", "item", "target", "visible", "clear", "shape", "feature", "appearance",
        "物体", "目标", "可见", "清晰", "外观", "特征",
    }

    token_freq = {}
    term_tokens = []
    for term in terms:
        tks = set(_tokenize(term))
        tks = {tk for tk in tks if tk not in seed_tokens and tk not in stop_tokens and len(tk) >= 2}
        term_tokens.append(tks)
        for tk in tks:
            token_freq[tk] = token_freq.get(tk, 0) + 1

    if not token_freq:
        return terms

    dominant, freq = max(token_freq.items(), key=lambda kv: kv[1])
    if freq < max(6, int(len(terms) * 0.55)):
        return terms

    cap = max(4, int(len(terms) * 0.35))
    kept = []
    dom_count = 0
    for term, tks in zip(terms, term_tokens):
        if dominant in tks:
            if dom_count >= cap:
                continue
            dom_count += 1
        kept.append(term)
    return kept


def _re_rank_terms(seed_prompts, terms, max_terms):
    seeds = dedupe_keep_order(seed_prompts)
    seed_norm = [normalize_text(x) for x in seeds]
    anchor_tokens = set()
    for s in seed_norm:
        for tk in _tokenize(s):
            if len(tk) >= 2:
                anchor_tokens.add(tk)

    scored = []
    feature_keywords = {
        "green", "red", "black", "white", "gray", "silver", "blue",
        "青", "红", "黑", "白", "灰", "银", "蓝",
        "spherical", "round", "elongated", "boxy", "tall", "wide", "narrow",
        "球形", "圆形", "细长", "方正", "高", "宽", "窄",
        "texture", "smooth", "rough", "speckles", "pattern",
        "纹理", "光滑", "粗糙", "斑点", "花纹",
        "clearance", "wheel", "wheel-arch", "roof", "rack", "five-door",
        "离地间隙", "轮毂", "轮拱", "车顶", "行李架", "五门",
        "接近角", "离去角", "侧窗", "车头", "纵轨", "横杆", "侧踏板", "备胎", "后门", "后风挡",
        "进气格栅", "腰线", "尾灯", "刹车灯", "扰流板", "后视镜", "宽体", "轮眉", "防擦条", "花纹",
    }
    for idx, term in enumerate(terms):
        t = normalize_text(term)
        score = 0

        if t in seed_norm:
            score += 100

        for s in seed_norm:
            if s and (s in t or t in s):
                score += 20

        for tk in anchor_tokens:
            if tk in t:
                score += 4

        tks = _tokenize(t)
        if any((tk in feature_keywords) for tk in tks):
            score += 2

        # 软惩罚过长短语，避免LLM输出整句描述
        word_count = len(tks)
        if word_count > 8:
            score -= (word_count - 8)

        scored.append((score, idx, term))

    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: (-x[0], x[1]))
    ordered = [x[2] for x in scored]
    ordered = dedupe_keep_order(ordered)

    return ordered[:max_terms]


def _prefer_english_terms(primary_terms, fallback_terms, max_terms=120):
    def has_ascii_alpha(text):
        return bool(re.search(r"[a-zA-Z]", str(text)))

    primary = dedupe_keep_order(primary_terms)
    fallback = dedupe_keep_order(fallback_terms)

    english = [t for t in primary if has_ascii_alpha(t)]
    if not english:
        english = [t for t in fallback if has_ascii_alpha(t)]
    if not english:
        english = primary or fallback

    return dedupe_keep_order(english)[:max_terms]


def _drop_non_object_terms(terms):
    banned = [
        "close-up",
        "extreme close-up",
        "under poor lighting",
        "poor lighting",
        "blurred",
        "occluded",
        "scene",
        "surface",
        "表面",
        "surface texture",
        "clear boundary",
        "typical shape",
        "appearance",
        "morphology",
        "high ground clearance",
        "posture",
        "road posture",
        "city road posture",
        "wheel design",
        "ground clearance shadow",
        "shadow under vehicle",
        "grip",
        "grip style",
        "holding posture",
        "握法",
        "持握",
        "持刀姿态",
    ]
    out = []
    for term in terms:
        t = normalize_text(term)
        if any(b in t for b in banned):
            continue
        out.append(term)
    return out


def _descriptive_terms_for_seed(seed):
    s = normalize_text(seed)

    if "parrot" in s or "鹦鹉" in s:
        return [
            seed,
            "parrot",
            "whole parrot",
            "full-body parrot",
            "entire parrot silhouette",
            "parrot bird",
            "hooked beak parrot",
            "long tail parrot",
            "parrot with wings folded",
            "colorful parrot",
            "鹦鹉",
            "完整鹦鹉",
            "鹦鹉全身",
            "鹦鹉整体轮廓",
            "鹦鹉长尾",
            "钩状喙鹦鹉",
        ]

    if "dagger" in s or "匕首" in s or "短剑" in s:
        return [
            seed,
            "dagger",
            "whole dagger",
            "entire dagger",
            "full dagger silhouette",
            "short dagger",
            "double-edged dagger",
            "pointed blade",
            "short straight blade",
            "narrow blade",
            "symmetrical blade",
            "metal blade",
            "dagger handle",
            "compact knife",
            "small stabbing knife",
            "匕首",
            "短剑",
            "双刃匕首",
            "尖刃",
            "短直刃",
            "窄刃",
            "对称刀身",
            "金属刀刃",
            "短刀柄",
            "小型刺刀",
            "刀尖清晰",
            "刀身反光",
        ]

    top_view_markers = ("top-view", "top view", "aerial", "bird", "俯视", "顶视", "航拍")
    sedan_markers = ("sedan", "轿车", "小轿车")
    front_view_markers = ("front-view", "front view", "frontal", "front", "正面", "前视", "车头")

    if any(m in s for m in front_view_markers) and any(m in s for m in sedan_markers):
        return [
            seed,
            "front-view sedan",
            "frontal sedan",
            "front-facing sedan",
            "sedan front fascia",
            "sedan grille",
            "sedan headlights",
            "sedan hood line",
            "three-box sedan",
            "lower body sedan",
            "narrow wheel arch sedan",
            "轿车正面",
            "前视轿车",
            "正面小轿车",
            "轿车前脸",
            "轿车进气格栅",
            "轿车前大灯",
            "轿车引擎盖线条",
            "三厢车前视轮廓",
        ]

    if any(m in s for m in top_view_markers) and any(m in s for m in sedan_markers):
        return [
            seed,
            "top-view sedan",
            "aerial-view sedan",
            "bird-eye-view sedan",
            "sedan",
            "small sedan",
            "three-box car",
            "roof visible",
            "trunk visible from top",
            "hood visible from top",
            "top-down vehicle silhouette",
            "compact car footprint",
            "俯视轿车",
            "俯视小轿车",
            "顶视轿车",
            "航拍轿车",
            "车顶清晰可见",
            "后备箱顶面可见",
            "引擎盖顶面可见",
            "顶视三厢轮廓",
            "车体占地较紧凑",
        ]

    if "suv" in s or "越野车" in s or "suv车" in s:
        return [
            seed,
            "suv",
            "SUV车",
            "sport utility vehicle",
            "taller body",
            "boxy silhouette",
            "larger wheel arches",
            "roof rails",
            "roof cross bars",
            "side steps",
            "external spare tire",
            "short front overhang",
            "short rear overhang",
            "high seating position",
            "off-road tire tread",
            "宽大轮胎",
            "车身厚重",
            "离地间隙大",
            "接近角大",
            "离去角大",
            "侧窗方正",
            "车头厚重",
            "方形轮拱",
            "车顶纵轨",
            "车顶横杆",
            "侧踏板",
            "外挂备胎",
            "后门侧开",
            "后风挡可开",
            "大面积进气格栅",
            "高腰线",
            "方形尾灯",
            "高位刹车灯",
            "车顶扰流板",
            "方形后视镜",
            "宽体车身",
            "轮眉外扩",
            "车身防擦条",
            "越野轮胎花纹",
        ]

    if "sedan" in s or "轿车" in s:
        return [
            seed,
            "sedan",
            "轿车",
            "three-box car",
            "lower ground clearance",
            "lower body",
            "longer trunk section",
            "smooth roofline",
            "smaller wheel arches",
            "long wheelbase look",
            "三厢车身",
            "低离地间隙",
            "较低车身",
            "较长后备箱段",
            "平顺车顶线条",
            "小轮拱",
            "low stance",
            "longer trunk than suv",
            "smaller wheel arch than suv",
        ]

    if "青苹果" in s or "green apple" in s:
        return [
            seed,
            "green apple",
            "apple",
            "green",
            "greenish",
            "spherical",
            "tiny speckles",
            "smooth skin",
            "青绿色",
            "球形",
            "少量斑点",
            "表皮光滑",
        ]

    if "apple" in s or "苹果" in s:
        return [
            seed,
            "apple",
            "fruit",
            "spherical",
            "smooth peel",
            "苹果果实",
            "球形",
        ]

    if "裂缝" in s or "crack" in s:
        return [
            seed,
            "crack",
            "fissure",
            "thin elongated",
            "irregular narrow",
            "细长",
            "不规则",
            "线状",
        ]

    if "细胞" in s or "cell" in s:
        return [
            seed,
            "cell",
            "irregular contour",
            "hyperchromatic nucleus",
            "abnormal nucleus-cytoplasm ratio",
            "轮廓不规则",
            "核深染",
        ]

    # 通用类目兜底：保持与种子词强绑定，输出可直接用于检测的对象短语。
    return dedupe_keep_order([
        seed,
        f"{seed}",
        f"{seed} shape",
        f"{seed} structure",
    ])


def refine_expanded_terms(seed_prompts, terms, max_terms):
    out = dedupe_keep_order(terms)
    out = _color_conflict_filter(seed_prompts, out)
    out = _vehicle_type_conflict_filter(seed_prompts, out)
    out = _sedan_roof_overexpansion_filter(seed_prompts, out)
    out = _drop_non_object_terms(out)
    out = _limit_dominant_token_family(seed_prompts, out)
    out = _re_rank_terms(seed_prompts, out, max_terms)
    if not out:
        out = dedupe_keep_order(seed_prompts)
    return out


def heuristic_expand(seed_prompts, domain_hint=""):
    _ = domain_hint
    expanded = []
    for seed in seed_prompts:
        expanded.extend(_descriptive_terms_for_seed(seed))
    return dedupe_keep_order(expanded)


def build_visual_prototype_terms(seed_prompts, domain_hint=""):
    """构建描述化外观词，避免跨类别发散。"""
    hints = []
    for seed in seed_prompts:
        hints.append(seed.strip())

    _ = domain_hint

    return dedupe_keep_order(hints)


def openai_compatible_expand(
    seed_prompts,
    domain_hint="",
    model="gpt-4o-mini",
    api_base=None,
    api_key=None,
    max_terms=200,
    temperature=0.1,
    request_timeout=45,
):
    api_base = (api_base or os.getenv("LLM_API_BASE") or "https://api.openai.com/v1").rstrip("/")
    api_key = api_key or os.getenv("LLM_API_KEY")

    if not api_key:
        raise RuntimeError("缺少 LLM API Key，请设置 --llm-api-key 或环境变量 LLM_API_KEY")

    system_prompt = (
        "你是视觉检测提示词工程专家。"
        "当前任务用于 YOLOE-26x 开放词表提示词，不是自然语言描述或caption。"
        "先在内部思考目标的可见样貌特征（颜色、形状、纹理、边界、常见缺陷），再输出拟合检测短语。"
        "不要做跨类别同义词扩展。"
        "请按实体词与样貌特征词分离输出，而不是实体+特征合并短语。"
        "实体词和特征词都用短词组，便于开放词表独立匹配。"
        "若目标与相近类别容易混淆（例如 SUV 与轿车），必须输出可区分特征词。"
        "无论类别是否有专用模板，输出词必须与种子词保持强语义锚定。"
        "不要引入与种子词冲突的属性词（例如青苹果不要扩展红苹果）。"
        "不要引入无关类别名，不要输出场景词。"
        "输出仅为逗号分隔短语，不要解释，不要编号。"
        "每个短语尽量短（2-8词），允许中文描述。"
        f"总条数不超过 {max_terms}。"
    )

    user_prompt = (
        f"任务领域: {domain_hint or 'general'}\n"
        f"种子提示词: {', '.join(seed_prompts)}\n"
        "请返回分离式扩词列表。示例: 青苹果, apple, green, spherical, 少量斑点, 表皮光滑。"
        "若种子词包含俯视/顶视/航拍等视角信息，必须输出不少于8条视角相关可检测短语。"
        "重点是提高可检测性，而不是写自然语言长句。"
        "优先输出可直接被检测的实体词与外观词，不要为凑数量而补齐。"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=f"{api_base}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=request_timeout) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM 请求失败: HTTP {e.code} {body}") from e
    except Exception as e:
        raise RuntimeError(f"LLM 请求失败: {e}") from e

    try:
        obj = json.loads(body)
        content = obj["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM 返回解析失败: {e}; 原始响应: {body[:500]}") from e

    llm_terms = split_terms(content)
    merged = dedupe_keep_order(list(seed_prompts) + llm_terms)
    return merged[:max_terms]


def ollama_expand(
    seed_prompts,
    domain_hint="",
    model="qwen2.5:7b-instruct",
    api_base=None,
    max_terms=200,
    temperature=0.1,
    request_timeout=60,
):
    api_base = (api_base or os.getenv("LLM_API_BASE") or "http://127.0.0.1:11434").rstrip("/")

    system_prompt = (
        "你是视觉检测提示词工程专家。"
        "当前任务用于 YOLOE-26x 开放词表提示词，不是自然语言描述或caption。"
        "先在内部思考目标的可见样貌特征（颜色、形状、纹理、边界、常见缺陷），再输出拟合检测短语。"
        "不要做跨类别同义词扩展。"
        "请按实体词与样貌特征词分离输出，而不是实体+特征合并短语。"
        "实体词和特征词都用短词组，便于开放词表独立匹配。"
        "若目标与相近类别容易混淆（例如 SUV 与轿车），必须输出可区分特征词。"
        "无论类别是否有专用模板，输出词必须与种子词保持强语义锚定。"
        "不要引入与种子词冲突的属性词（例如青苹果不要扩展红苹果）。"
        "不要引入无关类别名，不要输出场景词。"
        "输出仅为逗号分隔短语，不要解释，不要编号。"
        "每个短语尽量短（2-8词），允许中文描述。"
        f"总条数不超过 {max_terms}。"
    )

    user_prompt = (
        f"任务领域: {domain_hint or 'general'}\n"
        f"种子提示词: {', '.join(seed_prompts)}\n"
        "请返回分离式扩词列表。示例: 青苹果, apple, green, spherical, 少量斑点, 表皮光滑。"
        "若种子词包含俯视/顶视/航拍等视角信息，必须输出不少于8条视角相关可检测短语。"
        "重点是提高可检测性，而不是写自然语言长句。"
        "优先输出可直接被检测的实体词与外观词，不要为凑数量而补齐。"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=f"{api_base}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=request_timeout) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama 请求失败: HTTP {e.code} {body}") from e
    except Exception as e:
        raise RuntimeError(f"Ollama 请求失败: {e}") from e

    try:
        obj = json.loads(body)
        content = obj["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Ollama 返回解析失败: {e}; 原始响应: {body[:500]}") from e

    llm_terms = split_terms(content)
    merged = dedupe_keep_order(list(seed_prompts) + llm_terms)
    return merged[:max_terms]


def expand_prompts(
    seed_prompts,
    enable_llm=False,
    llm_provider="ollama",
    llm_model="qwen2.5:7b-instruct",
    llm_api_base=None,
    llm_api_key=None,
    llm_max_terms=200,
    llm_timeout=45,
    domain_hint="",
):
    base = dedupe_keep_order(seed_prompts)
    if not base:
        return [], {"mode": "empty", "added": 0, "error": None}

    # 先做稳定的规则扩展，保证离线可用
    expanded = heuristic_expand(base, domain_hint=domain_hint)
    expanded = dedupe_keep_order(expanded + build_visual_prototype_terms(base, domain_hint=domain_hint))
    expanded = refine_expanded_terms(base, expanded, llm_max_terms)
    info = {"mode": "heuristic", "added": max(0, len(expanded) - len(base)), "error": None}

    if not enable_llm:
        return expanded, info

    try:
        llm_seed_prompts = _prefer_english_terms(expanded, base, max_terms=min(llm_max_terms, 120))
        if llm_provider == "openai-compatible":
            llm_expanded = openai_compatible_expand(
                seed_prompts=llm_seed_prompts,
                domain_hint=domain_hint,
                model=llm_model,
                api_base=llm_api_base,
                api_key=llm_api_key,
                max_terms=llm_max_terms,
                request_timeout=llm_timeout,
            )
        elif llm_provider == "ollama":
            llm_expanded = ollama_expand(
                seed_prompts=llm_seed_prompts,
                domain_hint=domain_hint,
                model=llm_model,
                api_base=llm_api_base,
                max_terms=llm_max_terms,
                request_timeout=max(llm_timeout, 30),
            )
        else:
            raise RuntimeError(f"不支持的 llm_provider: {llm_provider}")

        llm_expanded = refine_expanded_terms(base, llm_expanded, llm_max_terms)
        info = {
            "mode": "heuristic+llm",
            "added": max(0, len(llm_expanded) - len(base)),
            "error": None,
        }
        return llm_expanded, info
    except Exception as e:
        info = {
            "mode": "heuristic_fallback",
            "added": max(0, len(expanded) - len(base)),
            "error": str(e),
        }
        return expanded, info


def _extract_first_json_object(text):
    s = str(text or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        frag = s[start : end + 1]
        try:
            return json.loads(frag)
        except Exception:
            return None
    return None


def suggest_annotation_setup_from_description(
    user_description,
    llm_provider="openai-compatible",
    llm_model="deepseek-chat",
    llm_api_base=None,
    llm_api_key=None,
    llm_timeout=45,
):
    system_prompt = (
        "你是目标检测与分割参数助手。"
        "请根据用户描述生成可直接用于YOLO标注的建议。"
        "必须只返回JSON，不要返回markdown。"
        "JSON字段固定为: "
        "need_more_desc(boolean), "
        "advice(string), "
        "prompt_text(string, 逗号分隔), "
        "negative_prompt_text(string, 逗号分隔), "
        "conf(number, 0.05-0.95), "
        "iou(number, 0.1-0.95), "
        "imgsz(integer, 320-2048)."
        "若描述信息不足，need_more_desc=true，并在advice中给出该怎么补充描述；"
        "此时也应给出保守默认参数。"
        "若信息充足，need_more_desc=false，advice简洁说明原因。"
    )
    user_prompt = (
        "用户原始描述如下:\n"
        f"{str(user_description).strip()}\n"
        "请严格返回JSON对象。"
    )

    if llm_provider == "openai-compatible":
        api_base = (llm_api_base or os.getenv("LLM_API_BASE") or "https://api.openai.com/v1").rstrip("/")
        api_key = llm_api_key or os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 LLM API Key，请填写接口密钥")

        payload = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
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
            with request.urlopen(req, timeout=max(10, int(llm_timeout))) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM 请求失败: HTTP {e.code} {body}") from e
        except Exception as e:
            raise RuntimeError(f"LLM 请求失败: {e}") from e

        try:
            obj = json.loads(body)
            content = obj["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"LLM 返回解析失败: {e}") from e
    elif llm_provider == "ollama":
        api_base = (llm_api_base or os.getenv("LLM_API_BASE") or "http://127.0.0.1:11434").rstrip("/")
        payload = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        req = request.Request(
            url=f"{api_base}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=max(10, int(llm_timeout))) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama 请求失败: HTTP {e.code} {body}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama 请求失败: {e}") from e

        try:
            obj = json.loads(body)
            content = obj["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama 返回解析失败: {e}") from e
    else:
        raise RuntimeError(f"不支持的 llm_provider: {llm_provider}")

    parsed = _extract_first_json_object(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM 未返回有效JSON")

    need_more_desc = bool(parsed.get("need_more_desc", False))
    advice = str(parsed.get("advice", "")).strip()
    prompt_text = ", ".join(split_terms(str(parsed.get("prompt_text", ""))))
    negative_prompt_text = ", ".join(split_terms(str(parsed.get("negative_prompt_text", ""))))

    def _clamp_float(v, lo, hi, default):
        try:
            x = float(v)
        except Exception:
            x = float(default)
        return max(lo, min(hi, x))

    def _clamp_int(v, lo, hi, default):
        try:
            x = int(v)
        except Exception:
            x = int(default)
        return max(lo, min(hi, x))

    conf = _clamp_float(parsed.get("conf", 0.30), 0.05, 0.95, 0.30)
    iou = _clamp_float(parsed.get("iou", 0.60), 0.10, 0.95, 0.60)
    imgsz = _clamp_int(parsed.get("imgsz", 1024), 320, 2048, 1024)

    return {
        "need_more_desc": need_more_desc,
        "advice": advice,
        "prompt_text": prompt_text,
        "negative_prompt_text": negative_prompt_text,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "raw": parsed,
    }
