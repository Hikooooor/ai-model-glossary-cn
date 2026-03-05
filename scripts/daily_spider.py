import os
import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse
import re

# ==========================================
# 多源采集 + 多信号评分 + 分层输出
# 数据源：Arxiv (6个核心AI类目) + HuggingFace Daily Papers
# ==========================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
MAX_RESULTS = 6
RECENT_DAYS = 10
HF_DAYS_BACK = 3          # T+3：采集3天前的HF论文，让社区投票充分沉淀
HISTORY_FILE = "data/radar-history.json"
HISTORY_INDEX_FILE = "data/history-index.json"
HISTORY_KEEP_DAYS = 180
MAX_WORKERS = 6            # DeepSeek 并发请求数
DEDUP_DAYS = 7             # 跨天去重窗口

# 从业者相关关键词（标题/摘要含此类词加分）
PRACTITIONER_KEYWORDS = [
    "deploy", "inference", "agent", "rag", "retrieval", "fine-tun", "finetun",
    "lora", "quantiz", "distill", "efficient", "cost", "latency", "throughput",
    "tool use", "benchmark", "multimodal", "multi-modal", "reasoning", "alignment",
    "推理", "部署", "微调", "多模态", "智能体"
]

# 顶会关键词
TOP_CONF_KEYWORDS = [
    "ICLR", "NeurIPS", "ICML", "ACL", "EMNLP", "CVPR", "ICCV", "ECCV", "NAACL"
]

# 顶级机构关键词
TOP_ORGS = [
    "google", "openai", "meta ", "microsoft", "anthropic", "deepmind", "apple",
    "tsinghua", "peking university", "stanford", "mit ", "cmu", "uc berkeley",
    "amazon", "nvidia", "samsung", "baidu", "alibaba", "tencent"
]


# ==========================================
# 数据结构规范化
# ==========================================

def normalize_record(item):
    """统一每日知识点字段结构，保证历史文件长期稳定可读。"""
    return {
        "vendor": item.get("vendor", "Unknown"),
        "date": item.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
        "url": item.get("url", "https://arxiv.org"),
        "concept_name": item.get("concept_name", "Unknown Concept"),
        "tag": item.get("tag", "论文速览"),
        "one_sentence_desc": item.get("one_sentence_desc", "暂无描述"),
        "deep_analysis": item.get("deep_analysis", "暂无深度分析"),
        "tier": item.get("tier", "notable"),
        "score": item.get("score", 0),
        "signals": item.get("signals", [])
    }


# ==========================================
# 历史文件读写（含增量保护 + 月度分片）
# ==========================================

def load_history():
    """读取历史文件，兼容旧版本结构。"""
    if not os.path.exists(HISTORY_FILE):
        return {"schema_version": 2, "timezone": "UTC", "daily": {}}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "daily" in data:
            data.setdefault("schema_version", 2)
            data.setdefault("timezone", "UTC")
            return data
        if isinstance(data, dict):
            return {"schema_version": 2, "timezone": "UTC", "daily": data}
    except Exception:
        pass
    return {"schema_version": 2, "timezone": "UTC", "daily": {}}


def get_recent_urls(history, days=DEDUP_DAYS):
    """获取近 N 天已收录的论文 URL 集合，用于跨天去重。"""
    daily = history.get("daily", {})
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    seen_urls = set()
    for day, records in daily.items():
        try:
            if datetime.strptime(day, "%Y-%m-%d").date() >= cutoff:
                for r in records:
                    url = r.get("url", "")
                    if url:
                        seen_urls.add(url)
        except ValueError:
            continue
    return seen_urls


def prune_history(history, keep_days=HISTORY_KEEP_DAYS):
    """仅保留最近 keep_days 天，防止仓库无限膨胀。"""
    daily = history.get("daily", {})
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=keep_days)
    valid_daily = {
        day: records for day, records in daily.items()
        if datetime.strptime(day, "%Y-%m-%d").date() >= cutoff
    }
    history["daily"] = dict(sorted(valid_daily.items()))
    return history


def save_monthly_file(today, today_records):
    """
    将今日数据写入月度文件 data/YYYY-MM.json。
    增量合并：同一天已有数据时按 URL 去重追加，不覆盖。
    返回月份字符串，如 "2026-03"。
    """
    month_key = today[:7]
    month_file = f"data/{month_key}.json"

    month_data = {}
    if os.path.exists(month_file):
        try:
            with open(month_file, "r", encoding="utf-8") as f:
                month_data = json.load(f)
        except Exception:
            month_data = {}

    if today not in month_data:
        month_data[today] = today_records
    else:
        existing_urls = {r.get("url") for r in month_data[today]}
        new = [r for r in today_records if r.get("url") not in existing_urls]
        month_data[today] = month_data[today] + new

    month_data = dict(sorted(month_data.items()))
    with open(month_file, "w", encoding="utf-8") as f:
        json.dump(month_data, f, ensure_ascii=False, indent=2)
    return month_key


def update_history_index(month_key):
    """更新 history-index.json，记录所有可用月份供前端按需加载。"""
    index_data = {"months": []}
    if os.path.exists(HISTORY_INDEX_FILE):
        try:
            with open(HISTORY_INDEX_FILE, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        except Exception:
            pass

    months = set(index_data.get("months", []))
    months.add(month_key)
    index_data["months"] = sorted(months, reverse=True)
    index_data["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    with open(HISTORY_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)


def save_today_history(radar_data):
    """
    保存今日数据到：
    1. data/radar-history.json（主历史文件，兼容旧前端）
    2. data/YYYY-MM.json（月度分片文件，供新前端按需加载）
    3. data/history-index.json（月份索引）
    增量写入保护：今天已有数据时合并去重，而非覆盖。
    """
    os.makedirs("data", exist_ok=True)
    history = load_history()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_records = [normalize_record(item) for item in radar_data][:MAX_RESULTS]

    if today in history["daily"]:
        existing_urls = {r.get("url") for r in history["daily"][today]}
        new_records = [r for r in today_records if r.get("url") not in existing_urls]
        history["daily"][today] = history["daily"][today] + new_records
        print(f"今日数据已存在，增量追加 {len(new_records)} 条（合并去重）。")
    else:
        history["daily"][today] = today_records

    history = prune_history(history, HISTORY_KEEP_DAYS)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    month_key = save_monthly_file(today, history["daily"][today])
    update_history_index(month_key)


# ==========================================
# 数据采集
# ==========================================

def fetch_hf_daily_papers(days_back=HF_DAYS_BACK):
    """
    抓取 HuggingFace Daily Papers（T+3 策略）。
    返回 {arxiv_id: upvotes} 映射，用于评分信号。
    """
    hf_map = {}
    target_date = datetime.now(timezone.utc).date() - timedelta(days=days_back)
    date_str = target_date.strftime("%Y-%m-%d")
    url = f"https://huggingface.co/api/daily_papers?date={date_str}"
    print(f"正在从 HuggingFace Daily Papers 抓取 {date_str} 的推荐论文...")
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    )
    try:
        response = urllib.request.urlopen(req, timeout=15)
        papers = json.loads(response.read().decode("utf-8"))
        for entry in papers:
            paper = entry.get("paper", {})
            arxiv_id = paper.get("id", "")
            upvotes = entry.get("upvotes", 0)
            if arxiv_id:
                hf_map[arxiv_id] = upvotes
        print(f"  HuggingFace: 获取到 {len(hf_map)} 篇推荐论文")
    except Exception as e:
        print(f"  HuggingFace 抓取失败（不影响主流程）: {e}")
    return hf_map


def fetch_recent_ai_papers(max_results=MAX_RESULTS, days=RECENT_DAYS):
    """
    通过 Arxiv API 拉取最新论文（扩展至 6 个核心 AI 分类）。
    拉取候选池比最终所需大 8 倍，保证评分有充足论文可筛选。
    """
    print(f"正在从 Arxiv 抓取最新论文（6个核心类目，过去 {days} 天优先）...")
    fetch_size = max(40, max_results * 8)
    categories = "cat:cs.AI+OR+cat:cs.CL+OR+cat:cs.LG+OR+cat:cs.CV+OR+cat:cs.MA+OR+cat:cs.IR"
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={categories}"
        "&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={fetch_size}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        response = urllib.request.urlopen(req, timeout=30)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        articles = []
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
        for entry in entries:
            title = entry.find("atom:title", ns).text.replace("\n", " ").strip()
            summary = entry.find("atom:summary", ns).text.replace("\n", " ").strip()
            published = entry.find("atom:published", ns).text[:10]
            link = entry.find("atom:id", ns).text
            published_date = datetime.strptime(published, "%Y-%m-%d").date()
            authors = entry.findall("atom:author/atom:name", ns)
            vendor = authors[0].text if authors else "Arxiv Researcher"
            articles.append({
                "vendor": f"Arxiv: {vendor} et al.",
                "url": link,
                "title": title,
                "date": published,
                "raw_text": summary,
                "is_recent": published_date >= cutoff
            })

        recent = [a for a in articles if a["is_recent"]]
        pool = recent if len(recent) >= max_results else articles
        print(f"  Arxiv: 获取到 {len(articles)} 篇，其中近{days}天 {len(recent)} 篇，候选池 {len(pool)} 篇")
        return pool
    except Exception as e:
        print(f"抓取 Arxiv 数据失败: {e}")
        return []


# ==========================================
# 多信号评分 + 分层
# ==========================================

def score_article(article, hf_upvotes_map, seen_urls):
    """
    多信号评分，返回 (score, signals列表)。
    score=-1 表示该论文命中去重规则，应被剔除。

    信号权重：
    - 跨天去重（近7天出现过） 直接剔除 (score=-1)
    - HF推荐                  +3
    - HF热度（50+赞）         +3 / （20+）+2 / （5+）+1
    - 有源码（含github链接）  +1
    - 从业者相关关键词        +1
    - 顶会收录                +1
    - 顶级机构                +1
    """
    url = article.get("url", "")
    title = article.get("title", "")
    abstract = article.get("raw_text", "")
    text = (title + " " + abstract).lower()
    authors_line = article.get("vendor", "").lower()

    if url in seen_urls:
        return -1, []

    score = 0
    signals = []

    arxiv_id = url.split("/abs/")[-1].split("v")[0] if "/abs/" in url else ""

    if arxiv_id and arxiv_id in hf_upvotes_map:
        upvotes = hf_upvotes_map[arxiv_id]
        score += 3
        signals.append("HF推荐")
        if upvotes >= 50:
            score += 3
            signals.append(f"HF热度({upvotes}赞)")
        elif upvotes >= 20:
            score += 2
            signals.append(f"HF热度({upvotes}赞)")
        elif upvotes >= 5:
            score += 1
            signals.append(f"HF热度({upvotes}赞)")

    if "github.com" in text:
        score += 1
        signals.append("有源码")

    for kw in PRACTITIONER_KEYWORDS:
        if kw.lower() in text:
            score += 1
            signals.append("从业者相关")
            break

    for conf in TOP_CONF_KEYWORDS:
        if conf.lower() in text:
            score += 1
            signals.append(f"顶会({conf})")
            break

    for org in TOP_ORGS:
        if org.lower() in authors_line or org.lower() in text[:300]:
            score += 1
            signals.append("顶级机构")
            break

    return score, signals


def select_top_articles(articles, hf_upvotes_map, seen_urls, max_results=MAX_RESULTS):
    """
    对候选池进行多信号评分、去重、排序，返回评分最高的 max_results 篇。
    附加 score / signals / tier 字段，前3为 featured，其余为 notable。
    """
    scored = []
    deduped_count = 0
    for article in articles:
        s, sigs = score_article(article, hf_upvotes_map, seen_urls)
        if s < 0:
            deduped_count += 1
            continue
        article["score"] = s
        article["signals"] = sigs
        scored.append(article)

    scored.sort(key=lambda x: x["score"], reverse=True)
    selected = scored[:max_results]

    for i, art in enumerate(selected):
        art["tier"] = "featured" if i < 3 else "notable"
        art.pop("is_recent", None)

    print(f"  多信号评分：候选 {len(scored)} 篇（去重剔除 {deduped_count} 篇） 选出 {len(selected)} 篇")
    return selected


# ==========================================
# DeepSeek 提炼（含指数退避重试 + 并发）
# ==========================================

_FEW_SHOT_EXAMPLE = """
示例输入：
【标题】：LoRA: Low-Rank Adaptation of Large Language Models
【摘要】：We propose LoRA, which freezes pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.

示例输出（JSON）：
{
  "concept_name": "LoRA（低秩适配微调）",
  "tag": "大模型微调",
  "one_sentence_desc": "LoRA通过在Transformer层注入低秩可训练矩阵，在冻结主干参数的前提下实现低成本微调，使企业能够以更低算力完成场景化模型适配。",
  "deep_analysis": "痛点与背景：全参数微调需要更新全部权重，训练资源和显存占用高，迭代周期长。\n核心做法：LoRA仅训练低秩适配参数，将更新限制在少量新增矩阵，保持原模型主体不变。\n效果与价值：据摘要与公开实验，LoRA显著减少可训练参数规模，常见设置下仅占原模型极小比例，在接近原效果的同时降低训练成本与部署复杂度。\n边界与建议：该方法对任务类型和秩设置敏感，落地时建议结合业务数据做小规模网格搜索，并保留全参数微调作为关键任务对照基线。"
}
"""


def _ensure_rich_output(parsed, article):
    """补齐字段并兜底增强文案厚度，保证前端展示稳定。"""
    title = article.get("title", "")
    summary = article.get("raw_text", "")

    concept_name = str(parsed.get("concept_name") or "").strip() or (title.split(":")[0][:80] if ":" in title else title[:80] or "Unknown Concept")
    tag = str(parsed.get("tag") or "").strip() or "论文速览"
    one_sentence_desc = str(parsed.get("one_sentence_desc") or "").strip()
    if not one_sentence_desc:
        one_sentence_desc = f"该论文围绕“{concept_name}”提出方法改进，重点影响模型效果、效率或可落地性。"

    deep_analysis = str(parsed.get("deep_analysis") or "").strip()
    if len(deep_analysis) < 180:
        deep_analysis = (
            f"痛点与背景：该方向通常面临效果、成本或稳定性之间的权衡，传统方案在真实业务中存在实施门槛。\n"
            f"核心做法：据摘要，论文通过模型结构、训练策略或数据流程改造来提升表现，关键信息为：{summary[:180]}...\n"
            f"效果与价值：从业者价值在于可提升任务质量并降低试错成本，建议优先在小流量场景验证。\n"
            f"边界与建议：应重点评估泛化能力、推理成本和工程复杂度，并以原论文实验配置作为复现基准。"
        )

    return {
        "concept_name": concept_name,
        "tag": tag,
        "one_sentence_desc": one_sentence_desc,
        "deep_analysis": deep_analysis,
        "vendor": article.get("vendor", "Arxiv Researcher"),
        "date": article.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
        "url": article.get("url", "https://arxiv.org"),
        "tier": article.get("tier", "notable"),
        "score": article.get("score", 0),
        "signals": article.get("signals", [])
    }


def _extract_key_facts(article, max_items=3):
    """从标题/摘要中提取可验证线索（数字、倍数、基准关键词）。"""
    text = f"{article.get('title', '')} {article.get('raw_text', '')}"
    number_hits = re.findall(r"\b\d+(?:\.\d+)?(?:%|x|倍|ms|FPS|B|M|K)?\b", text, flags=re.IGNORECASE)

    benchmark_keywords = [
        "MMLU", "GSM8K", "HumanEval", "MATH", "MMMU", "AIME", "GPQA",
        "ImageNet", "COCO", "WebArena", "tau", "FID", "mAP", "BLEU"
    ]
    benchmark_hits = [kw for kw in benchmark_keywords if kw.lower() in text.lower()]

    facts = []
    for item in number_hits[:max_items]:
        facts.append(f"关键数字：{item}")
    for item in benchmark_hits[:max_items]:
        facts.append(f"评测线索：{item}")

    unique = []
    for item in facts:
        if item not in unique:
            unique.append(item)
    return unique[:max_items]


def _sanitize_generated_text(text):
    """清理模型输出中的无效修饰和多余格式。"""
    content = str(text or "").strip()
    content = content.replace("###", "").replace("**", "")
    for noisy_word in ["革命性", "颠覆性", "史诗级", "爆炸性"]:
        content = content.replace(noisy_word, "显著")
    return content


def _enforce_quality(parsed, article):
    """统一质量门控：事实线索、结构化深度解析、更强落地建议。"""
    normalized = _ensure_rich_output(parsed, article)

    normalized["concept_name"] = _sanitize_generated_text(normalized.get("concept_name", ""))
    normalized["tag"] = _sanitize_generated_text(normalized.get("tag", ""))
    normalized["one_sentence_desc"] = _sanitize_generated_text(normalized.get("one_sentence_desc", ""))
    deep_analysis = _sanitize_generated_text(normalized.get("deep_analysis", ""))

    has_structure = all(
        marker in deep_analysis
        for marker in ["痛点与背景", "核心做法", "效果与价值", "边界与建议"]
    )

    if not has_structure:
        deep_analysis = (
            f"痛点与背景：该方向常见瓶颈在效果、成本或可靠性之间难以兼顾。\n"
            f"核心做法：据摘要，论文主要通过结构、训练或数据策略改造实现提升。\n"
            f"效果与价值：对工程团队的核心价值是提升质量稳定性并降低试错成本。\n"
            f"边界与建议：建议先以离线评测+小流量灰度验证，不要直接全量上线。"
        )

    facts = _extract_key_facts(article)
    signals = article.get("signals", [])
    fact_lines = []
    if signals:
        fact_lines.append(f"筛选信号：{', '.join(signals[:4])}")
    fact_lines.extend(facts)
    if fact_lines:
        deep_analysis = deep_analysis.rstrip() + "\n可验证线索：" + "；".join(fact_lines)

    normalized["deep_analysis"] = deep_analysis
    return normalized


def build_fallback_insight(article):
    """DeepSeek 全部重试失败时的降级方案，保证前端有稳定输出。"""
    title = article.get("title", "Unknown Paper")
    summary = article.get("raw_text", "")
    concept = title.split(":")[0][:80] if ":" in title else title[:80]
    return _enforce_quality({
        "concept_name": concept,
        "tag": "论文速览",
        "one_sentence_desc": f"该论文围绕“{concept}”提出新方法，聚焦提升模型能力与推理/训练效率。",
        "deep_analysis": (
            f"痛点与背景：现有方案在效果、资源消耗或稳定性上存在瓶颈。\n"
            f"核心做法：据摘要，论文提出的关键改动为：{summary[:160]}...\n"
            f"效果与价值：该方案有望在真实业务中改善质量与成本平衡。\n"
            f"边界与建议：建议结合你的数据与评测集做小规模复现，再决定是否进入生产。"
        )
    }, article)


def analyze_with_deepseek(article, max_retry=3):
    """
    调用 DeepSeek API 提炼单篇论文。
    含指数退避重试（最多 max_retry 次），全部失败后降级。
    """
    if not DEEPSEEK_API_KEY:
        print("  Warning: 未配置 DEEPSEEK_API_KEY，启用降级摘要生成。")
        return build_fallback_insight(article)

    api_url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    quality_hints = []
    if article.get("signals"):
        quality_hints.append("筛选信号：" + ", ".join(article.get("signals", [])[:5]))
    facts = _extract_key_facts(article)
    if facts:
        quality_hints.append("可验证线索：" + "；".join(facts))
    quality_text = "\n".join(quality_hints) if quality_hints else "可验证线索：据摘要提炼"

    prompt = f"""你是一位服务AI行业从业者（产品经理、工程师、创业者）的技术观察员。请阅读以下Arxiv论文摘要，提取其中最核心的1个技术概念或创新点。

编辑原则（必须遵守）：
1. 先说清楚"这个方向原来有什么痛点"，再说"这篇论文怎么解决的"
2. 从业者视角：重点是"这对我有什么用/影响"，而不是堆砌学术术语
3. 保持克制：不使用"突破性""革命性"等夸张措辞；不确定之处标注"据摘要"
4. 具体量化：如果摘要有性能数字，请引用（如"比baseline快2倍"）
5. 输出要有信息密度：
   - concept_name：优先中文名，可附英文缩写
   - tag：必须具体，如“推理优化/多模态对齐/数据合成”
   - one_sentence_desc：40~90字，强调业务价值
   - deep_analysis：至少220字，且按以下四段组织（可用换行分隔）：
     痛点与背景 / 核心做法 / 效果与价值 / 边界与建议
6. 如果摘要未给出量化指标，明确写“据摘要未给出量化指标”
7. 禁止空话套话；每段至少包含一个可执行判断或验证建议
{_FEW_SHOT_EXAMPLE}
现在请处理以下论文：
【标题】：{article['title']}
【摘要】：{article['raw_text'][:1400]}
【补充线索】：{quality_text}

严格输出合法JSON，包含且仅包含：concept_name、tag、one_sentence_desc、deep_analysis。"""

    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨的JSON管道处理器，只输出合规JSON对象，不附加任何说明或Markdown符号。"
            },
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.35,
        "top_p": 0.9,
        "max_tokens": 1400
    }

    for attempt in range(max_retry):
        req = urllib.request.Request(
            api_url, json.dumps(data).encode("utf-8"), headers, method="POST"
        )
        try:
            response = urllib.request.urlopen(req, timeout=30)
            result = json.loads(response.read().decode("utf-8"))
            raw_content = result["choices"][0]["message"]["content"].strip()

            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]

            parsed = json.loads(raw_content)
            return _enforce_quality(parsed, article)

        except Exception as e:
            wait = 2 ** attempt
            print(f"  DeepSeek 第{attempt + 1}次失败: {e}，{wait}s 后重试...")
            if attempt < max_retry - 1:
                time.sleep(wait)

    print("  DeepSeek 全部重试失败，启用降级摘要。")
    return build_fallback_insight(article)


# ==========================================
# 主流程
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="AI 前沿雷达生成脚本")
    parser.add_argument("--dry-run", action="store_true", help="只打印结果，不写入任何文件（调试用）")
    args = parser.parse_args()

    print(" 开始执行 AI 前沿雷达任务（多源采集 + 多信号评分 + 并发提炼）...")
    if args.dry_run:
        print("  --dry-run 模式：结果只打印，不写文件")

    OUTPUT_FILE = "latest-radar.js"

    history = load_history()
    seen_urls = get_recent_urls(history, days=DEDUP_DAYS)
    print(f"近{DEDUP_DAYS}天已收录 URL：{len(seen_urls)} 篇（用于去重）")

    hf_upvotes_map = fetch_hf_daily_papers(days_back=HF_DAYS_BACK)
    arxiv_articles = fetch_recent_ai_papers(max_results=MAX_RESULTS, days=RECENT_DAYS)

    if not arxiv_articles:
        print("未获取到任何论文，请检查网络。")
        return

    selected = select_top_articles(arxiv_articles, hf_upvotes_map, seen_urls, max_results=MAX_RESULTS)
    if not selected:
        print("评分筛选后无可用论文（可能全部命中去重）。")
        return

    print(f"\n评分排行（Top {len(selected)}）：")
    for i, art in enumerate(selected, 1):
        print(f"  [{i}] 分={art['score']} tier={art['tier']} 信号={art['signals']}")
        print(f"       {art['title'][:60]}...")

    print(f"\n 并发提炼中（最多 {MAX_WORKERS} 并发）...")
    radar_data = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(analyze_with_deepseek, art): i
            for i, art in enumerate(selected)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            if result:
                radar_data[idx] = result
                print(f"   [{idx + 1}] {result.get('concept_name', '')[:35]}")

    radar_data = [normalize_record(r) for r in radar_data if r is not None]

    if not radar_data:
        print("未生成有效结果。")
        return

    if args.dry_run:
        print("\n[dry-run] 最终输出预览：")
        print(json.dumps(radar_data, ensure_ascii=False, indent=2))
        return

    js_content = (
        "// 本文件由 GitHub Actions 每日触发，通过 Arxiv + HuggingFace + DeepSeek 自动生成\n"
        f"// 生成时间(UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"window.dailyRadarData = {json.dumps(radar_data, ensure_ascii=False, indent=2)};\n"
    )
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(js_content)

    save_today_history(radar_data)

    featured = [r for r in radar_data if r.get("tier") == "featured"]
    notable = [r for r in radar_data if r.get("tier") == "notable"]
    month_key = datetime.now(timezone.utc).strftime("%Y-%m")
    print(
        f"\n 完成！共 {len(radar_data)} 条"
        f"（精选 {len(featured)}  值得关注 {len(notable)}）\n"
        f"   写入：{OUTPUT_FILE} | {HISTORY_FILE} | data/{month_key}.json | {HISTORY_INDEX_FILE}"
    )


if __name__ == "__main__":
    main()