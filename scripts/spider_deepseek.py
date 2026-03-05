import json
import re
import time
import urllib.request
from datetime import datetime, timezone

from spider_config import DEEPSEEK_API_KEY

_FEW_SHOT_EXAMPLE = """
示例输入：
【标题】：LoRA: Low-Rank Adaptation of Large Language Models
【摘要】：We propose LoRA, which freezes pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.

示例输出（JSON）：
{
  "concept_name": "LoRA（低秩适配微调）",
  "tag": "大模型微调",
  "one_sentence_desc": "LoRA在冻结主干参数的前提下，用低秩矩阵完成低成本微调，让团队在有限算力下更快迭代场景模型。",
  "deep_analysis": "这项工作的**核心改动**是在不更新全部参数的情况下完成有效适配，从而同时缓解训练成本和部署复杂度；据摘要与实验线索，它在保持接近效果的同时显著降低可训练参数规模，对工程侧最直接的**落地价值**是缩短迭代周期并降低显存门槛，但上线前仍需结合你的业务评测集验证泛化边界，尤其关注不同任务与超参数设置下的稳定性。"
}
"""


def _extract_key_facts(article, max_items=3):
    text = f"{article.get('title', '')} {article.get('raw_text', '')}"
    number_hits = re.findall(r"\b\d+(?:\.\d+)?(?:%|x|倍|ms|FPS|B|M|K)?\b", text, flags=re.IGNORECASE)
    benchmark_keywords = [
        "MMLU", "GSM8K", "HumanEval", "MATH", "MMMU", "AIME", "GPQA",
        "ImageNet", "COCO", "WebArena", "FID", "mAP", "BLEU"
    ]
    benchmark_hits = [kw for kw in benchmark_keywords if kw.lower() in text.lower()]

    lines = []
    lines.extend([f"关键数字{item}" for item in number_hits[:max_items]])
    lines.extend([f"评测{item}" for item in benchmark_hits[:max_items]])

    uniq = []
    for line in lines:
        if line not in uniq:
            uniq.append(line)
    return uniq[:max_items]


def _sanitize_text(text):
    content = str(text or "").strip().replace("###", "")
    content = re.sub(r"[\r\n]+", " ", content)
    content = re.sub(r"\s{2,}", " ", content)
    for noisy_word in ["革命性", "颠覆性", "史诗级", "爆炸性"]:
        content = content.replace(noisy_word, "显著")
    return content


def _ensure_single_paragraph(text):
    cleaned = _sanitize_text(text)
    cleaned = re.sub(r"\s*[•\-]\s*", " ", cleaned)
    return cleaned.strip()


def _count_bold_segments(text):
    return len(re.findall(r"\*\*[^*]+\*\*", text or ""))


def _ensure_emphasis(text, concept_name, facts=None):
    result = text
    if _count_bold_segments(result) == 0 and concept_name:
        key = concept_name[:18]
        if key and key in result:
            result = result.replace(key, f"**{key}**", 1)

    if facts:
        for fact in facts[:2]:
            fact_key = fact[:12]
            if fact_key and fact_key in result and f"**{fact_key}**" not in result:
                result = result.replace(fact_key, f"**{fact_key}**", 1)

    for marker in ["核心改动", "落地价值", "上线建议"]:
        if marker in result and f"**{marker}**" not in result:
            result = result.replace(marker, f"**{marker}**", 1)

    if _count_bold_segments(result) < 3:
        for marker in ["方法", "价值", "风险"]:
            if marker in result and f"**{marker}**" not in result:
                result = result.replace(marker, f"**{marker}**", 1)
                if _count_bold_segments(result) >= 3:
                    break

    if _count_bold_segments(result) < 3:
        for marker in ["痛点", "效率", "成本", "效果", "泛化"]:
            if marker in result and f"**{marker}**" not in result:
                result = result.replace(marker, f"**{marker}**", 1)
                if _count_bold_segments(result) >= 3:
                    break

    if _count_bold_segments(result) < 3:
        result = f"**核心结论**：{result}"

    if _count_bold_segments(result) < 3:
        fallback_marks = ["**方法可行性**", "**业务价值**", "**上线风险**"]
        missing = 3 - _count_bold_segments(result)
        result = result.rstrip("。") + "，" + "，".join(fallback_marks[:missing]) + "。"
    return result


def _fallback_paragraph(article, concept):
    summary = article.get("raw_text", "")
    return (
        f"这篇论文围绕**{concept}**提出方法改进，主要通过模型结构、训练策略或数据流程优化来改善效果与效率；"
        f"据摘要可见其关注点在工程可用性与泛化表现的平衡，核心线索包括“{summary[:120]}...”，"
        "对团队的**落地价值**在于降低试错成本并提升迭代速度，但在正式上线前仍建议基于你的业务数据进行小流量灰度验证。"
    )


def _enforce_quality(parsed, article):
    title = article.get("title", "")
    concept_name = _sanitize_text(parsed.get("concept_name") or "") or (title.split(":")[0][:80] if ":" in title else title[:80] or "Unknown Concept")
    tag = _sanitize_text(parsed.get("tag") or "") or "论文速览"
    one_sentence_desc = _sanitize_text(parsed.get("one_sentence_desc") or "")
    if not one_sentence_desc:
        one_sentence_desc = f"该论文围绕{concept_name}给出新方案，重点改善工程效果、效率与可落地性平衡。"

    deep_analysis = _ensure_single_paragraph(parsed.get("deep_analysis") or "")
    if len(deep_analysis) < 160:
        deep_analysis = _fallback_paragraph(article, concept_name)

    facts = _extract_key_facts(article)
    if facts:
        deep_analysis = deep_analysis.rstrip("。") + f"，可验证线索包括{'、'.join(facts)}。"

    deep_analysis = _ensure_single_paragraph(deep_analysis)
    deep_analysis = _ensure_emphasis(deep_analysis, concept_name, facts)

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
        "signals": article.get("signals", []),
    }


def build_fallback_insight(article):
    title = article.get("title", "Unknown Paper")
    concept = title.split(":")[0][:80] if ":" in title else title[:80]
    payload = {
        "concept_name": concept,
        "tag": "论文速览",
        "one_sentence_desc": f"该论文围绕{concept}提出改进方案，聚焦效果、效率与可部署性的平衡。",
        "deep_analysis": _fallback_paragraph(article, concept),
    }
    return _enforce_quality(payload, article)


def analyze_with_deepseek(article, max_retry=3):
    if not DEEPSEEK_API_KEY:
        print("  Warning: 未配置 DEEPSEEK_API_KEY，启用降级摘要生成。")
        return build_fallback_insight(article)

    hints = []
    if article.get("signals"):
        hints.append("筛选信号：" + ", ".join(article.get("signals", [])[:5]))
    facts = _extract_key_facts(article)
    if facts:
        hints.append("可验证线索：" + "；".join(facts))
    hint_text = "\n".join(hints) if hints else "可验证线索：据摘要提炼"

    prompt = f"""你是一位服务AI行业从业者（产品经理、工程师、创业者）的技术观察员。请阅读以下Arxiv论文摘要，提炼最核心的技术概念。

输出要求（必须遵守）：
1. 仅输出 JSON，字段仅包含 concept_name、tag、one_sentence_desc、deep_analysis。
2. deep_analysis 必须是单段落，不允许分点、不允许换行、不允许列表符号。
3. deep_analysis 字数建议 220~320 字，内容包含：痛点、方法、价值、上线建议，但必须写成自然段。
4. 对关键内容使用 Markdown 粗体（**关键词**）标记，至少 3 处。
5. 不夸张，不写“革命性”等词；不确定信息标注“据摘要”。

{_FEW_SHOT_EXAMPLE}
现在请处理：
【标题】：{article.get('title', '')}
【摘要】：{article.get('raw_text', '')[:1400]}
【补充线索】：{hint_text}
"""

    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "你是一个严格的 JSON 生成器，只输出合法 JSON，不输出 Markdown 代码块。"
            },
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.35,
        "top_p": 0.9,
        "max_tokens": 1300,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }

    for attempt in range(max_retry):
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            json.dumps(data).encode("utf-8"),
            headers,
            method="POST",
        )
        try:
            response = urllib.request.urlopen(req, timeout=30)
            result = json.loads(response.read().decode("utf-8"))
            raw_content = result["choices"][0]["message"]["content"].strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
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
