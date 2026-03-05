import json
import re
import time
import urllib.request
from datetime import datetime, timezone

from spider_config import DEEPSEEK_API_KEY

_TAG_TAXONOMY = (
    "大模型微调 | 模型推理加速 | 多模态理解 | AI Agent | RAG/检索增强 | "
    "代码生成 | 图像/视频生成 | 模型压缩与量化 | 安全对齐 | 强化学习 | "
    "机器人与具身智能 | 医疗AI | 科学计算 | 数据合成 | 长上下文 | "
    "多智能体 | 基准评测 | 预训练方法 | 其他"
)

_FEW_SHOT_EXAMPLE = """
示例输入：
【标题】：LoRA: Low-Rank Adaptation of Large Language Models
【摘要】：We propose LoRA, which freezes pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to full fine-tuning, LoRA reduces trainable parameters by up to 10,000x while achieving comparable performance on GPT-3.

示例输出（JSON）：
{
  "concept_name": "LoRA（低秩适配微调）",
  "tag": "大模型微调",
  "one_sentence_desc": "LoRA在冻结主干参数的前提下，用低秩矩阵完成低成本微调，让团队在有限算力下更快迭代场景模型。",
  "deep_analysis": "**核心结论**：LoRA通过在每层Transformer中注入低秩分解矩阵，实现冻结主干参数的前提下高效适配，据摘要可训练参数量最高降低 **10,000倍**，同时保持与全参微调相当的效果；**落地价值**在于显著降低显存占用与训练成本，让中小团队在消费级GPU上完成业务迭代，缩短从实验到上线的周期；**上线建议**：正式使用前需在你的业务评测集上验证，重点关注低秩维度 r 的取值对任务泛化的影响，秩过小可能导致效果明显下降。"
}
"""


def _extract_key_facts(article, max_items=4):
    """Extract meaningful numbers (with units) and benchmark names from abstract."""
    text = f"{article.get('title', '')} {article.get('raw_text', '')}"
    # Only numbers with units are meaningful; bare 1/2/3 carry no context
    unit_numbers = re.findall(
        r"\b\d+(?:[,，]\d+)*(?:\.\d+)?(?:%|x|×|倍|ms|fps|gb|mb|[BbMmKk]\b)",
        text, flags=re.IGNORECASE
    )
    benchmark_keywords = [
        "MMLU", "GSM8K", "HumanEval", "MATH", "MMMU", "AIME", "GPQA",
        "ImageNet", "COCO", "WebArena", "FID", "mAP", "BLEU", "SWE-bench", "ROUGE"
    ]
    benchmark_hits = [kw for kw in benchmark_keywords if kw.lower() in text.lower()]

    facts = []
    seen = set()
    for item in (unit_numbers + benchmark_hits):
        key = item.lower()
        if key not in seen:
            seen.add(key)
            facts.append(item)
    return facts[:max_items]


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


def _refine_keypoint_expression(text):
    """Rewrite common weak phrases to make conclusion/value/risk points more explicit."""
    result = str(text or "")
    replacements = [
        ("这篇论文围绕", "**核心结论**：该工作围绕"),
        ("主要通过", "关键做法是通过"),
        ("据摘要可见", "据摘要，"),
        ("其关注点在", "重点在于"),
        ("对团队的", "对业务侧的"),
        ("但在正式上线前仍建议", "**上线建议**：正式上线前建议"),
    ]
    for old, new in replacements:
        if old in result:
            result = result.replace(old, new, 1)

    if "上线建议" not in result and "建议" in result:
        result = result.replace("建议", "**上线建议**：", 1)

    if "落地价值" not in result and "价值" in result:
        result = result.replace("价值", "**落地价值**", 1)

    return result


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

    deep_analysis = _refine_keypoint_expression(deep_analysis)
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
        hints.append(
            "摘要中出现的关键数据点（请在 deep_analysis 中自然引用并加粗）：" + "、".join(facts)
        )
    hint_text = "\n".join(hints) if hints else "（无特殊补充）"

    prompt = f"""你是一位服务AI行业从业者（产品经理、工程师、创业者）的技术观察员。请阅读以下Arxiv论文摘要，提炼最核心的技术概念。

输出要求（必须遵守）：
1. 仅输出 JSON，字段仅包含 concept_name、tag、one_sentence_desc、deep_analysis。
2. tag 必须从以下分类中选最匹配的一个：{_TAG_TAXONOMY}
3. deep_analysis 必须是单段落，不允许分点、不允许换行、不允许列表符号。
4. deep_analysis 字数建议 220~320 字，内容按顺序展开：①核心结论（方法做了什么、效果如何），②落地价值（对工程/产品团队的直接收益），③上线建议/风险提示（需注意什么），整体写成连贯自然段。
5. 粗体（**词语**）标记至少 3 处；优先对摘要中出现的具体数字、技术名称、评测指标加粗，而非泛化词汇。
6. 语言直接可执行，少形容词，每句有明确信息点；不写"革命性"等夸张词；不确定信息标注"据摘要"。

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
