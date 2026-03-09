"""DeepSeek analysis adapter with strict JSON output and local quality fallback."""

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
    "deep_analysis": "核心结论是，LoRA通过在每层Transformer中插入低秩适配矩阵，在冻结主干参数的前提下完成任务适配；据摘要，可训练参数量最高可比全参微调减少10,000x，同时保持接近的任务效果。它的直接价值在于显著降低显存与训练成本，让中小团队更容易在有限GPU资源下快速迭代下游场景。落地时需要重点验证低秩维度r的取值与任务复杂度是否匹配，因为秩设置过小可能削弱泛化能力。"
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
    """Normalize whitespace and remove sensational wording for stable display text."""
    content = str(text or "").strip().replace("###", "")
    content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)
    content = re.sub(r"[\r\n]+", " ", content)
    content = re.sub(r"\s{2,}", " ", content)
    for noisy_word in ["革命性", "颠覆性", "史诗级", "爆炸性"]:
        content = content.replace(noisy_word, "显著")
    return content


def _ensure_single_paragraph(text):
    """Collapse generated content into one paragraph required by frontend layout."""
    cleaned = _sanitize_text(text)
    cleaned = re.sub(r"\s*[•\-]\s*", " ", cleaned)
    return cleaned.strip()


def _refine_keypoint_expression(text):
    """Rewrite common weak phrases to make conclusion/value/risk points more explicit."""
    result = str(text or "")
    replacements = [
        ("这篇论文围绕", "核心结论是，该工作围绕"),
        ("主要通过", "关键做法是通过"),
        ("据摘要可见", "据摘要，"),
        ("其关注点在", "重点在于"),
        ("对团队的", "对业务侧的"),
        ("但在正式上线前仍建议", "上线前建议"),
    ]
    for old, new in replacements:
        if old in result:
            result = result.replace(old, new, 1)

    if "上线建议" not in result and "建议" in result:
        result = result.replace("建议", "上线建议是", 1)

    if "落地价值" not in result and "价值" in result:
        result = result.replace("价值", "落地价值", 1)

    return result


def _ensure_analysis_structure(text, concept_name, facts=None):
    result = _sanitize_text(text)

    if concept_name and concept_name not in result:
        result = f"{concept_name}的核心结论是，{result}"

    if facts:
        fact = facts[0]
        if fact and fact not in result:
            result = result.rstrip("。") + f" 据摘要，文中还给出了{fact}这一关键信号。"

    if not re.search(r"落地|部署|工程|业务|产品", result):
        result = result.rstrip("。") + " 对工程团队而言，它更适合用于降低试错成本、压缩验证周期，帮助判断是否值得进入原型验证。"

    if not re.search(r"风险|建议|边界|前提|适用", result):
        result = result.rstrip("。") + " 上线前仍需结合你的业务数据验证适用边界，重点检查数据分布、算力开销与泛化稳定性。"

    return _ensure_single_paragraph(result)


def _fallback_paragraph(article, concept):
    summary = article.get("raw_text", "")
    facts = _extract_key_facts(article)
    fact_clause = f"据摘要，文中给出的关键信号包括{facts[0]}，" if facts else "据摘要，当前公开信息更适合从方法机制与适用边界来理解，"
    return (
        f"这篇论文围绕{concept}提出方法改进，关键做法是通过模型结构、训练策略或数据流程优化来改善效果与效率；"
        f"{fact_clause}核心线索包括“{summary[:120]}...”，重点仍在工程可用性与泛化表现之间的平衡；"
        "它对团队的直接价值是帮助更快验证方案是否具备上线潜力，但正式投入业务前仍需基于真实数据做小流量验证，重点检查收益是否能覆盖额外复杂度。"
    )


def _enforce_quality(parsed, article):
    """Fill missing fields and enforce minimum readability constraints."""
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
    deep_analysis = _ensure_analysis_structure(deep_analysis, concept_name, facts)

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
    """Build deterministic fallback when API key is missing or calls fail."""
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
    """Call DeepSeek Chat Completions and return normalized radar record."""
    if not DEEPSEEK_API_KEY:
        print("  Warning: 未配置 DEEPSEEK_API_KEY，启用降级摘要生成。")
        return build_fallback_insight(article)

    hints = []
    if article.get("signals"):
        hints.append("筛选信号：" + ", ".join(article.get("signals", [])[:5]))
    facts = _extract_key_facts(article)
    if facts:
        hints.append(
            "摘要中出现的关键数据点（请在 deep_analysis 中自然引用，不要使用 Markdown 标记）：" + "、".join(facts)
        )
    hint_text = "\n".join(hints) if hints else "（无特殊补充）"

    prompt = f"""你是一位服务AI行业从业者（产品经理、工程师、创业者）的技术观察员。请阅读以下Arxiv论文摘要，提炼最核心的技术概念。

输出要求（必须遵守）：
1. 仅输出 JSON，字段仅包含 concept_name、tag、one_sentence_desc、deep_analysis。
2. tag 必须从以下分类中选最匹配的一个：{_TAG_TAXONOMY}
3. concept_name 优先提炼成可读的技术名词，不要只截标题前几个单词；如果英文术语更通用，可保留英文并补一个中文解释。
4. one_sentence_desc 控制在 45~80 字，必须同时交代“做了什么”和“对谁有价值”，不要复读标题。
5. deep_analysis 必须是单段落，不允许分点、不允许换行、不允许列表符号、不允许 Markdown 粗体或其他格式标记。
6. deep_analysis 字数建议 220~320 字，按顺序自然展开三层信息：先讲核心改动与证据，再讲对工程/产品团队的直接价值，最后讲适用前提、风险或上线建议。
7. 若摘要里有数字、指标、数据集、速度/成本变化，请至少引用 1 个具体事实；若没有明确量化结果，就明确说明“据摘要”只能确认方法机制与适用场景。
8. 语言要像给从业者写晨报：信息密度高、判断克制、少空话，避免“革命性”“颠覆性”这类夸张表述，也不要照抄摘要原句。

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

    # Retry with exponential backoff to tolerate transient network/API failures.
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
