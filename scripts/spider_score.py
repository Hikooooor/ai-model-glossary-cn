"""Heuristic scoring and tier assignment for candidate papers."""

from spider_config import PRACTITIONER_KEYWORDS, TOP_CONF_KEYWORDS, TOP_ORGS


def score_article(article, hf_upvotes_map, seen_urls):
    """Score a paper using multi-signal heuristics and return (score, signals)."""
    url = article.get("url", "")
    title = article.get("title", "")
    abstract = article.get("raw_text", "")
    text = (title + " " + abstract).lower()
    authors_line = article.get("vendor", "").lower()

    # Hard dedup: skip URLs that appeared recently.
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


def select_top_articles(articles, hf_upvotes_map, seen_urls, max_results):
    """Rank candidates, drop deduped items, and label featured/notable tiers."""
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
