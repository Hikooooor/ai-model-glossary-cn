"""Source fetchers for HuggingFace Daily Papers and Arxiv Atom feed."""

import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from spider_config import HF_DAYS_BACK, MAX_RESULTS, RECENT_DAYS


def fetch_hf_daily_papers(days_back=HF_DAYS_BACK):
    """Return {arxiv_id: upvotes} for a target HF Daily Papers date."""
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
    """Fetch recent Arxiv entries and keep a recency flag for later scoring/selecting."""
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
        root = ET.fromstring(response.read())
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
            articles.append(
                {
                    "vendor": f"Arxiv: {vendor} et al.",
                    "url": link,
                    "title": title,
                    "date": published,
                    "raw_text": summary,
                    "is_recent": published_date >= cutoff,
                }
            )

        # Prefer recent papers; fall back to full pool if recent count is insufficient.
        recent = [a for a in articles if a["is_recent"]]
        pool = recent if len(recent) >= max_results else articles
        print(f"  Arxiv: 获取到 {len(articles)} 篇，其中近{days}天 {len(recent)} 篇，候选池 {len(pool)} 篇")
        return pool
    except Exception as e:
        print(f"抓取 Arxiv 数据失败: {e}")
        return []
