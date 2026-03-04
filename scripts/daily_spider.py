import os
import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

# ==========================================
# 真实数据源：抓取过去几天内最新的 6 篇顶会/核心 AI 论文摘要 (Arxiv)
# ==========================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
MAX_RESULTS = 6
RECENT_DAYS = 10
HISTORY_FILE = "data/radar-history.json"
HISTORY_KEEP_DAYS = 180


def normalize_record(item):
    """
    统一每日知识点字段结构，保证历史文件长期稳定可读。
    """
    return {
        "vendor": item.get("vendor", "Unknown"),
        "date": item.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
        "url": item.get("url", "https://arxiv.org"),
        "concept_name": item.get("concept_name", "Unknown Concept"),
        "tag": item.get("tag", "论文速览"),
        "one_sentence_desc": item.get("one_sentence_desc", "暂无描述"),
        "deep_analysis": item.get("deep_analysis", "暂无深度分析")
    }


def load_history():
    """
    读取历史文件，兼容旧版本结构。
    新结构: {schema_version, timezone, daily}
    旧结构: {"YYYY-MM-DD": [records...]}
    """
    if not os.path.exists(HISTORY_FILE):
        return {
            "schema_version": 1,
            "timezone": "UTC",
            "daily": {}
        }

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "daily" in data:
            data.setdefault("schema_version", 1)
            data.setdefault("timezone", "UTC")
            return data

        # 兼容老版本：直接是日期映射
        if isinstance(data, dict):
            return {
                "schema_version": 1,
                "timezone": "UTC",
                "daily": data
            }
    except Exception:
        pass

    return {
        "schema_version": 1,
        "timezone": "UTC",
        "daily": {}
    }


def prune_history(history, keep_days=HISTORY_KEEP_DAYS):
    """
    仅保留最近 keep_days 天，防止仓库无限膨胀。
    """
    daily = history.get("daily", {})
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=keep_days)

    valid_daily = {}
    for day, records in daily.items():
        try:
            day_date = datetime.strptime(day, "%Y-%m-%d").date()
            if day_date >= cutoff:
                valid_daily[day] = records
        except ValueError:
            continue

    history["daily"] = dict(sorted(valid_daily.items(), key=lambda item: item[0]))
    return history


def save_today_history(radar_data):
    """
    保存今天的6条知识点到历史文件。
    """
    os.makedirs("data", exist_ok=True)
    history = load_history()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_records = [normalize_record(item) for item in radar_data][:MAX_RESULTS]
    history["daily"][today] = today_records

    history = prune_history(history, HISTORY_KEEP_DAYS)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def fetch_recent_ai_papers(max_results=MAX_RESULTS, days=RECENT_DAYS):
    """
    通过 Arxiv API 拉取最新论文，并优先筛选过去 days 天内的数据。
    若过去 days 天不足 max_results 条，会用最新论文补齐，确保页面稳定有内容。
    """
    print(f"正在从 Arxiv 抓取最新论文，目标 {max_results} 条（过去 {days} 天优先）...")
    fetch_size = max(20, max_results * 4)
    url = (
        "http://export.arxiv.org/api/query?"
        "search_query=cat:cs.AI+OR+cat:cs.CL+OR+cat:cs.LG"
        "&sortBy=submittedDate"
        "&sortOrder=descending"
        f"&max_results={fetch_size}"
    )
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        response = urllib.request.urlopen(req)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        # XML namespace 对于 arxiv atom 标准
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        articles = []
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
        for entry in entries:
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
            published = entry.find('atom:published', ns).text[:10]  # 只取 YYYY-MM-DD
            link = entry.find('atom:id', ns).text
            published_date = datetime.strptime(published, "%Y-%m-%d").date()
            
            # 由于 arxiv 上的作者很多，这里暂时用第一作者代表团队/来源
            authors = entry.findall('atom:author/atom:name', ns)
            vendor = authors[0].text if authors else "Arxiv Researcher"
            
            articles.append({
                "vendor": f"Arxiv: {vendor} et al.",
                "url": link,
                "title": title,
                "date": published,
                "raw_text": summary,
                "is_recent": published_date >= cutoff
            })

        recent_articles = [item for item in articles if item["is_recent"]]
        selected = (recent_articles[:max_results] if len(recent_articles) >= max_results
                    else articles[:max_results])

        for item in selected:
            item.pop("is_recent", None)
        return selected
    except Exception as e:
        print(f"抓取真实学术数据失败: {e}")
        return []


def build_fallback_insight(article):
    """
    DeepSeek 不可用时的降级方案，保证前端仍有稳定输出。
    """
    title = article.get("title", "Unknown Paper")
    summary = article.get("raw_text", "")
    concept = title.split(":")[0][:80] if ":" in title else title[:80]
    return {
        "concept_name": concept,
        "tag": "论文速览",
        "one_sentence_desc": f"该论文提出了围绕“{concept}”的新方法，聚焦提升模型能力与训练/推理效率。",
        "deep_analysis": f"核心思路来自论文摘要：{summary[:120]}... 该工作通常通过结构改造、训练策略或评测设计来改善现有方法在精度、泛化和成本上的平衡。",
        "vendor": article.get("vendor", "Arxiv Researcher"),
        "date": article.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
        "url": article.get("url", "https://arxiv.org")
    }

def analyze_with_deepseek(article):
    """
    调用 DeepSeek 的 API，让大模型充当『专业文献学术观察员』
    """
    if not DEEPSEEK_API_KEY:
        print("Warning: 未配置 DEEPSEEK_API_KEY，启用降级摘要生成。")
        return build_fallback_insight(article)

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    # 这是喂给大语言模型的核心Prompt，逼迫它吐出前端需要的字段
    prompt = f"""你是一个顶级的 AI 架构师和前沿观察员。请阅读这篇刚刚过去几天内在 Arxiv 上最新发表的AI学术论文摘要，并为我们提取出其中最具代表性的**『1个核心技术或新概念』**。
即使摘要很学术，你也需要用通俗且极具专业度的中文概括。

务必严格以合法的 JSON 格式输出，包含且仅包含以下四个字段：
- "concept_name": 这个新技术的名称或论文提出的核心模型/算法名称。
- "tag": 简短的标签，如 "大模型微调", "多模态架构", "强化学习框架" 等。
- "one_sentence_desc": 用一句话（具有极强新闻感和行业洞见）解释这个算法、模型或概念到底是什么。
- "deep_analysis": 用大约80到120个汉字，深度大白话剖析它的原理以及解决了业界什么痛点（它让什么变快了？变准了？省钱了？）。

原始论文信息：
【标题】：{article['title']}
【摘要摘要】：{article['raw_text']}
"""

    data = {
        "model": "deepseek-chat", # 或者考虑用深思推理版
        "messages": [
            {"role": "system", "content": "你是一个严谨 JSON 管道处理器，除了完整的合规 JSON 对象，不得输出多余的解释字眼或Markdown符号。"},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.4
    }
    
    req = urllib.request.Request(url, json.dumps(data).encode('utf-8'), headers, method="POST")
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode('utf-8'))
        raw_content = result['choices'][0]['message']['content'].strip()
        
        # 防止大模型强行附带 markdown 标识符
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
            
        parsed_json = json.loads(raw_content)
        
        # 将原始爬虫的公开发表日期等回填
        parsed_json['vendor'] = article['vendor']
        parsed_json['date'] = article['date']
        parsed_json['url'] = article['url']
        return parsed_json
        
    except Exception as e:
        print(f"DeepSeek 解析失败: {e}，启用降级摘要生成。")
        return build_fallback_insight(article)

def main():
    print("🚀 开始执行 AI 前沿雷达任务（过去10天优先）...")
    
    OUTPUT_FILE = "latest-radar.js"
    radar_data = []
    
    recent_articles = fetch_recent_ai_papers(max_results=MAX_RESULTS, days=RECENT_DAYS)
    
    if not recent_articles:
        print("未获取到动态数据，请检查网络...")
        return

    for idx, article in enumerate(recent_articles, 1):
        print(f"[{idx}/{len(recent_articles)}] 正在提炼: {article['title'][:40]}...")
        result = analyze_with_deepseek(article)
        if result:
            radar_data.append(result)

    if radar_data:
        radar_data = [normalize_record(item) for item in radar_data][:MAX_RESULTS]

        js_content = (
            "// 本文件由 GitHub Actions 每日触发，通过 Arxiv + DeepSeek 自动生成\n"
            f"// 生成时间(UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        js_content += f"window.dailyRadarData = {json.dumps(radar_data, ensure_ascii=False, indent=2)};\n"

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(js_content)

        save_today_history(radar_data)
            
        print(f"🔥 完成！共生成 {len(radar_data)} 条前沿知识点，结果写入 {OUTPUT_FILE} 与 {HISTORY_FILE}")
    else:
        print("未生成有效结果。")

if __name__ == "__main__":
    main()