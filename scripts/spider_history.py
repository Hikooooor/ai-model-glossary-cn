import json
import os
from datetime import datetime, timedelta, timezone

from spider_config import DEDUP_DAYS, HISTORY_FILE, HISTORY_INDEX_FILE, HISTORY_KEEP_DAYS, MAX_RESULTS


def normalize_record(item):
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
        "signals": item.get("signals", []),
    }


def load_history():
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
    daily = history.get("daily", {})
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=keep_days)
    valid_daily = {
        day: records
        for day, records in daily.items()
        if datetime.strptime(day, "%Y-%m-%d").date() >= cutoff
    }
    history["daily"] = dict(sorted(valid_daily.items()))
    return history


def save_monthly_file(today, today_records):
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
        new_items = [r for r in today_records if r.get("url") not in existing_urls]
        month_data[today] = month_data[today] + new_items

    month_data = dict(sorted(month_data.items()))
    with open(month_file, "w", encoding="utf-8") as f:
        json.dump(month_data, f, ensure_ascii=False, indent=2)
    return month_key


def update_history_index(month_key):
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
