import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from spider_config import DEDUP_DAYS, HF_DAYS_BACK, MAX_RESULTS, MAX_WORKERS, RECENT_DAYS
from spider_deepseek import analyze_with_deepseek
from spider_fetch import fetch_hf_daily_papers, fetch_recent_ai_papers
from spider_history import get_recent_urls, load_history, normalize_record, save_today_history
from spider_score import select_top_articles


def run_pipeline(dry_run=False):
    print(" 开始执行 AI 前沿雷达任务（模块化管线）...")
    if dry_run:
        print("  --dry-run 模式：结果只打印，不写文件")

    history = load_history()
    seen_urls = get_recent_urls(history, days=DEDUP_DAYS)
    print(f"近{DEDUP_DAYS}天已收录 URL：{len(seen_urls)} 篇（用于去重）")

    hf_upvotes_map = fetch_hf_daily_papers(days_back=HF_DAYS_BACK)
    arxiv_articles = fetch_recent_ai_papers(max_results=MAX_RESULTS, days=RECENT_DAYS)

    if not arxiv_articles:
        print("未获取到任何论文，请检查网络。")
        return []

    selected = select_top_articles(arxiv_articles, hf_upvotes_map, seen_urls, max_results=MAX_RESULTS)
    if not selected:
        print("评分筛选后无可用论文（可能全部命中去重）。")
        return []

    print(f"\n评分排行（Top {len(selected)}）：")
    for i, art in enumerate(selected, 1):
        print(f"  [{i}] 分={art['score']} tier={art['tier']} 信号={art['signals']}")
        print(f"       {art['title'][:60]}...")

    print(f"\n 并发提炼中（最多 {MAX_WORKERS} 并发）...")
    radar_data = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(analyze_with_deepseek, art): i for i, art in enumerate(selected)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            if result:
                radar_data[idx] = result
                print(f"   [{idx + 1}] {result.get('concept_name', '')[:35]}")

    return [normalize_record(r) for r in radar_data if r is not None]


def write_outputs(radar_data):
    output_file = "latest-radar.js"
    now_utc = datetime.now(timezone.utc)
    now_bj = now_utc.astimezone(timezone(timedelta(hours=8)))
    meta = {
        "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_at_beijing": now_bj.strftime("%Y-%m-%d %H:%M:%S"),
    }
    js_content = (
        "// 本文件由 GitHub Actions 每日触发，通过 Arxiv + HuggingFace + DeepSeek 自动生成\n"
        f"// 生成时间(UTC): {meta['generated_at_utc']}\n"
        f"window.dailyRadarMeta = {json.dumps(meta, ensure_ascii=False)};\n"
        f"window.dailyRadarData = {json.dumps(radar_data, ensure_ascii=False, indent=2)};\n"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(js_content)

    save_today_history(radar_data)

    featured = [r for r in radar_data if r.get("tier") == "featured"]
    notable = [r for r in radar_data if r.get("tier") == "notable"]
    month_key = datetime.now(timezone.utc).strftime("%Y-%m")
    print(
        f"\n 完成！共 {len(radar_data)} 条"
        f"（精选 {len(featured)}  值得关注 {len(notable)}）\n"
        f"   写入：{output_file} | data/radar-history.json | data/{month_key}.json | data/history-index.json"
    )


def main():
    parser = argparse.ArgumentParser(description="AI 前沿雷达生成脚本")
    parser.add_argument("--dry-run", action="store_true", help="只打印结果，不写入任何文件（调试用）")
    args = parser.parse_args()

    radar_data = run_pipeline(dry_run=args.dry_run)
    if not radar_data:
        print("未生成有效结果。")
        return

    if args.dry_run:
        print("\n[dry-run] 最终输出预览：")
        print(json.dumps(radar_data, ensure_ascii=False, indent=2))
        return

    write_outputs(radar_data)


if __name__ == "__main__":
    main()
