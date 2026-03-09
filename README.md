# AI 每日前沿技术雷达

一个静态站点 + 自动化数据管线项目。每天定时抓取 Arxiv 论文，结合 HuggingFace 热度信号进行打分筛选，再调用 DeepSeek 生成结构化解读，最终自动更新前端展示与历史归档。

## 在线地址

- https://hikooooor.github.io/ai-model-glossary-cn/

## 你将得到什么

- 每日固定 6 条 AI 前沿卡片（精选 + 值得关注）
- 自动落地当前日数据到 `latest-radar.js`
- 自动维护历史数据（总表 + 月分片 + 索引）
- 前端支持“今日卡片 + 历史懒加载”浏览
- GitHub Actions 全自动执行与自动提交

## 项目全景

### 目录结构

- `index.html`：首页结构与两段核心脚本（今日卡片渲染、历史懒加载）
- `styles.css`：页面样式
- `latest-radar.js`：每日产物，挂载 `window.dailyRadarMeta` 与 `window.dailyRadarData`
- `data/radar-history.json`：全量历史总表（按日期聚合）
- `data/YYYY-MM.json`：月分片历史，供前端按月加载
- `data/history-index.json`：月分片目录索引
- `data/last-run.json`：最近一次工作流运行信息
- `scripts/daily_spider.py`：主入口，编排完整抓取与生成流程
- `scripts/spider_fetch.py`：数据抓取层（HF + Arxiv）
- `scripts/spider_score.py`：评分与分层（featured/notable）
- `scripts/spider_deepseek.py`：DeepSeek 调用与质量兜底
- `scripts/spider_history.py`：历史归档、去重、裁剪、月分片索引
- `scripts/spider_config.py`：集中配置参数
- `.github/workflows/daily-update.yml`：定时任务 + 自动提交

### 端到端数据流

1. GitHub Actions 在 UTC `00:00` 和 `00:15` 触发。
2. 执行 `python scripts/daily_spider.py`。
3. 读取历史，提取近 `DEDUP_DAYS` 的 URL 作为去重集合。
4. 抓取 HuggingFace Daily Papers 作为热度加权信号。
5. 抓取 Arxiv 6 个 AI 类目最新论文，优先近 `RECENT_DAYS` 天。
6. 按多信号策略评分，选出 `MAX_RESULTS=6`。
7. 并发调用 DeepSeek 生成结构化内容，失败时回退本地兜底摘要。
8. 写入 `latest-radar.js`（前端今日区数据源）。
9. 同步写入历史：
10. 更新 `data/radar-history.json`。
11. 更新 `data/YYYY-MM.json`。
12. 更新 `data/history-index.json`。
13. 工作流写入 `data/last-run.json`。
14. 若存在变更，自动 `git commit` + `git push`。

## 核心工作流程（详细）

### 1) 入口编排：`scripts/daily_spider.py`

- `run_pipeline(dry_run=False)`
- 加载历史并计算近几天 URL 去重集合。
- 抓取 HF 与 Arxiv。
- 调用 `select_top_articles` 做排序筛选。
- 用线程池并发调用 `analyze_with_deepseek`。
- 对返回结果统一 `normalize_record`，保证字段完整。
- `write_outputs(radar_data)`
- 生成 `window.dailyRadarMeta` + `window.dailyRadarData`。
- 写入 `latest-radar.js`。
- 调用 `save_today_history` 写历史文件。

### 2) 抓取层：`scripts/spider_fetch.py`

- `fetch_hf_daily_papers(days_back)`
- 拉取 HuggingFace Daily Papers 指定日期推荐。
- 返回 `{arxiv_id: upvotes}` 映射，作为“外部热度信号”。
- `fetch_recent_ai_papers(max_results, days)`
- 查询 Arxiv Atom API：
- `cs.AI` `cs.CL` `cs.LG` `cs.CV` `cs.MA` `cs.IR`
- 标记每篇 `is_recent`。
- 若近期论文不足则回退使用全池，避免当天结果为空。

### 3) 评分与筛选：`scripts/spider_score.py`

- `score_article(article, hf_upvotes_map, seen_urls)`
- 去重命中直接返回 `-1`（硬过滤）。
- 多信号打分项：
- HF 推荐与赞数分档加权
- 是否含源码（`github.com`）
- 是否命中从业者关键词（部署/推理/微调/多模态等）
- 是否命中顶会关键词（ICLR/NeurIPS/...）
- 是否命中顶级机构
- `select_top_articles(...)`
- 按分数降序取前 6。
- Top 3 标记 `featured`，其余为 `notable`。

### 4) AI 提炼与兜底：`scripts/spider_deepseek.py`

- `analyze_with_deepseek(article, max_retry=3)`
- 若无 `DEEPSEEK_API_KEY`，直接降级到 `build_fallback_insight`。
- 有 Key 时调用 DeepSeek Chat Completions，要求只返回 JSON。
- 失败采用指数退避重试（1s, 2s, 4s）。
- `_enforce_quality(parsed, article)`
- 统一字段兜底。
- 强制单段落、最小信息密度，以及“结论-价值-边界”三层表达。
- 文案不足时自动补全，保证前端展示稳定。

### 5) 历史持久化：`scripts/spider_history.py`

- `load_history()`
- 兼容旧结构并补默认 schema。
- `get_recent_urls(history, days)`
- 取最近 N 天 URL 做跨天去重。
- `save_today_history(radar_data)`
- 先更新 `data/radar-history.json`。
- 再写月分片 `data/YYYY-MM.json`。
- 再更新 `data/history-index.json`。
- `prune_history(history, keep_days)`
- 只保留最近 `HISTORY_KEEP_DAYS`，防止历史无限膨胀。

## 前端加载流程

### 今日卡片（`index.html`）

1. 加载 `latest-radar.js`。
2. 读取 `window.dailyRadarData` 和 `window.dailyRadarMeta`。
3. 渲染卡片网格，展示供应方、日期、标签、摘要、深度解析、信号徽章。
4. 顶部显示生成时间（北京时间/UTC）。

### 历史卡片（`index.html`）

1. 先请求 `data/history-index.json` 获取可用月份。
2. 按月依次拉取 `data/YYYY-MM.json` 并合并。
3. 以“按天分组”的方式分批渲染（初始 3 天，滚动追加 2 天）。
4. 若月分片失败，回退读取 `data/radar-history.json`。
5. 无数据时显示空态文案。

## 数据格式约定

### `latest-radar.js`

```js
window.dailyRadarMeta = {
   "generated_at_utc": "YYYY-MM-DD HH:mm:ss",
   "generated_at_beijing": "YYYY-MM-DD HH:mm:ss"
};

window.dailyRadarData = [
   {
      "vendor": "Arxiv: ...",
      "date": "YYYY-MM-DD",
      "url": "http://arxiv.org/abs/...",
      "concept_name": "...",
      "tag": "...",
      "one_sentence_desc": "...",
      "deep_analysis": "...",
      "tier": "featured|notable",
      "score": 0,
      "signals": ["..."]
   }
];
```

### `data/radar-history.json`

```json
{
   "schema_version": 2,
   "timezone": "UTC",
   "daily": {
      "2026-03-05": [
         {
            "vendor": "...",
            "date": "2026-03-05",
            "url": "...",
            "concept_name": "...",
            "tag": "...",
            "one_sentence_desc": "...",
            "deep_analysis": "...",
            "tier": "featured",
            "score": 2,
            "signals": ["HF推荐"]
         }
      ]
   }
}
```

### `data/history-index.json`

```json
{
   "months": ["2026-03", "2026-02"],
   "updated": "YYYY-MM-DD HH:mm:ss"
}
```

### `data/last-run.json`

```json
{
   "last_run_utc": "YYYY-MM-DD HH:mm:ss",
   "last_run_beijing": "YYYY-MM-DD HH:mm:ss",
   "workflow": "daily-update",
   "run_id": "...",
   "run_number": "..."
}
```

## 配置说明

### 环境变量

- `DEEPSEEK_API_KEY`：DeepSeek API Key（可选，但强烈建议）

未配置时，系统会走降级摘要，不会中断主流程。

### 核心参数（`scripts/spider_config.py`）

- `MAX_RESULTS = 6`：每天输出条数
- `RECENT_DAYS = 10`：优先最近天数
- `HF_DAYS_BACK = 3`：HF 信号回看天数
- `MAX_WORKERS = 6`：DeepSeek 并发数
- `DEDUP_DAYS = 7`：跨天 URL 去重窗口
- `HISTORY_KEEP_DAYS = 180`：历史保留天数

## GitHub Actions 工作流说明

文件：`.github/workflows/daily-update.yml`

- 触发器：
- `schedule`: UTC `00:00` + `00:15`
- `workflow_dispatch`: 手动触发
- `push`: 修改 `scripts/daily_spider.py` 时触发一次
- 执行环境：`ubuntu-latest`, Python `3.11`
- 权限：`contents: write`（用于自动提交）
- 产物提交策略：仅有文件差异时才提交，避免空提交

## 本地开发与调试

### 1) 本地预览网页

```bash
python -m http.server 8000
```

访问：`http://localhost:8000`

### 2) 本地执行完整爬取流程

Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY = "你的key"
python scripts/daily_spider.py
```

### 3) 仅调试，不写文件

```powershell
python scripts/daily_spider.py --dry-run
```

## 常见问题与排查

### 定时任务未触发

- 确认默认分支是 `main`
- 确认仓库 Actions 未被暂停
- 看 `.github/workflows/daily-update.yml` 是否在默认分支

### 有运行但页面未更新

- 检查 `data/last-run.json` 时间是否刷新
- 检查 `latest-radar.js` 是否有新时间戳
- 检查 Actions 日志里是否出现 API 超时或抓取失败

### DeepSeek 调用失败

- 验证 `DEEPSEEK_API_KEY` 是否有效
- 查看是否触发了重试后降级
- 即使降级，页面也会有基础内容，不会空白

### 历史显示异常

- 确认 `data/history-index.json` 中 `months` 非空
- 确认对应 `data/YYYY-MM.json` 文件存在
- 前端会自动回退 `data/radar-history.json`

## 维护建议

- 若想提高“新鲜度”：调小 `HF_DAYS_BACK` 与 `RECENT_DAYS`
- 若想提高“稳定性”：适当降低 `MAX_WORKERS`
- 若想提高“去重力度”：增大 `DEDUP_DAYS`
- 若仓库增长过快：减小 `HISTORY_KEEP_DAYS`

## 许可证与说明

- 本项目聚合公开论文摘要并生成解读文本。
- 解读内容基于公开摘要自动生成，仅供学习参考。
- 学术结论请以原论文为准。
