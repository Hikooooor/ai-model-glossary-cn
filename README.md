# AI 每日前沿技术雷达

一个纯静态前端项目：每日自动抓取最新 AI 论文（Arxiv），再由 DeepSeek 提炼成可读的 6 条核心知识点。

## 在线地址

- https://hikooooor.github.io/ai-model-glossary-cn/

## 核心功能

- 每日雷达卡片（来源、日期、概念、摘要、深度解析）
- GitHub Actions 定时任务自动更新 `latest-radar.js`

## 项目结构

- `index.html`：页面结构 + 交互逻辑 + 雷达渲染
- `styles.css`：全站样式（含雷达模块样式）
- `latest-radar.js`：每日自动生成的雷达数据
- `data/radar-history.json`：每日历史归档（按日期保存每天6条）
- `scripts/daily_spider.py`：Arxiv 抓取 + DeepSeek 解析脚本
- `.github/workflows/daily-update.yml`：每日自动任务

## 自动化流程

1. GitHub Actions 每日触发（UTC 0 点）。
2. `daily_spider.py` 抓取 Arxiv 最新论文，优先筛选过去 10 天内容。
3. 调用 DeepSeek 生成结构化结果：
   - `concept_name`
   - `tag`
   - `one_sentence_desc`
   - `deep_analysis`
4. 输出到 `latest-radar.js`（当天展示）并写入 `data/radar-history.json`（历史归档）。
5. 工作流自动提交这两个文件到仓库。

## 定时任务排查（Schedule 不触发）

若发现没有按预期时间更新，优先检查：

- 仓库默认分支是否为 `main`（`schedule` 仅在默认分支生效）。
- 仓库是否长期无活动（GitHub 可能暂停定时任务，重新启用 Actions 后恢复）。
- Actions 页面是否存在失败记录（网络波动/外部 API 超时会导致当次失败）。

当前工作流已加入以下保障：

- 每天 UTC `00:00` + `00:15` 双触发（第二次作为兜底补偿）。
- 自动写入 `data/last-run.json`，可直接查看最近一次实际运行时间。
- 自动提交 `latest-radar.js` 与 `data/*.json`，避免新增历史分片未提交。

## 历史归档结构

`data/radar-history.json` 结构如下：

```json
{
   "schema_version": 1,
   "timezone": "UTC",
   "daily": {
      "2026-03-04": [
         {
            "vendor": "...",
            "date": "2026-03-04",
            "url": "...",
            "concept_name": "...",
            "tag": "...",
            "one_sentence_desc": "...",
            "deep_analysis": "..."
         }
      ]
   }
}
```

## 必要配置

在仓库 Settings -> Secrets and variables -> Actions 中新增：

- `DEEPSEEK_API_KEY`：你的 DeepSeek API Key

## 本地运行

```bash
git clone https://github.com/Hikooooor/ai-model-glossary-cn.git
cd ai-model-glossary-cn
python -m http.server 8000
```

浏览器打开：`http://localhost:8000`

## 本地测试爬虫

```bash
set DEEPSEEK_API_KEY=你的key
python scripts/daily_spider.py
```

执行后会刷新 `latest-radar.js`。

## 说明

- 若 DeepSeek 暂时不可用，脚本会自动降级生成基础摘要，避免页面空白。
- 当前策略固定输出最多 6 条，优先使用过去 10 天数据。
- 历史文件默认只保留最近 180 天，避免仓库文件无限增长。
