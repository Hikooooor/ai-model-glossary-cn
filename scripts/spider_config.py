"""Centralized constants for fetch scope, scoring signals, and persistence behavior."""

import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
# Fixed daily output size shown on homepage and saved into history.
MAX_RESULTS = 6
RECENT_DAYS = 10
HF_DAYS_BACK = 3
HISTORY_FILE = "data/radar-history.json"
HISTORY_INDEX_FILE = "data/history-index.json"
# Keep rolling history window to prevent unbounded repo growth.
HISTORY_KEEP_DAYS = 180
MAX_WORKERS = 6
DEDUP_DAYS = 7

PRACTITIONER_KEYWORDS = [
    "deploy", "inference", "agent", "rag", "retrieval", "fine-tun", "finetun",
    "lora", "quantiz", "distill", "efficient", "cost", "latency", "throughput",
    "tool use", "benchmark", "multimodal", "multi-modal", "reasoning", "alignment",
    "推理", "部署", "微调", "多模态", "智能体"
]

TOP_CONF_KEYWORDS = [
    "ICLR", "NeurIPS", "ICML", "ACL", "EMNLP", "CVPR", "ICCV", "ECCV", "NAACL"
]

TOP_ORGS = [
    "google", "openai", "meta ", "microsoft", "anthropic", "deepmind", "apple",
    "tsinghua", "peking university", "stanford", "mit ", "cmu", "uc berkeley",
    "amazon", "nvidia", "samsung", "baidu", "alibaba", "tencent"
]
