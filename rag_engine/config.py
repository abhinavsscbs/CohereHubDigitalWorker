import os
import re
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


def _coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if re.fullmatch(r"-?\d+", raw):
        try:
            return int(raw)
        except Exception:
            return value
    if re.fullmatch(r"-?\d+\.\d+", raw):
        try:
            return float(raw)
        except Exception:
            return value
    return value


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        expanded = _ENV_PATTERN.sub(
            lambda m: os.getenv(m.group(1), m.group(2) or ""),
            value,
        )
        return _coerce_scalar(expanded)
    return value


def _load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env(data)


def _resolve_path(path_value: str) -> str:
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(BASE_DIR, path_value)


load_dotenv()

CONFIG_PATH = os.getenv(
    "RAG_ENGINE_CONFIG",
    os.path.join(BASE_DIR, "backend", "config.yaml"),
)
CONFIG: Dict[str, Any] = _load_config(CONFIG_PATH)

_rag_cfg = CONFIG.get("rag", {}) or {}
_llm_cfg = CONFIG.get("llm", {}) or {}
_session_cfg = CONFIG.get("session", {}) or {}

COHERE_ENDPOINT_URL = _llm_cfg.get("endpoint_url", "http://YOUR_COHERE_ENDPOINT")
COHERE_API_KEY = _llm_cfg.get("api_key", "")
COHERE_REQUEST_TIMEOUT_SEC = float(_llm_cfg.get("timeout_sec", 120))
COHERE_VERIFY_SSL = bool(_llm_cfg.get("verify_ssl", False))

LLM_MODELS = _llm_cfg.get("models", {}) or {}
LLM_TEMPERATURES = _llm_cfg.get("temperatures", {}) or {}
LLM_MAX_TOKENS = _llm_cfg.get("max_tokens", {}) or {}

if not LLM_MODELS:
    LLM_MODELS = {
        "mini": "cohere-mini",
        "full": "cohere-full",
        "extractor": "cohere-extractor",
        "relevance": "cohere-relevance",
    }
if not LLM_TEMPERATURES:
    LLM_TEMPERATURES = {
        "mini": 0.0,
        "full": 0.0,
        "extractor": 0.0,
        "relevance": 0.0,
    }
if not LLM_MAX_TOKENS:
    LLM_MAX_TOKENS = {
        "mini": 30000,
        "full": 32000,
        "extractor": 32000,
        "relevance": 8000,
    }

STAGE_1_THRESHOLD = float(_rag_cfg.get("thresholds", {}).get("stage_1", 0.5))
STAGE_2_PERCENTILE = float(_rag_cfg.get("thresholds", {}).get("stage_2_percentile", 0.30))
RAG_SEED = int(_rag_cfg.get("seed", 42))
_embed_cfg = _rag_cfg.get("embedder", {}) or {}
EMBEDDINGS_MODEL = _embed_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDINGS_DEVICE = _embed_cfg.get("device", "cpu")

DB_PATHS: List[Dict[str, Any]] = []
for item in _rag_cfg.get("db_paths", []) or []:
    if not isinstance(item, dict):
        continue
    db_path = _resolve_path(str(item.get("path", "")))
    DB_PATHS.append(
        {
            "path": db_path,
            "name": item.get("name", "DB"),
            "score_threshold": float(item.get("score_threshold", 0.5)),
        }
    )

if not DB_PATHS:
    DB_PATHS = [
        {
            "path": _resolve_path("IFRS_A_embed_test"),
            "name": "IFRS A",
            "score_threshold": 0.5,
        },
        {
            "path": _resolve_path(os.path.join("IFRS_B_embed_test", "IFRS_B_embed_test")),
            "name": "IFRS B",
            "score_threshold": 0.5,
        },
        {
            "path": _resolve_path("IFRS_C_embed_test"),
            "name": "IFRS C",
            "score_threshold": 0.5,
        },
        {
            "path": _resolve_path("EY_embed_test"),
            "name": "EY",
            "score_threshold": 0.5,
        },
        {
            "path": _resolve_path("PwC_embed_test"),
            "name": "PwC",
            "score_threshold": 0.5,
        },
    ]

REPLACE_EXCEL_PATH = _resolve_path(_rag_cfg.get("replace_excel_path", "Replacement_data.xlsx"))

REDIS_URL = (
    _session_cfg.get("redis", {}) or {}
).get("url", "redis://localhost:6379/0")
CHAT_HISTORY_DIR = _resolve_path(
    (_session_cfg.get("file", {}) or {}).get("dir", os.path.join("backend", "chat_history"))
)

__all__ = [
    "BASE_DIR",
    "CONFIG_PATH",
    "CONFIG",
    "COHERE_ENDPOINT_URL",
    "COHERE_API_KEY",
    "COHERE_REQUEST_TIMEOUT_SEC",
    "COHERE_VERIFY_SSL",
    "LLM_MODELS",
    "LLM_TEMPERATURES",
    "LLM_MAX_TOKENS",
    "STAGE_1_THRESHOLD",
    "STAGE_2_PERCENTILE",
    "RAG_SEED",
    "EMBEDDINGS_MODEL",
    "EMBEDDINGS_DEVICE",
    "DB_PATHS",
    "REPLACE_EXCEL_PATH",
    "REDIS_URL",
    "CHAT_HISTORY_DIR",
]
