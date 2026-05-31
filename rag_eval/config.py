from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Primary model (generator + primary judge)
    model_base_url: str = "http://localhost:11434/v1"
    model_api_key: str = "none"
    model_name: str = "qwen2.5-coder:7b"

    # Secondary judge (dual-judge agreement)
    # Default: same Ollama server, lighter model for diversity
    secondary_base_url: str = "http://localhost:11434/v1"
    secondary_api_key: str = "none"
    secondary_model: str = "llama3.2:3b"

    # Eval knobs
    correctness_threshold: float = 0.7
    max_workers: int = 1
    request_timeout_sec: int = 120

    # Paths
    winmerge_repo_path: str = "./winmerge"

    # Dense retrieval (hybrid BM25 + dense, fused with RRF — see spec §1)
    use_dense: bool = True
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_device: str = "cpu"   # cpu avoids VRAM contention with Ollama
    bm25_min_score: float = 15.0    # BM25 joins fusion only above this (real lexical match)

    # Dry-run: return mock responses instead of real API calls
    dry_run: bool = False

    # Token limits — critical for CPU inference speed
    max_tokens_generator: int = 512   # generator answer length cap
    max_tokens_judge: int = 200       # judge JSON output is short (~80 tok)
    num_ctx: int = 4096               # Ollama context window (default 2048 is too small)

    # MLflow experiment tracking (optional; set to "" to disable)
    mlflow_tracking_uri: str = "./mlruns"

    # SQLite metrics database for Grafana datasource
    metrics_db_path: str = "./reports/metrics.db"

    @model_validator(mode="after")
    def _check_secondary(self) -> "Settings":
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
