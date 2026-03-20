import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
GENERATIONS_DIR = RESULTS_DIR / "generations"
JUDGMENTS_DIR = RESULTS_DIR / "judgments"
METRICS_DIR = RESULTS_DIR / "metrics"

for d in (GENERATIONS_DIR, JUDGMENTS_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Model registry: name -> provider + OpenRouter model ID
# All generation routes through OpenRouter; judge still uses Anthropic directly
MODELS: dict[str, dict] = {
    "claude-opus-4-6":        {"provider": "openrouter", "api_model": "anthropic/claude-opus-4.6"},
    "claude-sonnet-4-6":      {"provider": "openrouter", "api_model": "anthropic/claude-sonnet-4.6"},
    "gpt-5.2":                {"provider": "openrouter", "api_model": "openai/gpt-5.2"},
    "gpt-5.3-chat-latest":    {"provider": "openrouter", "display": "gpt-5.3", "api_model": "openai/gpt-5.3-chat"},
    "gemini-3-flash-preview": {"provider": "openrouter", "display": "gemini-3-flash", "api_model": "google/gemini-3-flash-preview"},
    "gemini-3.1-pro-preview": {"provider": "openrouter", "display": "gemini-3.1-pro", "api_model": "google/gemini-3.1-pro-preview"},
    "deepseek-reasoner":      {"provider": "openrouter", "display": "deepseek-r1", "api_model": "deepseek/deepseek-r1"},
}

# Dataset registry: short key -> xlsx filename
DATASETS: dict[str, str] = {
    "slides":     "Slide Training for Consultants.xlsx",
    "audit":      "Audit_Assurance Accounting Spreadsheet.xlsx",
    "commodities":"Commodities Trading S&T.xlsx",
    "insurance":  "Insurance Adjudication Post-Training Dataset.xlsx",
    "insurance_v2": "Updated Adjudication Post-Training Dataset.xlsx",
    "insurance_v3": "Revised Insurance Dataset.xlsx",
    "multistep":  "Multi-Step Business Reasoning Datset.xlsx",
    "california": "california_pc_ho_mix_dataset.xlsx",
    "ca_claims":  "ca_claims_reasoning_dataset_n10.xlsx",
    "hard_pc":    "hard_pc_claims_benchmark_dataset.xlsx",
    "new_approach": "New Approach.xlsx",
    "pc_interim":   "P&C Insurance Interim.xlsx",
    "pc_sample":    "P&C Insurance Sample Test.xlsx",
    "pc_pass3_ready": "P&C Insurance Pass@3 Ready.xlsx",
}

JUDGE_MODEL = "claude-sonnet-4-6"

K = 3                   # completions per problem
MAX_JUDGE_CONCURRENCY = 5   # global cap on concurrent judge calls across all parallel model runs
PASS_THRESHOLD = 0.7    # fraction of rubric criteria met to count as pass
GENERATION_TEMPERATURE = 0.8
MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds

API_KEYS = {
    "anthropic":   os.getenv("ANTHROPIC_API_KEY", ""),
    "openrouter":  os.getenv("OPENROUTER_API_KEY", ""),
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
