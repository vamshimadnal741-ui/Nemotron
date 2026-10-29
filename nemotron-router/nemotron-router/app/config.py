import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file from the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def parse_price(val: str | None, default: tuple[float, float]) -> tuple[float, float]:
    """Parse comma-separated input,output price per 1M tokens."""
    if not val:
        return default
    try:
        parts = [float(p.strip()) for p in val.split(",")]
        if len(parts) == 2:
            return parts[0], parts[1]
    except Exception:
        pass
    return default


def clean_env(name: str, default: str = "") -> str:
    """Read an env var, tolerating quotes and an accidentally doubled 'NAME=' prefix."""
    val = os.getenv(name, default).strip().strip("\"'").strip()
    if val.upper().startswith(name.upper() + "="):
        val = val[len(name) + 1:].strip()
    return val


def clean_model(name: str) -> str | None:
    """Model override must look like a model id (contains letters); ignore e.g. price strings."""
    val = clean_env(name)
    if not val or not any(c.isalpha() for c in val):
        return None
    return val


class Settings:
    def __init__(self):
        self.root_dir = ROOT_DIR
        self.api_key = clean_env("NEBIUS_API_KEY")
        self.base_url = clean_env("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/") or "https://api.tokenfactory.nebius.com/v1/"
        
        # Ensure trailing slash on base_url for OpenAI client
        if not self.base_url.endswith("/"):
            self.base_url += "/"

        # Overrides
        self.model_small = clean_model("MODEL_SMALL")
        self.model_mid = clean_model("MODEL_MID")
        self.model_large = clean_model("MODEL_LARGE")

        # Pricing per 1M tokens in USD: (input, output)
        self.price_small = parse_price(os.getenv("PRICE_SMALL"), (0.10, 0.20))
        self.price_mid = parse_price(os.getenv("PRICE_MID"), (0.60, 1.20))
        self.price_large = parse_price(os.getenv("PRICE_LARGE"), (2.50, 8.00))

        # Placeholder indicator
        raw_placeholder = os.getenv("PRICES_ARE_PLACEHOLDER", "true").strip().lower()
        self.prices_are_placeholder = raw_placeholder in ("true", "1", "yes")

        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = self._int_env("PORT", 8000)

        # Generation limits / client behaviour
        self.max_output_tokens = self._int_env("MAX_OUTPUT_TOKENS", 1024)
        self.classifier_max_tokens = self._int_env("CLASSIFIER_MAX_TOKENS", 256)
        self.request_timeout = float(os.getenv("REQUEST_TIMEOUT", "60") or 60)

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)).strip())
        except ValueError:
            return default

    @property
    def key_looks_like_nvidia(self) -> bool:
        """NVIDIA build.nvidia.com keys start with 'nvapi-'; they don't work on Nebius."""
        return self.api_key.startswith("nvapi-") and "nvidia.com" not in self.base_url

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key != "your_nebius_api_key_here")

    def get_openai_client(self) -> OpenAI:
        # Never crash on empty key at startup, use placeholder so client init succeeds
        key = self.api_key if self.has_api_key else "placeholder-key-for-discovery"
        return OpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=self.request_timeout,
            max_retries=1,
        )


settings = Settings()
