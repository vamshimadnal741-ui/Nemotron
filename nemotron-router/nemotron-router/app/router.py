import json
import logging
import re
import time
from collections import deque
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger("nemotron-router")

# Regex helpers
THINK_TAG_REGEX = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK_REGEX = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
PARAM_SIZE_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]")

# Keywords to exclude from chat model discovery
EXCLUDE_KEYWORDS = [
    "embed",
    "rerank",
    "guard",
    "safety",
    "reward",
    "vision",
    "audio",
    "speech",
]


def clean_think_tags(text: str) -> str:
    """Remove any <think>...</think> and unclosed <think>... from text."""
    if not text:
        return ""
    cleaned = THINK_TAG_REGEX.sub("", text)
    cleaned = UNCLOSED_THINK_REGEX.sub("", cleaned)
    return cleaned.strip()


def parse_param_size(model_id: str) -> float:
    """Extract parameter count in billions from model ID (e.g. '9B', '49B', '253B')."""
    match = PARAM_SIZE_REGEX.search(model_id)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0


def calculate_cost(prompt_tokens: int, completion_tokens: int, price: Tuple[float, float]) -> float:
    """Calculate cost in USD given prompt/completion tokens and (input, output) price per 1M tokens."""
    input_rate, output_rate = price
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000.0


class NemotronRouter:
    def __init__(self):
        self.lock = Lock()
        self.client = settings.get_openai_client()
        
        # Discovered and assigned models
        self.discovered_models: List[Dict[str, Any]] = []
        self.tier_models: Dict[str, str] = {
            "small": settings.model_small or "nvidia/nemotron-mini-4b-instruct",
            "mid": settings.model_mid or "nvidia/llama-3.1-nemotron-70b-instruct",
            "large": settings.model_large or "nvidia/nemotron-4-340b-instruct",
        }
        self.models_discovered_ok = False
        self.discovery_error: Optional[str] = None

        # Statistics & History
        self.total_requests: int = 0
        self.total_cost: float = 0.0
        self.total_baseline: float = 0.0
        self.per_tier_counts: Dict[str, int] = {"small": 0, "mid": 0, "large": 0}
        self.recent_requests: deque = deque(maxlen=100)

    def discover_models(self) -> Dict[str, str]:
        """
        Query GET /models, filter NVIDIA/Nemotron chat models, sort by size,
        and assign small, mid, large tiers (allowing env overrides).
        """
        try:
            self.client = settings.get_openai_client()
            response = self.client.models.list()
            raw_models = getattr(response, "data", []) or []

            nemotron_models = []
            seen_ids = set()
            for m in raw_models:
                m_id = getattr(m, "id", "")
                m_id_lower = m_id.lower()
                
                # Must contain nvidia or nemotron
                if "nvidia" not in m_id_lower and "nemotron" not in m_id_lower:
                    continue
                
                # Exclude non-chat / specialized models
                if any(kw in m_id_lower for kw in EXCLUDE_KEYWORDS):
                    continue
                
                if m_id in seen_ids:
                    continue
                seen_ids.add(m_id)

                size = parse_param_size(m_id)
                nemotron_models.append({"id": m_id, "size_b": size})

            if nemotron_models:
                # Sort ascending by parameter size, then by name
                nemotron_models.sort(key=lambda x: (x["size_b"], x["id"]))
                self.discovered_models = nemotron_models

                # Select small, mid, large
                # Models whose size can't be parsed (size 0) sort first and would
                # wrongly become "small", so tier only on sized models when possible.
                sized = [m for m in nemotron_models if m["size_b"] > 0] or nemotron_models
                n = len(sized)
                small_candidate = sized[0]["id"]
                mid_candidate = sized[n // 2]["id"] if n > 2 else sized[-1]["id"]
                large_candidate = sized[-1]["id"]

                self.tier_models["small"] = settings.model_small or small_candidate
                self.tier_models["mid"] = settings.model_mid or mid_candidate
                self.tier_models["large"] = settings.model_large or large_candidate
                self.models_discovered_ok = True
                self.discovery_error = None
                logger.info(
                    f"Discovered {n} Nemotron models. Selected: small={self.tier_models['small']}, "
                    f"mid={self.tier_models['mid']}, large={self.tier_models['large']}"
                )
            else:
                self.discovery_error = "No NVIDIA/Nemotron chat models found in API response; using defaults."
                self.models_discovered_ok = False
                logger.warning(self.discovery_error)
        except Exception as e:
            self.discovery_error = str(e)
            if "401" in str(e) and settings.api_key.startswith("nvapi-"):
                self.discovery_error += (
                    " | Hint: this looks like an NVIDIA (nvapi-) key; "
                    "the hackathon requires a Nebius Token Factory key."
                )
            self.models_discovered_ok = False
            logger.warning(f"Model discovery failed ({e}); falling back to default models.")

        # Always apply manual overrides if provided in .env
        if settings.model_small:
            self.tier_models["small"] = settings.model_small
        if settings.model_mid:
            self.tier_models["mid"] = settings.model_mid
        if settings.model_large:
            self.tier_models["large"] = settings.model_large

        return self.tier_models

    def classify_difficulty(self, prompt: str) -> Tuple[int, int, int]:
        """
        Ask the small model to classify prompt difficulty (1-5) as JSON.
        Returns: (difficulty, prompt_tokens, completion_tokens).
        Falls back to 4 if len(prompt) > 600 else 3 on error.
        """
        fallback_difficulty = 4 if len(prompt) > 600 else 3
        small_model = self.tier_models["small"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a prompt difficulty evaluator for an LLM router. "
                    "Rate the complexity of answering the user prompt from 1 to 5:\n"
                    "1: Simple greeting, trivial fact, basic translation, or short lookup\n"
                    "2: Straightforward factual answer or simple summary\n"
                    "3: Moderate reasoning, structured comparison, or basic coding\n"
                    "4: Multi-step reasoning, advanced coding, or in-depth technical analysis\n"
                    "5: Complex mathematics, architecture design, edge-case debugging, or rigorous logic puzzle\n\n"
                    "Respond with ONLY a JSON object: {\"difficulty\": N} where N is 1-5. "
                    "Do not include any thought process, markdown code fences, or additional text."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            resp = self.client.chat.completions.create(
                model=small_model,
                messages=messages,
                temperature=0.0,
                max_tokens=settings.classifier_max_tokens,
            )

            raw_content = resp.choices[0].message.content or ""
            cleaned = clean_think_tags(raw_content)

            usage = getattr(resp, "usage", None)
            p_tokens = getattr(usage, "prompt_tokens", 0) or 0
            c_tokens = getattr(usage, "completion_tokens", 0) or 0

            # Parse JSON
            difficulty = None
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "difficulty" in parsed:
                    difficulty = int(parsed["difficulty"])
            except Exception:
                # Regex search fallback for {"difficulty": N}
                # Also try the raw text: a truncated <think> block would be stripped to "".
                match = re.search(r'"difficulty"\s*:\s*"?([1-5])\b', cleaned or raw_content)
                if match:
                    difficulty = int(match.group(1))

            if difficulty and 1 <= difficulty <= 5:
                return difficulty, p_tokens, c_tokens
            
            logger.warning(f"Classifier returned non-standard format: '{cleaned}'. Using fallback {fallback_difficulty}")
            return fallback_difficulty, p_tokens, c_tokens
        except Exception as e:
            logger.warning(f"Classifier call failed ({e}). Using fallback {fallback_difficulty}")
            return fallback_difficulty, 0, 0

    def route_prompt(self, difficulty: int) -> str:
        """Route difficulty score: 1-2 -> small, 3 -> mid, 4-5 -> large."""
        if difficulty <= 2:
            return "small"
        elif difficulty == 3:
            return "mid"
        else:
            return "large"

    def get_tier_price(self, tier: str) -> Tuple[float, float]:
        if tier == "small":
            return settings.price_small
        elif tier == "mid":
            return settings.price_mid
        else:
            return settings.price_large

    def ask(self, prompt: str) -> Dict[str, Any]:
        """
        Main routing pipeline:
        1. Classify prompt using small model.
        2. Route to target tier.
        3. Execute call, retrying on next larger tier if failure occurs.
        4. Calculate costs and baseline.
        5. Clean response (<think> removal).
        6. Record stats and return result.
        """
        start_time = time.perf_counter()

        # Step 1: Classify
        difficulty, class_prompt_tokens, class_comp_tokens = self.classify_difficulty(prompt)
        initial_tier = self.route_prompt(difficulty)

        # Tier escalation chain for retries
        tier_hierarchy = ["small", "mid", "large"]
        start_idx = tier_hierarchy.index(initial_tier)
        available_tiers = tier_hierarchy[start_idx:]

        answer = None
        final_tier = initial_tier
        retried = False
        last_error = None
        gen_prompt_tokens = 0
        gen_comp_tokens = 0
        chosen_model = ""

        # Step 2 & 3: Call with retry on next larger tier
        for idx, tier in enumerate(available_tiers):
            model_name = self.tier_models[tier]
            try:
                if idx > 0:
                    retried = True
                    logger.info(f"Retrying request on escalated tier '{tier}' ({model_name})...")

                resp = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=settings.max_output_tokens,
                )

                raw_answer = resp.choices[0].message.content or ""
                cleaned_answer = clean_think_tags(raw_answer)
                if not cleaned_answer:
                    # Empty after stripping <think> (e.g. truncated reasoning): escalate.
                    raise ValueError("Model returned an empty answer after cleaning")
                answer = cleaned_answer
                usage = getattr(resp, "usage", None)
                gen_prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                gen_comp_tokens = getattr(usage, "completion_tokens", 0) or 0
                final_tier = tier
                chosen_model = model_name
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Tier '{tier}' call failed with error: {e}")

        if answer is None:
            raise RuntimeError(f"All tiers failed to process prompt. Last error: {last_error}")

        latency = round(time.perf_counter() - start_time, 4)

        # Step 4: Cost calculation
        # Classifier cost (small tier pricing)
        classifier_cost = calculate_cost(class_prompt_tokens, class_comp_tokens, settings.price_small)
        # Generation cost (served tier pricing)
        tier_price = self.get_tier_price(final_tier)
        generation_cost = calculate_cost(gen_prompt_tokens, gen_comp_tokens, tier_price)
        total_cost = round(classifier_cost + generation_cost, 7)

        # Baseline cost = same generation tokens evaluated at large tier
        large_price = settings.price_large
        baseline_cost = round(calculate_cost(gen_prompt_tokens, gen_comp_tokens, large_price), 7)

        saved_amount = round(baseline_cost - total_cost, 7)
        saved_percent = round(((baseline_cost - total_cost) / baseline_cost) * 100, 2) if baseline_cost > 0 else 0.0

        # Step 5: Update stats thread-safely
        request_record = {
            "id": 0,  # assigned under lock below
            "timestamp": time.strftime("%H:%M:%S"),
            "prompt_preview": prompt[:80] + ("..." if len(prompt) > 80 else ""),
            "difficulty": difficulty,
            "initial_tier": initial_tier,
            "tier": final_tier,
            "model": chosen_model,
            "cost": total_cost,
            "baseline": baseline_cost,
            "saved_amount": saved_amount,
            "saved_percent": saved_percent,
            "latency": latency,
            "retried": retried,
            "tokens": {
                "classifier_prompt": class_prompt_tokens,
                "classifier_completion": class_comp_tokens,
                "generation_prompt": gen_prompt_tokens,
                "generation_completion": gen_comp_tokens,
            },
        }

        with self.lock:
            self.total_requests += 1
            request_record["id"] = self.total_requests
            self.total_cost += total_cost
            self.total_baseline += baseline_cost
            self.per_tier_counts[final_tier] = self.per_tier_counts.get(final_tier, 0) + 1
            self.recent_requests.appendleft(request_record)

        return {
            "answer": answer,
            "tier": final_tier,
            "model": chosen_model,
            "difficulty": difficulty,
            "cost": total_cost,
            "baseline": baseline_cost,
            "latency": latency,
            "saved_amount": saved_amount,
            "saved_percent": saved_percent,
            "retried": retried,
            "tokens": request_record["tokens"],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return total stats, savings percent, per-tier counts, and last 20 requests."""
        with self.lock:
            total_saved = self.total_baseline - self.total_cost
            savings_pct = (
                round((total_saved / self.total_baseline) * 100, 2)
                if self.total_baseline > 0
                else 0.0
            )

            recent_20 = list(self.recent_requests)[:20]

            return {
                "total_requests": self.total_requests,
                "total_cost": round(self.total_cost, 6),
                "total_baseline": round(self.total_baseline, 6),
                "total_saved": round(total_saved, 6),
                "savings_percent": savings_pct,
                "per_tier_counts": dict(self.per_tier_counts),
                "recent_requests": recent_20,
                "is_placeholder_pricing": settings.prices_are_placeholder,
                "models": dict(self.tier_models),
            }


# Singleton router instance
router = NemotronRouter()
