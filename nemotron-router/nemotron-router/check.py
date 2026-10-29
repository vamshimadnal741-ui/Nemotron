"""
Pre-flight verification script for Nemotron Router.
Verifies API key presence, discovers models, and makes one real test call per tier.
"""

import sys
import time
from app.config import settings
from app.router import router, clean_think_tags


def run_checks():
    print("=" * 60)
    print(" NEMOTRON ROUTER - PRE-FLIGHT SYSTEM CHECK")
    print("=" * 60)

    # 1. Check API Key
    is_mock = "localhost" in settings.base_url or "127.0.0.1" in settings.base_url
    print(f"\n[1] Verifying API Configuration:")
    print(f"    - Base URL: {settings.base_url}")
    
    if not settings.has_api_key and not is_mock:
        print("    [!] ERROR: NEBIUS_API_KEY is missing or empty in .env.")
        print("        Please open .env and insert your Nebius Token Factory API key:")
        print("        NEBIUS_API_KEY=your_actual_key_here")
        sys.exit(1)
    elif is_mock:
        print("    [*] Offline Mock Mode detected (localhost endpoint).")
    else:
        print("    [+] NEBIUS_API_KEY is present.")

    # 2. Model Discovery
    print(f"\n[2] Discovering NVIDIA Nemotron Models:")
    tier_models = router.discover_models()
    
    print(f"    - Small Tier : {tier_models['small']}")
    print(f"    - Mid Tier   : {tier_models['mid']}")
    print(f"    - Large Tier : {tier_models['large']}")

    if router.discovery_error:
        print(f"    [!] Warning during discovery: {router.discovery_error}")

    # 3. Test Call Per Tier
    tiers = [
        ("small", tier_models["small"], "Say 'Small model test ok' in 5 words or less."),
        ("mid", tier_models["mid"], "Say 'Mid model test ok' in 5 words or less."),
        ("large", tier_models["large"], "Say 'Large model test ok' in 5 words or less."),
    ]

    print(f"\n[3] Testing LLM Execution Across Tiers:")
    client = router.client
    all_ok = True

    for tier_name, model_id, prompt in tiers:
        print(f"\n    Testing {tier_name.upper()} Tier [{model_id}]...")
        start_t = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            elapsed = time.perf_counter() - start_t
            raw_content = resp.choices[0].message.content or ""
            cleaned = clean_think_tags(raw_content)
            tokens = getattr(resp.usage, "total_tokens", 0)
            print(f"    [+] OK ({elapsed:.2f}s, {tokens} tokens): \"{cleaned.strip()}\"")
        except Exception as e:
            all_ok = False
            print(f"    [!] FAILED: {e}")

    print("\n" + "=" * 60)
    if all_ok:
        print(" [+] ALL PRE-FLIGHT CHECKS PASSED SUCCESSFULLY.")
        print("=" * 60)
        sys.exit(0)
    else:
        print(" [!] ONE OR MORE TIERS FAILED. Review errors above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    run_checks()
