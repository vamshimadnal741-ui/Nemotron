"""
Benchmark suite for Nemotron Router.
Runs 12 mixed-difficulty prompts through the router and through the large model.
Calculates cost savings %, average latency, and optional LLM-as-a-judge score (1-10).
"""

import argparse
import json
import re
import sys
import time
from typing import Any, Dict, List
from tabulate import tabulate

from app.config import settings
from app.router import router, clean_think_tags, calculate_cost

BENCHMARK_PROMPTS = [
    # 4 Easy Prompts (Difficulty 1-2)
    {
        "category": "Easy",
        "prompt": "What is the capital of Japan and what official currency is used there?",
    },
    {
        "category": "Easy",
        "prompt": "Translate 'Good morning, how are you today?' into Spanish and French.",
    },
    {
        "category": "Easy",
        "prompt": "Calculate 15% of 240, showing the single step multiplication.",
    },
    {
        "category": "Easy",
        "prompt": "Name the three primary colors in traditional pigment color theory.",
    },
    # 4 Medium Prompts (Difficulty 3)
    {
        "category": "Medium",
        "prompt": "Explain the difference between synchronous and asynchronous programming in Python with a brief example.",
    },
    {
        "category": "Medium",
        "prompt": "Compare SQL relational databases vs MongoDB document stores for an e-commerce catalog, noting read/write trade-offs.",
    },
    {
        "category": "Medium",
        "prompt": "What are the SOLID principles in object-oriented software engineering? Summarize each in one clear sentence.",
    },
    {
        "category": "Medium",
        "prompt": "Write a Python regular expression to validate an IPv4 address and explain each component of the pattern.",
    },
    # 4 Hard Prompts (Difficulty 4-5)
    {
        "category": "Hard",
        "prompt": (
            "Write a complete Python implementation of the Traveling Salesperson Problem "
            "using dynamic programming with bitmasking (Held-Karp algorithm). Explain the recurrence relation "
            "and space-time complexity O(n^2 * 2^n)."
        ),
    },
    {
        "category": "Hard",
        "prompt": (
            "Describe the Raft consensus algorithm in detail, explaining leader election, "
            "log replication consistency, and how network partitions (split-brain) are prevented."
        ),
    },
    {
        "category": "Hard",
        "prompt": (
            "Explain how to implement a thread-safe lock-free circular ring buffer in systems programming, "
            "detailing atomic memory ordering (acquire/release) semantics and ABA problem prevention."
        ),
    },
    {
        "category": "Hard",
        "prompt": (
            "Derive the backpropagation gradient updates for a 2-layer multi-layer perceptron "
            "with cross-entropy loss and softmax output layer from first principles using matrix calculus."
        ),
    },
]


def judge_answer(client, large_model: str, prompt: str, reference: str, candidate: str):
    """Ask large model to judge candidate answer against reference on 1-10 scale."""
    judge_prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert AI benchmark judge. Evaluate the Candidate answer against the Reference answer "
                "for correctness, completeness, and clarity. Score the candidate from 1 to 10 (where 10 is equal "
                "or superior to reference, 7-9 is high quality with minor omissions, and <5 has notable flaws). "
                "Respond ONLY with a JSON object: {\"score\": N} where N is an integer from 1 to 10."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {prompt}\n\n"
                f"Reference Answer (from Large model):\n{reference}\n\n"
                f"Candidate Answer (from Routed model):\n{candidate}"
            ),
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=large_model,
            messages=judge_prompt,
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content or ""
        cleaned = clean_think_tags(raw)
        
        # Parse JSON
        try:
            parsed = json.loads(cleaned)
            if "score" in parsed:
                return float(parsed["score"])
        except Exception:
            m = re.search(r'\{\s*"score"\s*:\s*(\d+(?:\.\d+)?)\s*\}', cleaned)
            if m:
                return float(m.group(1))
            m2 = re.search(r'(?:score|rating)[:\s]+(\d+(?:\.\d+)?)', cleaned, re.IGNORECASE)
            if m2:
                return float(m2.group(1))
    except Exception as e:
        print(f"      [!] Judge call failed: {e}")
    return None  # Judge failed: excluded from the average instead of faking a score


def run_benchmark(enable_judge: bool = False):
    print("=" * 75)
    print(" NEMOTRON ROUTER - 12 PROMPT BENCHMARK SUITE")
    print("=" * 75)
    
    # Ensure models are discovered
    router.discover_models()
    large_model = router.tier_models["large"]
    client = router.client

    print(f"[*] Small Tier Model : {router.tier_models['small']}")
    print(f"[*] Mid Tier Model   : {router.tier_models['mid']}")
    print(f"[*] Large Tier Model : {large_model}")
    print(f"[*] LLM-as-Judge     : {'ENABLED (Large Model Scoring 1-10)' if enable_judge else 'DISABLED (pass --judge to enable)'}")
    print("=" * 75 + "\n")

    results = []
    total_routed_cost = 0.0
    total_large_cost = 0.0
    routed_latencies = []
    large_latencies = []
    judge_scores = []

    for idx, item in enumerate(BENCHMARK_PROMPTS, 1):
        prompt = item["prompt"]
        category = item["category"]
        print(f"[{idx}/12] ({category}) {prompt[:50]}...")

        # 1. Router Call
        routed_res = router.ask(prompt)
        routed_cost = routed_res["cost"]
        routed_lat = routed_res["latency"]
        routed_tier = routed_res["tier"]
        routed_diff = routed_res["difficulty"]
        routed_ans = routed_res["answer"]

        # 2. Large Model Direct Call (Baseline)
        t0 = time.perf_counter()
        try:
            resp_large = client.chat.completions.create(
                model=large_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            large_lat = round(time.perf_counter() - t0, 4)
            large_raw = resp_large.choices[0].message.content or ""
            large_ans = clean_think_tags(large_raw)
            p_tok = getattr(resp_large.usage, "prompt_tokens", 0) or 0
            c_tok = getattr(resp_large.usage, "completion_tokens", 0) or 0
            large_cost = round(calculate_cost(p_tok, c_tok, settings.price_large), 7)
        except Exception as e:
            print(f"    [!] Large model call failed: {e}")
            large_lat = routed_lat
            large_cost = routed_res["baseline"]
            large_ans = "Large model call failed"

        # 3. Optional Judge Scoring
        score_val = "-"
        if enable_judge and large_ans != "Large model call failed":
            score = judge_answer(client, large_model, prompt, large_ans, routed_ans)
            if score is None:
                score_val = "n/a"
            else:
                score_val = f"{score:.1f}/10"
                judge_scores.append(score)

        # Accumulate
        total_routed_cost += routed_cost
        total_large_cost += large_cost
        routed_latencies.append(routed_lat)
        large_latencies.append(large_lat)

        saved_pct = ((large_cost - routed_cost) / large_cost) * 100 if large_cost > 0 else 0.0

        row = [
            idx,
            category,
            routed_tier.upper(),
            routed_diff,
            f"${routed_cost:.6f}",
            f"${large_cost:.6f}",
            f"{saved_pct:+.1f}%",
            f"{routed_lat:.2f}s",
            f"{large_lat:.2f}s",
        ]
        if enable_judge:
            row.append(score_val)
        results.append(row)

    # Print Table
    headers = ["#", "Category", "Tier", "Diff", "Routed $", "Large $", "Saved %", "R-Lat", "L-Lat"]
    if enable_judge:
        headers.append("Judge")

    print("\n" + tabulate(results, headers=headers, tablefmt="github"))

    # Overall Summary
    total_saved = total_large_cost - total_routed_cost
    overall_saved_pct = (total_saved / total_large_cost * 100) if total_large_cost > 0 else 0.0
    avg_routed_lat = sum(routed_latencies) / len(routed_latencies) if routed_latencies else 0.0
    avg_large_lat = sum(large_latencies) / len(large_latencies) if large_latencies else 0.0

    print("\n" + "=" * 75)
    print(" BENCHMARK SUMMARY")
    print("=" * 75)
    print(f" Total Routed Cost       : ${total_routed_cost:.6f}")
    print(f" Total Large-Only Cost   : ${total_large_cost:.6f}")
    print(f" Net Cost Savings        : ${total_saved:.6f} ({overall_saved_pct:.2f}% SAVED)")
    print(f" Average Routed Latency  : {avg_routed_lat:.2f}s")
    print(f" Average Large Latency   : {avg_large_lat:.2f}s")
    if enable_judge and judge_scores:
        avg_judge = sum(judge_scores) / len(judge_scores)
        print(f" Average Judge Quality   : {avg_judge:.2f} / 10.0")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nemotron Router 12-Prompt Benchmark")
    parser.add_argument("--judge", action="store_true", help="Have large model judge quality 1-10")
    args = parser.parse_args()

    run_benchmark(enable_judge=args.judge)
