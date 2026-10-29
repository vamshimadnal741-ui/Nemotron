"""
Mock Server for Nebius Token Factory (Offline Testing Only).
Emulates GET /v1/models and POST /v1/chat/completions for NVIDIA Nemotron models.
DO NOT USE THIS FOR THE FINAL HACKATHON RUN.
"""

import argparse
import re
import time
from typing import Any, Dict, List
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI(title="Nebius Token Factory Mock API")

MOCK_MODELS = [
    {
        "id": "nvidia/nemotron-mini-4b-instruct",
        "object": "model",
        "created": 1715000000,
        "owned_by": "nvidia",
    },
    {
        "id": "nvidia/llama-3.1-nemotron-70b-instruct",
        "object": "model",
        "created": 1715000000,
        "owned_by": "nvidia",
    },
    {
        "id": "nvidia/nemotron-4-340b-instruct",
        "object": "model",
        "created": 1715000000,
        "owned_by": "nvidia",
    },
]


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """Return mock NVIDIA Nemotron models."""
    return {"object": "list", "data": MOCK_MODELS}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def create_chat_completion(req: ChatCompletionRequest):
    """Simulate OpenAI chat completion with realistic tokens and think tags."""
    user_prompt = ""
    system_prompt = ""
    full_text = ""

    for m in req.messages:
        full_text += f" {m.content}"
        if m.role == "user":
            user_prompt = m.content
        elif m.role == "system":
            system_prompt = m.content

    # Determine if this is a classifier call
    is_classifier = "difficulty" in system_prompt.lower() or "difficulty" in user_prompt.lower()
    
    # Determine if this is a judge call
    is_judge = "judge" in full_text.lower() or ("score" in full_text.lower() and "1-10" in full_text.lower())

    if is_classifier:
        prompt_len = len(user_prompt)
        prompt_lower = user_prompt.lower()

        # Classify difficulty realistically
        if any(w in prompt_lower for w in ["traveling salesperson", "bitmask", "a*", "np-hard", "proof", "compiler", "mutex"]):
            diff = 5
        elif any(w in prompt_lower for w in ["nosql", "architecture", "tradeoff", "algorithm", "concurrency", "optimize"]):
            diff = 4
        elif prompt_len > 300:
            diff = 4
        elif prompt_len > 100 or any(w in prompt_lower for w in ["compare", "explain", "python", "how does"]):
            diff = 3
        elif any(w in prompt_lower for w in ["capital", "hello", "hi", "what is", "translate", "2+2", "ok"]):
            diff = 1
        else:
            diff = 2

        # Include simulated <think> tag to test tag stripping
        content = f"<think>Evaluating complexity: prompt length {prompt_len}, assigned difficulty {diff}</think>{{\"difficulty\": {diff}}}"
    elif is_judge:
        content = (
            "<think>Comparing response against large reference.</think>\n"
            "{\"score\": 9, \"reason\": \"The candidate response accurately addresses all core requirements and matches the reference quality.\"}"
        )
    else:
        # Regular chat completion response
        if "test ok" in user_prompt.lower():
            content = f"{req.model} test ok."
        elif "capital of france" in user_prompt.lower():
            content = (
                "<think>Identifying European capitals...</think>\n"
                "The capital of France is Paris, and the official language spoken is French."
            )
        else:
            content = (
                f"<think>Synthesizing answer using {req.model}...</think>\n"
                f"Response from {req.model}:\n"
                f"Regarding your query: '{user_prompt[:60]}...'\n"
                f"This answer demonstrates accurate tiered routing under simulated execution."
            )

    # Estimate realistic tokens
    prompt_tokens = max(10, len(full_text.split()) * 2)
    completion_tokens = max(5, len(content.split()) * 2)

    return {
        "id": f"chatcmpl-mock-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mock Nebius Token Factory Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8001, help="Port (default: 8001)")
    args = parser.parse_args()

    print(f"[*] Starting Nebius Mock Server on http://{args.host}:{args.port}")
    print("[*] Serving simulated NVIDIA Nemotron models at /v1/models and /v1/chat/completions")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
