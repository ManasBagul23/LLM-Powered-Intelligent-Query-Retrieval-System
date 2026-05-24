from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List

import httpx
from fastapi import HTTPException

from app.settings import Settings


class AdaptiveGroqAPIKeyPool:
    def __init__(self, keys: List[str], base_cooldown: int = 30, max_cooldown: int = 300, cooldown_multiplier: float = 2.0):
        self.keys = keys
        self.base_cooldown = base_cooldown
        self.max_cooldown = max_cooldown
        self.cooldown_multiplier = cooldown_multiplier
        self.lock = asyncio.Lock()
        self.key_state = {
            k: {
                "cooldown_until": 0.0,
                "cooldown": float(base_cooldown),
                "failures": 0,
                "successes": 0,
                "usage_count": 0,
            }
            for k in keys
        }

    async def get_key(self) -> str:
        while True:
            async with self.lock:
                now = time.time()
                ready_keys = [k for k, s in self.key_state.items() if s["cooldown_until"] <= now]
                if ready_keys:
                    chosen = sorted(ready_keys, key=lambda k: self.key_state[k]["failures"])[0]
                    self.key_state[chosen]["usage_count"] += 1
                    return chosen

                soonest_key = min(self.keys, key=lambda k: self.key_state[k]["cooldown_until"])
                wait = max(self.key_state[soonest_key]["cooldown_until"] - now, 0.1)
            await asyncio.sleep(wait)

    async def mark_success(self, key: str) -> None:
        async with self.lock:
            s = self.key_state[key]
            s["successes"] += 1
            s["cooldown"] = float(self.base_cooldown)

    async def mark_rate_limited(self, key: str) -> None:
        async with self.lock:
            s = self.key_state[key]
            s["failures"] += 1
            s["cooldown"] = min(s["cooldown"] * self.cooldown_multiplier, float(self.max_cooldown))
            s["cooldown_until"] = time.time() + s["cooldown"]

    async def mark_failure(self, key: str) -> None:
        async with self.lock:
            self.key_state[key]["failures"] += 1


class LlmClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pool = AdaptiveGroqAPIKeyPool(settings.groq_keys)

    async def ask(self, query: str, context: str) -> Dict:
        payload = {
            "model": self.settings.groq_model,
            "messages": [
                {"role": "system", "content": "You are an expert insurance policy analyst. Respond only with JSON."},
                {"role": "user", "content": f"QUESTION: {query}\n\nPOLICY DOCUMENT:\n{context}"},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        for _ in range(len(self.settings.groq_keys) * 2):
            key = await self.pool.get_key()
            async with httpx.AsyncClient(timeout=self.settings.groq_timeout) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {key}"},
                )
            if response.status_code == 429:
                await self.pool.mark_rate_limited(key)
                continue
            if response.status_code >= 400:
                await self.pool.mark_failure(key)
                raise HTTPException(status_code=response.status_code, detail=response.text)
            await self.pool.mark_success(key)
            try:
                return json.loads(response.json()["choices"][0]["message"]["content"])
            except Exception:
                return {"answer": response.text}
        raise HTTPException(status_code=500, detail="All Groq API attempts failed")
