import json
from typing import Any

import httpx


class OllamaExplainer:
    """Optional local LLM for operator-facing text; never used for pixel classification."""

    def __init__(self, enabled: bool, base_url: str, model: str) -> None:
        self.enabled, self.base_url, self.model = enabled, base_url.rstrip("/"), model

    async def explain(self, detection: dict[str, Any], language: str = "ru") -> str:
        fallback = (
            "Экспериментальный screening signal: тёмная SAR-аномалия требует сравнения с ветром, "
            "предыдущим снимком, береговой маской и AIS, затем ручной и полевой проверки."
        )
        if not self.enabled:
            return fallback
        prompt = {
            "role": "environmental monitoring assistant",
            "language": language,
            "facts": {
                "area_km2": detection["area_km2"],
                "mean_confidence": detection["mean_confidence"],
                "max_confidence": detection["max_confidence"],
                "model_version": detection["model_version"],
            },
            "rules": [
                "Call it an experimental screening signal, not a confirmed spill.",
                "Mention SAR false positives and required operator/field verification.",
                "Do not invent wind, AIS, weather, source, or chemical evidence.",
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": json.dumps(prompt, ensure_ascii=False), "stream": False},
                )
            response.raise_for_status()
            return str(response.json().get("response") or fallback).strip()
        except (httpx.HTTPError, ValueError):
            return fallback
