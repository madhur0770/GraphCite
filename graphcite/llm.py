"""LLM access behind a tiny interface so the pipeline can be tested without a model."""
from __future__ import annotations

import os
from typing import Protocol


class LLM(Protocol):
    def complete_json(self, prompt: str) -> str:
        """Return a JSON document (as text) answering ``prompt``."""

    def complete(self, prompt: str) -> str:
        """Return free text answering ``prompt``."""


class LLMError(RuntimeError):
    """The model could not be reached or returned an unusable response."""


class OllamaLLM:
    """Local model served by Ollama (https://ollama.com)."""

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("GRAPHCITE_MODEL", "llama3.2")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        try:
            from ollama import Client
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LLMError("The 'ollama' package is not installed. Run: pip install -r requirements.txt") from exc
        self._client = Client(host=self.host)

    def _chat(self, prompt: str, **kwargs) -> str:
        try:
            response = self._client.chat(model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface any transport/model error uniformly
            raise LLMError(
                f"Could not get a response from Ollama at {self.host} using model '{self.model}'. "
                f"Is the server running and the model pulled (ollama pull {self.model})? Details: {exc}"
            ) from exc
        return response["message"]["content"]

    def complete_json(self, prompt: str) -> str:
        return self._chat(prompt, format="json", options={"temperature": 0})

    def complete(self, prompt: str) -> str:
        return self._chat(prompt, options={"temperature": 0})
