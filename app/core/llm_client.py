from __future__ import annotations
"""
InsightEngine AI - LLM Client
Unified async client for Groq and OpenRouter APIs.
Validates keys before sending — gives clear error instead of 'Illegal header value'.
"""
import httpx
import json
import logging
from typing import Optional
from enum import Enum

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    GROQ = "groq"
    OPENROUTER = "openrouter"


class LLMResponse:
    def __init__(self, content: str, model: str, provider: str):
        self.content = content
        self.model = model
        self.provider = provider


class LLMError(Exception):
    """Raised when LLM API call fails."""
    pass


class LLMKeyMissingError(LLMError):
    """Raised when API key is empty or unconfigured."""
    pass


class LLMClient:
    """
    Async LLM client supporting Groq and OpenRouter.
    Validates keys before every request so you get a clear error message.
    """

    def _get_headers(self, provider: LLMProvider) -> dict:
        if provider == LLMProvider.GROQ:
            key = settings.GROQ_API_KEY or ""
            if not key or key in ("your_groq_api_key", ""):
                raise LLMKeyMissingError(
                    "GROQ_API_KEY is not set in your .env file. "
                    "Get a free key at https://console.groq.com and add it to .env"
                )
            return {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
        else:
            key = settings.OPENROUTER_API_KEY or ""
            if not key or key in ("your_openrouter_api_key", ""):
                raise LLMKeyMissingError(
                    "OPENROUTER_API_KEY is not set in your .env file. "
                    "Get a key at https://openrouter.ai and add it to .env"
                )
            return {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://insightengine.ai",
                "X-Title": "InsightEngine AI",
            }

    def _get_base_url(self, provider: LLMProvider) -> str:
        if provider == LLMProvider.GROQ:
            return settings.GROQ_BASE_URL
        return settings.OPENROUTER_BASE_URL

    async def chat(
        self,
        prompt: str,
        model: str,
        provider: LLMProvider = LLMProvider.GROQ,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # Validate key first — clear error before any network call
        headers = self._get_headers(provider)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self._get_base_url(provider)}/chat/completions"

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return LLMResponse(content=content, model=model, provider=provider.value)
            except httpx.HTTPStatusError as e:
                logger.error(f"LLM HTTP error [{provider}]: {e.response.status_code} — {e.response.text[:200]}")
                raise LLMError(f"LLM API error {e.response.status_code}: {e.response.text[:100]}")
            except httpx.RequestError as e:
                logger.error(f"LLM request error [{provider}]: {e}")
                raise LLMError(f"LLM connection failed: {str(e)}")
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logger.error(f"LLM response parse error: {e}")
                raise LLMError("Failed to parse LLM response")

    async def chat_groq(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        return await self.chat(
            prompt=prompt,
            model=model or settings.CODER_A_MODEL,
            provider=LLMProvider.GROQ,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def chat_openrouter(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        return await self.chat(
            prompt=prompt,
            model=model or settings.CODER_B_MODEL,
            provider=LLMProvider.OPENROUTER,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# Singleton
llm_client = LLMClient()