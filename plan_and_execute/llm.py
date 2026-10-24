"""Chat model factory: local ollama by default, Anthropic optional, a fake for offline runs."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel

from plan_and_execute.config import Settings
from plan_and_execute.fake import FakePlanModel

__all__ = ["ModelConfigError", "is_loopback_url", "make_model"]


class ModelConfigError(RuntimeError):
    """The configured provider cannot be used as configured."""


def is_loopback_url(url: str) -> bool:
    host = urlparse(url if "://" in url else f"http://{url}").hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_model(settings: Settings, *, json_mode: bool = False) -> BaseChatModel:
    """Build a chat model. ``json_mode`` asks ollama for JSON-only output (planner, replanner,
    joiner); the executor and the solver use the plain model."""
    if settings.provider == "fake":
        return FakePlanModel(latency=settings.fake_latency)
    if settings.provider == "ollama":
        if not settings.allow_remote_ollama and not is_loopback_url(settings.ollama_host):
            raise ModelConfigError(
                f"OLLAMA_HOST={settings.ollama_host!r} is not loopback; set "
                "PLAN_EXECUTE_ALLOW_REMOTE_OLLAMA=1 to send prompts to another machine"
            )
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.model,
            base_url=settings.ollama_host,
            temperature=settings.temperature,
            reasoning=False,
            format="json" if json_mode else None,
        )
    if settings.provider == "anthropic":  # pragma: no cover - needs the optional extra + a key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ModelConfigError("ANTHROPIC_API_KEY is not set")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ModelConfigError(
                "pip install 'langgraph-plan-and-execute[anthropic]' to use Anthropic"
            ) from exc
        return ChatAnthropic(model=settings.model, temperature=settings.temperature)
    raise ModelConfigError(f"unknown provider {settings.provider!r}")
