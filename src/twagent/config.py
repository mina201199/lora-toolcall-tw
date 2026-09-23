"""Typed view of configs/*.yaml.

Unknown keys are an error rather than silently ignored: a typo in a config
file must not produce a run that looks configured but is not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 5
    parse_retries: int = 1
    schema_retries: int = 2


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "qwen"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 60.0


@dataclass(frozen=True)
class ToolsConfig:
    mode: str = "mock"


@dataclass(frozen=True)
class Config:
    agent: AgentConfig
    llm: LLMConfig
    tools: ToolsConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, section: str, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) in [{section}]: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {"agent", "llm", "tools"}
    if unknown:
        raise ValueError(f"unknown section(s): {sorted(unknown)}")
    return Config(
        agent=_build(AgentConfig, "agent", raw.get("agent")),
        llm=_build(LLMConfig, "llm", raw.get("llm")),
        tools=_build(ToolsConfig, "tools", raw.get("tools")),
    )
