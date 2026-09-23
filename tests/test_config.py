from pathlib import Path

import pytest

from twagent.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.agent.max_steps == 5
    assert cfg.agent.parse_retries == 1
    assert cfg.agent.schema_retries == 2
    assert cfg.llm.temperature == 0.0
    assert cfg.tools.mode == "mock"


def test_to_dict_round_trips_every_section():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert set(cfg.to_dict()) == {"agent", "llm", "tools", "synth"}


def test_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("agent:\n  max_step: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_step"):
        load_config(p)


def test_unknown_section_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("agnet:\n  max_steps: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="agnet"):
        load_config(p)
