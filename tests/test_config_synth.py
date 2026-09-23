from pathlib import Path

import pytest

from twagent.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_synth_section_and_prices():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.synth.model == "gpt-6-sol"
    assert cfg.synth.fallback_model == "gpt-6-luna"
    assert cfg.synth.budget_usd == 5.0
    assert cfg.synth.reasoning_effort == "none"
    price = cfg.synth.prices["gpt-6-sol"]
    assert (price.input, price.output, price.cache_read, price.cache_write) == (2.0, 10.0, 0.2, 2.5)
    assert set(cfg.synth.prices) == {cfg.synth.model, cfg.synth.fallback_model}


def test_synth_section_is_optional(tmp_path):
    p = tmp_path / "min.yaml"
    p.write_text("agent:\n  max_steps: 3\n", encoding="utf-8")
    assert load_config(p).synth.prices == {}


def test_unknown_synth_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("synth:\n  modle: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="modle"):
        load_config(p)


def test_unknown_price_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "synth:\n  prices:\n    m: {input: 1, output: 1, cache_read: 1, cache_write: 1, extra: 2}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="extra"):
        load_config(p)
