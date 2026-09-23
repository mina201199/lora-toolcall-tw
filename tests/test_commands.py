import json
import os
import re
from pathlib import Path

import pytest

import twagent.cli as cli
from twagent.data import commands
from twagent.gpt import Usage

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "configs" / "default.yaml")


class FakeGPT:
    """Stands in for GPTClient: answers query prompts with JSON, and
    plays a teacher that always asks a clarifying question."""

    instances: list["FakeGPT"] = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.usage = Usage()
        FakeGPT.instances.append(self)

    def complete(self, messages):
        self.usage.requests += 1
        self.usage.input_tokens += 1000
        self.usage.output_tokens += 100
        if messages[0]["role"] == "system":
            return "請問你想查哪一天呢？"
        ids = re.findall(r"id：(s\d+)", messages[0]["content"])
        return json.dumps(
            [{"id": i, "message": f"訊息 {i} 幫我查一下"} for i in ids], ensure_ascii=False
        )


@pytest.fixture
def fake_gpt(monkeypatch, tmp_path):
    FakeGPT.instances = []
    monkeypatch.setattr(commands, "GPTClient", FakeGPT)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.chdir(tmp_path)  # .env lookup and default paths stay inside tmp
    return FakeGPT


def test_load_dotenv_sets_missing_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nTWA_NEW="abc"\nTWA_OLD=file\nnot a pair\n', encoding="utf-8")
    monkeypatch.setenv("TWA_OLD", "shell")
    monkeypatch.delenv("TWA_NEW", raising=False)
    commands.load_dotenv(env)
    assert os.environ["TWA_NEW"] == "abc" and os.environ["TWA_OLD"] == "shell"


def test_pilot_writes_records_ledger_and_projection(fake_gpt, tmp_path, capsys):
    out = tmp_path / "pilot.jsonl"
    assert cli.main(["synth", "--pilot", "20", "--out", str(out), "--config", CONFIG]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    ledger = json.loads((tmp_path / "pilot.jsonl.cost.json").read_text(encoding="utf-8"))
    assert ledger["spent_usd"] > 0 and ledger["runs"][0]["written"] == 20
    printed = capsys.readouterr().out
    assert "keep rate" in printed and "projection for train_target=3000" in printed
    assert {c.model for c in fake_gpt.instances} == {"gpt-6-sol"}


def test_resumed_run_accumulates_spend(fake_gpt, tmp_path):
    out = tmp_path / "raw.jsonl"
    cli.main(["synth", "--total", "10", "--out", str(out), "--config", CONFIG])
    first = json.loads((tmp_path / "raw.jsonl.cost.json").read_text(encoding="utf-8"))["spent_usd"]
    cli.main(["synth", "--total", "20", "--out", str(out), "--config", CONFIG])
    ledger = json.loads((tmp_path / "raw.jsonl.cost.json").read_text(encoding="utf-8"))
    assert ledger["spent_usd"] > first and len(ledger["runs"]) == 2


def test_budget_stops_before_spending(fake_gpt, tmp_path, capsys):
    out = tmp_path / "raw.jsonl"
    cli.main(["synth", "--total", "10", "--budget", "0", "--out", str(out), "--config", CONFIG])
    assert "stopped: budget" in capsys.readouterr().out
    assert not out.exists() or out.read_text(encoding="utf-8") == ""


def test_unknown_model_is_refused(fake_gpt, capsys):
    assert cli.main(["synth", "--pilot", "5", "--model", "gpt-nope", "--config", CONFIG]) == 2
    assert "no price" in capsys.readouterr().out
    assert fake_gpt.instances == []


def test_missing_api_key_is_refused(fake_gpt, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY")
    assert cli.main(["synth", "--pilot", "5", "--config", CONFIG]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().out
    assert fake_gpt.instances == []


def test_model_override(fake_gpt, tmp_path):
    out = tmp_path / "p.jsonl"
    cli.main(
        [
            "synth",
            "--pilot",
            "5",
            "--model",
            "gpt-6-luna",
            "--out",
            str(out),
            "--config",
            CONFIG,
        ]
    )
    assert {c.model for c in fake_gpt.instances} == {"gpt-6-luna"}


def test_build_dataset_end_to_end(fake_gpt, tmp_path, capsys):
    raw = tmp_path / "raw.jsonl"
    cli.main(["synth", "--total", "300", "--out", str(raw), "--config", CONFIG])
    code = cli.main(
        ["build-dataset", "--raw", str(raw), "--out-dir", str(tmp_path / "ds"), "--config", CONFIG]
    )
    assert code == 0
    report = json.loads((tmp_path / "ds" / "data_report.json").read_text(encoding="utf-8"))
    # The fake teacher only ever asks a question and never calls a tool, which
    # is valid for clarify and no_tool scenarios and nothing else.
    assert report["kept"] > 0
    assert set(report["splits"]["train"]["by_category"]) == {"clarify", "no_tool"}
    assert report["config"]["synth"]["model"] == "gpt-6-sol"
    for name in ("train.jsonl", "valid.jsonl", "test_cases.jsonl"):
        assert (tmp_path / "ds" / name).exists()


def test_build_dataset_empty_raw(tmp_path, capsys):
    assert (
        cli.main(["build-dataset", "--raw", str(tmp_path / "none.jsonl"), "--config", CONFIG]) == 1
    )
