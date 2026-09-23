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
    assert "llm errors 0" in printed.splitlines()[0]
    assert {c.model for c in fake_gpt.instances} == {"gpt-6-sol"}


def test_repaired_partial_line_is_reported(fake_gpt, tmp_path, capsys):
    out = tmp_path / "pilot.jsonl"
    out.write_text('{"scenario": {"id": "s0', encoding="utf-8")
    assert cli.main(["synth", "--pilot", "5", "--out", str(out), "--config", CONFIG]) == 0
    assert "WARNING repaired" in capsys.readouterr().out
    assert len(out.read_text(encoding="utf-8").splitlines()) == 5


def test_resumed_run_accumulates_spend(fake_gpt, tmp_path):
    out = tmp_path / "raw.jsonl"
    cli.main(["synth", "--total", "10", "--out", str(out), "--config", CONFIG])
    first = json.loads((tmp_path / "raw.jsonl.cost.json").read_text(encoding="utf-8"))["spent_usd"]
    cli.main(["synth", "--total", "20", "--out", str(out), "--config", CONFIG])
    ledger = json.loads((tmp_path / "raw.jsonl.cost.json").read_text(encoding="utf-8"))
    assert ledger["spent_usd"] > first and len(ledger["runs"]) == 2


class CrashingGPT(FakeGPT):
    """A FakeGPT whose teacher dies with a non-LLM error on its 13th trajectory."""

    teacher_calls = 0

    def complete(self, messages):
        if messages[0]["role"] == "system":
            CrashingGPT.teacher_calls += 1
            if CrashingGPT.teacher_calls > 12:
                raise RuntimeError("boom")
        return super().complete(messages)


def test_ledger_survives_a_crash_and_resume_accumulates(fake_gpt, monkeypatch, tmp_path):
    out = tmp_path / "raw.jsonl"
    ledger_path = tmp_path / "raw.jsonl.cost.json"
    CrashingGPT.teacher_calls = 0
    monkeypatch.setattr(commands, "GPTClient", CrashingGPT)
    argv = ["synth", "--total", "30", "--out", str(out), "--config", CONFIG]
    with pytest.raises(RuntimeError, match="boom"):
        cli.main(argv)
    on_disk = len(out.read_text(encoding="utf-8").splitlines())
    assert on_disk == 12  # one full batch of 10, then 2 of the next
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["spent_usd"] > 0
    assert len(ledger["runs"]) == 1 and ledger["runs"][0]["written"] == on_disk
    assert ledger["runs"][0]["cost_usd"] == ledger["spent_usd"]
    assert not ledger_path.with_name(ledger_path.name + ".tmp").exists()

    monkeypatch.setattr(commands, "GPTClient", FakeGPT)
    assert cli.main(argv) == 0
    resumed = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert resumed["spent_usd"] > ledger["spent_usd"]
    assert [r["written"] for r in resumed["runs"]] == [12, 18]
    assert resumed["runs"][0] == ledger["runs"][0]


def test_ledger_is_checkpointed_after_every_batch(fake_gpt, monkeypatch, tmp_path):
    out = tmp_path / "raw.jsonl"
    seen = []
    real = commands.synthesize

    def spy(*args, on_progress, **kwargs):
        def progress():
            on_progress()
            ledger = json.loads((tmp_path / "raw.jsonl.cost.json").read_text(encoding="utf-8"))
            seen.append(ledger["runs"][-1]["written"])

        return real(*args, on_progress=progress, **kwargs)

    monkeypatch.setattr(commands, "synthesize", spy)
    cli.main(["synth", "--total", "25", "--out", str(out), "--config", CONFIG])
    assert seen == [10, 20, 25]


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


def test_projection_verdict_uses_the_effective_budget(fake_gpt, tmp_path, capsys):
    out = tmp_path / "p.jsonl"
    argv = ["synth", "--pilot", "20", "--out", str(out), "--config", CONFIG]
    cli.main([*argv, "--budget", "1000"])
    [projection] = [line for line in capsys.readouterr().out.splitlines() if "projection" in line]
    assert projection.endswith("(within budget $1000.00)")
    cli.main([*argv, "--budget", "0.0000001"])  # the pilot is already complete: no new spend
    [projection] = [line for line in capsys.readouterr().out.splitlines() if "projection" in line]
    assert projection.endswith("(EXCEEDS budget $0.00)")
