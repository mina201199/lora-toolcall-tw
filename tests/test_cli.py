import json
from datetime import datetime, timezone
from pathlib import Path

import twagent.cli as cli
from twagent.llm import ScriptedLLM

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "configs" / "default.yaml")


def test_check_cases_ok(capsys):
    assert cli.main(["check-cases", str(ROOT / "data" / "handwritten_test.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "valid case(s)" in out and "ERROR" not in out


def test_check_cases_reports_errors(tmp_path, capsys):
    p = tmp_path / "bad.jsonl"
    p.write_text("{oops\n", encoding="utf-8")
    assert cli.main(["check-cases", str(p)]) == 1
    assert "ERROR line 1" in capsys.readouterr().out


def test_chat_runs_agent_with_mock_tools(monkeypatch, capsys):
    scripted = ScriptedLLM(
        [
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "臺中市", "date": "2026-09-24"}}\n</tool_call>',
            "明天臺中多雲。",
        ]
    )
    monkeypatch.setattr(cli, "OpenAICompatClient", lambda *a, **k: scripted)
    code = cli.main(
        ["chat", "明天台中會下雨嗎", "--today", "2026-09-23", "--config", CONFIG, "--json"]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["stopped_reason"] == "answer"
    assert out["tool_calls"][0]["name"] == "get_weather"
    assert "2026-09-23（星期三）" in scripted.calls[0][0]["content"]


def test_chat_defaults_to_taipei_date_when_today_omitted(monkeypatch, capsys):
    # 2026-09-23T20:00 UTC is already 2026-09-24T04:00 in Taipei (UTC+8): if
    # `main` used naive local/UTC time instead of Asia/Taipei, this would
    # pick the wrong day.
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 23, 20, 0, tzinfo=timezone.utc).astimezone(tz)

    scripted = ScriptedLLM(["你好"])
    monkeypatch.setattr(cli, "OpenAICompatClient", lambda *a, **k: scripted)
    monkeypatch.setattr(cli, "datetime", FrozenDateTime)
    code = cli.main(["chat", "嗨", "--config", CONFIG, "--json"])
    assert code == 0
    assert "2026-09-24" in scripted.calls[0][0]["content"]
