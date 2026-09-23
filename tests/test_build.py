import json

from twagent.cases import load_cases
from twagent.data.build import build_dataset, to_case, to_training_example, write_dataset
from twagent.data.scenarios import build_scenarios
from twagent.data.synth import run_trajectory
from twagent.dates import resolve_date_expr
from twagent.llm import ScriptedLLM


def call(name, **args):
    body = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    return f"<tool_call>\n{body}\n</tool_call>"


def perfect_teacher(s):
    """Scripted replies that satisfy each scenario (only for the categories used below)."""
    d = resolve_date_expr(s.date_expr, s.today).isoformat() if s.date_expr else None
    if s.category == "single" and s.tools == ("get_weather",):
        return [call("get_weather", city="臺中市", date=d), "明天臺中多雲。"]
    if s.category == "clarify":
        return ["請問你要查哪一天、哪個地方呢？"]
    if s.category == "no_tool":
        return ["你好！有什麼可以幫你？"]
    return None


def make_raw(n=400, seed=1):
    raw = []
    for s in build_scenarios(n, seed=seed):
        replies = perfect_teacher(s)
        if replies is None:
            continue
        msg = f"{s.id} {s.style} {s.category} 使用者訊息 {s.date_expr or ''} {s.kind}"
        raw.append(run_trajectory(s, msg, ScriptedLLM(replies)))
    return raw


def test_training_example_marks_only_assistant_text():
    record = make_raw()[0]
    ex = to_training_example(record)
    trained = "".join(seg["text"] for seg in ex["segments"] if seg["train"])
    assert trained.endswith("<|im_end|>") and "<|im_start|>" not in trained
    assert ex["group"].startswith(ex["category"] + "|")


def test_case_round_trips_through_cases_validation():
    record = next(r for r in make_raw() if r["scenario"]["category"] == "single")
    case = to_case(record)
    assert case["expected"]["decision"] == "call"
    assert case["expected"]["calls"][0]["name"] == "get_weather"


def test_build_drops_bad_and_contaminated_records_and_reports_why():
    raw = make_raw()
    bad = run_trajectory(
        build_scenarios(400, seed=1)[0],
        "明天台中會下雨嗎",
        ScriptedLLM(["<tool_call>x</tool_call>"]),
    )
    handwritten = [raw[0]["message"]]
    split, report = build_dataset(raw + [bad], handwritten_messages=handwritten, seed=3)
    assert report["raw"] == len(raw) + 1
    assert report["rejected"] == 1 and report["reject_reasons"] == {"stopped": 1}
    assert report["contaminated"] == 1
    assert report["raw"] == (
        report["kept"] + report["rejected"] + report["contaminated"] + report["near_duplicates"]
    )
    assert sum(p["records"] for p in report["splits"].values()) == report["kept"]


def test_write_dataset_outputs_loadable_cases(tmp_path):
    split, _ = build_dataset(make_raw(), handwritten_messages=[], seed=3)
    paths = write_dataset(split, tmp_path)
    cases, errors = load_cases(paths["test_cases.jsonl"])
    assert errors == [] and len(cases) == len(split.test) > 0
    first = json.loads(paths["train.jsonl"].read_text(encoding="utf-8").splitlines()[0])
    assert {"id", "group", "category", "segments"} <= set(first)
