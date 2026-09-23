# Plan 1：基礎建設與 W1 驗證 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 驗證 1050 Ti 本機推論可行，並完成純邏輯層（工具、日期、解析器、Agent 迴圈、測試題格式、CLI），讓原始 Qwen 能在本機端到端跑通（B0）。

**Architecture:** `src/twagent/` 是不 import torch 的純邏輯層，全部在 CPU 上測試。提示詞由我們自己渲染（不送 `tools` 欄位），讓 `parsing.parse_output` 成為唯一判斷工具呼叫的地方；`scripts/check_template.py` 對照 llama-server 的官方模板，確認兩者逐字相同。Agent 迴圈只有 `agent.run_agent` 一份，之後的評估與 FastAPI 服務都會呼叫它。

**Tech Stack:** Python ≥3.10、uv、PyYAML、jsonschema、httpx、pytest、ruff；llama.cpp（llama-server）、Docker Desktop；Qwen3 GGUF。

**Spec:** `docs/superpowers/specs/2026-09-23-tw-toolcall-agent-design.md`

**本計畫在整體中的位置：** 規格拆成 4 份計畫，本計畫完成後再撰寫下一份（W1 的驗證結果可能改變後續做法）：
1. **Plan 1（本文件）** — 基礎建設與 W1 驗證（規格 §3、§4、§9 的迴圈部分、§10 的對應測試、§11 W1）
2. Plan 2 — 資料合成、驗證、分組切分、訓練格式化（§5）
3. Plan 3 — 訓練、匯出 GGUF、評估與消融（§6、§7）
4. Plan 4 — live 工具、FastAPI、網頁、docker-compose、fallback 到 Claude（§8、§9）

## Global Constraints

- Python `>=3.10`（本機為 3.13）；虛擬環境在 `.venv`，用 `uv venv .venv` 與 `uv pip install -p .venv -e ".[dev]"` 建立。Windows 上執行檔路徑為 `.venv/Scripts/...`。
- 純邏輯層（`src/twagent/` 目前所有模組）**不得** import `torch`、`transformers`、`unsloth`。
- 單一來源：工具定義只在 `tools/schemas.py` 的 `TOOLS`；工具呼叫解析只在 `parsing.parse_output`；Agent 迴圈只在 `agent.run_agent`；系統提示只由 `prompting.build_system_prompt` 產生。
- 解碼一律 greedy：`temperature: 0.0`。
- 列舉值一律使用正式「臺」字寫法（`臺北市`、`臺中`）。
- live 工具失敗時**不得**改用 mock 資料。
- 所有檔案讀寫明確指定 `encoding="utf-8"`。
- ruff：line-length 100，規則 `E, F, I, UP, B, SIM`（忽略 E501）。
- 每個 commit 訊息結尾加上：`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- 下載檔案（模型權重）前先向使用者確認檔名、來源與大小。

## File Structure

| 檔案 | 職責 |
|---|---|
| `pyproject.toml`、`.gitignore`、`README.md` | 套件設定、忽略模型與產出物 |
| `configs/default.yaml`、`src/twagent/config.py` | 設定檔與其型別化載入（未知鍵報錯） |
| `src/twagent/dates.py` | 相對日期、民國發票期別換算 |
| `src/twagent/tools/schemas.py` | 7 個工具的 JSON Schema 與參數驗證 |
| `src/twagent/tools/base.py` | `ToolError`、`ToolExecutor` 協定 |
| `src/twagent/tools/mock.py` | 確定性 mock 實作與 in-memory SQLite 行事曆 |
| `src/twagent/prompting.py` | 系統提示（人設＋今天日期＋Qwen 工具區塊） |
| `src/twagent/parsing.py` | `<tool_call>` 解析與標準渲染 |
| `src/twagent/llm.py` | `LLMClient` 協定、`OpenAICompatClient`、`ScriptedLLM` |
| `src/twagent/agent.py` | Agent 迴圈與失敗處理策略 |
| `src/twagent/cases.py` | 測試題格式、驗證、載入 |
| `src/twagent/cli.py` | `twagent check-cases`、`twagent chat` |
| `scripts/check_template.py` | 對照 llama-server 官方模板 |
| `data/handwritten_test.jsonl`、`data/README.md` | 手寫測試集與撰寫規則 |
| `docs/spike-w1.md` | W1 硬體驗證紀錄 |
| `.github/workflows/ci.yml` | ruff ＋ pytest（CPU） |

---

### Task 1: W1 硬體驗證（spike，丟棄式）

目的：在寫任何程式之前，確認 1050 Ti 能跑 llama-server 與 Qwen，並決定後續用哪種執行方式。本任務不產生程式碼，只產生 `docs/spike-w1.md` 紀錄。所有指令在 Git Bash 執行，工作目錄為專案根目錄 `C:\Users\User\Desktop\Lora微調專題`。

背景知識：
- GTX 1050 Ti 是 Pascal 架構（compute capability 6.1）。**CUDA 13 已移除 Pascal 支援**，所以任何以 CUDA 13 編譯的 llama.cpp 都會出現 `no kernel image is available` 之類的錯誤；必須使用 CUDA 12.x 版本的建置。驅動 582.66 可以執行 CUDA 12 程式。
- Git Bash 會自動改寫 `/models` 這類路徑，傳給 docker 的指令前面要加 `MSYS_NO_PATHCONV=1`。

**Files:**
- Create: `docs/spike-w1.md`

- [ ] **Step 1: 確認基底模型版本**

在 Hugging Face 查看 `Qwen` 組織下 ≤4B、支援工具呼叫的最新 instruct 模型。預設選擇：
- 主力：`Qwen3-4B-Instruct-2507`（非思考模式），GGUF 來源 `unsloth/Qwen3-4B-Instruct-2507-GGUF`，檔案 `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
- 備援：`Qwen3-1.7B`，GGUF 來源 `unsloth/Qwen3-1.7B-GGUF`，檔案 `Qwen3-1.7B-Q4_K_M.gguf`

若有更新的版本，記錄在 `docs/spike-w1.md` 並改用新版本。實際檔名以 repo 的 Files 頁面為準。

- [ ] **Step 2: 確認 Docker 容器能看到 GPU**

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
Expected：輸出表格中出現 `NVIDIA GeForce GTX 1050 Ti`。若失敗，記錄錯誤訊息，直接跳到 Step 5 的方案 B。

- [ ] **Step 3: 下載模型（先向使用者確認）**

向使用者說明：將從 Hugging Face 下載 `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`（約 2.5 GB）到 `models/`，取得同意後執行：
```bash
uvx --from huggingface_hub hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir models
```
Expected：`models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` 存在。`models/` 已在 `.gitignore`（Task 2 建立；若 Task 2 尚未做，先手動建立 `.gitignore` 並寫入 `models/`）。

- [ ] **Step 4: 方案 A — Docker 版 llama-server**

```bash
MSYS_NO_PATHCONV=1 docker run --rm --gpus all -p 8080:8080 -v "$(pwd -W)/models:/models" ghcr.io/ggml-org/llama.cpp:server-cuda -m /models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --jinja -ngl 99 -c 4096
```
（用 Bash 工具的 `run_in_background` 執行。）觀察啟動 log：
- 成功：出現 `offloaded 37/37 layers to GPU`（層數依模型而定，重點是全部 offload）以及 `server is listening`
- 失敗：出現 `no kernel image is available`、`CUDA error` 或 `unsupported gpu architecture` → 停止容器，改做 Step 5

先用 `docker run --rm ghcr.io/ggml-org/llama.cpp:server-cuda --version` 查看它用哪個 CUDA 版本建置，並記錄。

- [ ] **Step 5: 方案 A 失敗時，依序嘗試 B → C → D**

從 https://github.com/ggml-org/llama.cpp/releases 最新 release 下載（每個檔案下載前都先向使用者確認）：
- **B（Windows 原生 CUDA 12）**：`llama-<build>-bin-win-cuda-12.x-x64.zip` 以及同版本的 `cudart-llama-bin-win-cuda-12.x-x64.zip`，解壓到 `C:\tools\llama.cpp\`
- **C（Windows 原生 Vulkan）**：`llama-<build>-bin-win-vulkan-x64.zip`
- **D（CPU）**：`llama-<build>-bin-win-cpu-x64.zip`，最後手段

啟動（以 B 為例）：
```bash
/c/tools/llama.cpp/llama-server.exe -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --host 127.0.0.1 --port 8080 --jinja -ngl 99 -c 4096
```
若使用原生方案，Plan 4 的 docker-compose 會改成讓容器連到 `host.docker.internal:8080`。

- [ ] **Step 6: 冒煙測試與效能量測**

```bash
curl -s localhost:8080/health
curl -s localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"用一句話介紹台北"}],"temperature":0,"max_tokens":64}'
```
Expected：`/health` 回 `{"status":"ok"}`；第二個指令回傳繁體（或簡體）中文句子，以及 `timings.predicted_per_second`（每秒 token 數）。

伺服器執行時另開終端執行 `nvidia-smi`，記錄顯存用量。

- [ ] **Step 7: 確認 LoRA 動態切換的支援**

```bash
curl -s localhost:8080/lora-adapters
```
Expected：回傳 `[]`（端點存在，只是還沒載入 adapter）。再執行 `llama-server --help`（或 `docker run --rm ghcr.io/ggml-org/llama.cpp:server-cuda --help`），確認有 `--lora` 與 `--lora-init-without-apply` 參數。實際的每請求 scale 測試要等 Plan 3 有 adapter 之後才做。

- [ ] **Step 8: 1.7B 備援量測**

徵得使用者同意下載 `Qwen3-1.7B-Q4_K_M.gguf`（約 1.1 GB）後，用 Step 4/5 成功的方案重跑 Step 6，記錄 tokens/s 與顯存。

- [ ] **Step 9: 寫下紀錄**

建立 `docs/spike-w1.md`，填入實際量到的數值（不可填估計值）：

```markdown
# W1 硬體驗證紀錄

日期：（執行日期）

## 結論
- 採用的執行方式：A / B / C / D（擇一）
- 原因：

## 模型
| 模型 | GGUF 檔 | 顯存用量 | tokens/s（生成） | 全部層 offload 到 GPU？ |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 Q4_K_M | | | | |
| Qwen3-1.7B Q4_K_M | | | | |

## 各方案結果
| 方案 | 結果 | 錯誤訊息／備註 |
|---|---|---|
| A Docker CUDA | | |
| B 原生 CUDA 12 | | |
| C 原生 Vulkan | | |

## LoRA 支援
- `/lora-adapters` 端點：
- `--lora-init-without-apply`：

## 模板對照（Task 12 填寫）

## B0 冒煙結果（Task 12 填寫）
```

- [ ] **Step 10: Commit**

```bash
git add docs/spike-w1.md
git commit -m "docs: record W1 hardware spike results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

**決策閘門：** 若 A–C 全部失敗（只能用 CPU），停下來回報使用者，因為規格 §8 的部署前提改變了，需要重新討論。

---

### Task 2: 專案骨架與設定檔

**Files:**
- Create: `pyproject.toml`、`.gitignore`、`README.md`、`configs/default.yaml`、`src/twagent/__init__.py`、`src/twagent/config.py`、`.github/workflows/ci.yml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `twagent.config.load_config(path) -> Config`；`Config.agent: AgentConfig(max_steps: int, parse_retries: int, schema_retries: int)`、`Config.llm: LLMConfig(base_url, model, temperature, max_tokens, timeout_s)`、`Config.tools: ToolsConfig(mode)`；`Config.to_dict() -> dict`。未知鍵或未知區段拋 `ValueError`。

- [ ] **Step 1: 建立套件設定與空套件**

`pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "twagent"
version = "0.1.0"
description = "LoRA fine-tuned Traditional Chinese function-calling agent for Taiwan daily-life tools, deployable on a 4 GB GPU."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }

# The pure layer -- config, dates, tools, parsing, agent -- must install and
# test on any laptop without a GPU. Anything heavy goes behind an extra.
dependencies = [
    "pyyaml>=6.0",
    "jsonschema>=4.21",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.6"]

[project.scripts]
twagent = "twagent.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = ["live: needs network or a running llama-server; skipped in CI"]

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]
```

`.gitignore`：

```text
__pycache__/
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/
.coverage
# Model weights and generated artefacts never go into git.
models/
outputs/
data/synthetic/
logs/
.env
```

`README.md`：

````markdown
# lora-toolcall-tw

繁體中文 Function Calling Agent（臺灣生活助理）：用 QLoRA 微調小型 Qwen 模型，
量化後在 4 GB 顯卡上本機部署。設計見 `docs/superpowers/specs/`。

```bash
uv venv .venv
uv pip install -p .venv -e ".[dev]"
.venv/Scripts/python -m pytest          # Windows；Linux/macOS 用 .venv/bin/python
```
````

`configs/default.yaml`：

```yaml
# Default run configuration.
#
# Every number that ends up in a report comes from this file, and the file is
# embedded verbatim in each report so a result can be traced back to the
# settings that produced it.

agent:
  max_steps: 5          # hard cap on LLM calls per request (loop guard)
  parse_retries: 1      # re-ask once when <tool_call> is malformed
  schema_retries: 2     # schema-invalid calls tolerated before giving up

llm:
  base_url: http://localhost:8080/v1
  model: qwen
  # Greedy. Tool calling wants the single most likely call, not variety.
  temperature: 0.0
  max_tokens: 512
  timeout_s: 60.0

tools:
  mode: mock            # mock | live (live arrives with the deployment plan)
```

`src/twagent/__init__.py`：

```python
"""Traditional Chinese function-calling agent for Taiwan daily-life tools."""

__version__ = "0.1.0"
```

- [ ] **Step 2: 建立虛擬環境並安裝**

```bash
uv venv .venv
uv pip install -p .venv -e ".[dev]"
```
Expected：安裝成功，`.venv/Scripts/python -c "import twagent; print(twagent.__version__)"` 印出 `0.1.0`。

- [ ] **Step 3: 寫失敗的測試**

`tests/test_config.py`：

```python
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
    assert set(cfg.to_dict()) == {"agent", "llm", "tools"}


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
```

- [ ] **Step 4: 執行測試，確認失敗**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: FAIL，錯誤為 `ModuleNotFoundError: No module named 'twagent.config'`

- [ ] **Step 5: 寫實作**

`src/twagent/config.py`：

```python
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
```

- [ ] **Step 6: 執行測試並檢查 lint**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: 4 passed

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: 通過

- [ ] **Step 7: 加上 CI**

`.github/workflows/ci.yml`：

```yaml
name: CI

# The pure layer needs no GPU, no model download and no network, so every
# push runs lint plus the full test suite on plain GitHub runners.

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .

  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install -e ".[dev]"
      - run: pytest -q -m "not live"
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore README.md configs src tests .github
git commit -m "chore: scaffold twagent package with typed config and CI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 日期換算 `dates.py`

**Files:**
- Create: `src/twagent/dates.py`
- Test: `tests/test_dates.py`

**Interfaces:**
- Produces: `format_today(d: date) -> str`（`2026-09-23（星期三）`）、`resolve_date_expr(expr: str, today: date) -> date`（不支援的說法拋 `ValueError`）、`roc_year(d) -> int`、`invoice_period(d) -> str`（`115-09-10`）、`previous_invoice_period(d) -> str`。週的範圍是週一到週日。

- [ ] **Step 1: 寫失敗的測試**

`tests/test_dates.py`：

```python
from datetime import date

import pytest

from twagent.dates import (
    format_today,
    invoice_period,
    previous_invoice_period,
    resolve_date_expr,
    roc_year,
)

WED = date(2026, 9, 23)  # a Wednesday
SUN = date(2026, 9, 27)


def test_format_today():
    assert format_today(WED) == "2026-09-23（星期三）"
    assert format_today(SUN) == "2026-09-27（星期日）"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("今天", date(2026, 9, 23)),
        ("明天", date(2026, 9, 24)),
        ("後天", date(2026, 9, 25)),
        ("大後天", date(2026, 9, 26)),
        ("昨天", date(2026, 9, 22)),
        ("3天後", date(2026, 9, 26)),
        ("十天後", date(2026, 10, 3)),
        ("兩天之後", date(2026, 9, 25)),
        ("這週五", date(2026, 9, 25)),
        ("本週一", date(2026, 9, 21)),  # already past, still this week
        ("這禮拜天", date(2026, 9, 27)),
        ("下週五", date(2026, 10, 2)),
        ("下禮拜一", date(2026, 9, 28)),
        ("下個星期日", date(2026, 10, 4)),
        ("週五", date(2026, 9, 25)),
        ("星期三", date(2026, 9, 23)),  # today counts
        ("禮拜二", date(2026, 9, 29)),  # already past -> next week
    ],
)
def test_resolve_date_expr(expr, expected):
    assert resolve_date_expr(expr, WED) == expected


def test_resolve_crosses_year_boundary():
    assert resolve_date_expr("明天", date(2026, 12, 31)) == date(2027, 1, 1)
    assert resolve_date_expr("下週一", date(2026, 12, 30)) == date(2027, 1, 4)


def test_next_week_from_sunday_is_the_following_week():
    # Sunday belongs to the Mon-Sun week that started six days earlier.
    assert resolve_date_expr("下週一", SUN) == date(2026, 9, 28)


def test_unsupported_expression_raises():
    with pytest.raises(ValueError, match="unsupported"):
        resolve_date_expr("過陣子", WED)


def test_roc_year():
    assert roc_year(WED) == 115


@pytest.mark.parametrize(
    ("d", "period", "previous"),
    [
        (date(2026, 9, 23), "115-09-10", "115-07-08"),
        (date(2026, 10, 1), "115-09-10", "115-07-08"),
        (date(2026, 1, 15), "115-01-02", "114-11-12"),
        (date(2026, 12, 31), "115-11-12", "115-09-10"),
    ],
)
def test_invoice_periods(d, period, previous):
    assert invoice_period(d) == period
    assert previous_invoice_period(d) == previous
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `.venv/Scripts/python -m pytest tests/test_dates.py -v`
Expected: FAIL，錯誤為 `ModuleNotFoundError: No module named 'twagent.dates'`

- [ ] **Step 3: 寫實作**

`src/twagent/dates.py`：

```python
"""Date arithmetic the model is expected to do, done deterministically.

Used to build the system prompt (today's date is injected, never read from
the clock inside the agent) and, later, to check synthetic labels: a sample
that says "下週五" must carry the date this module computes, or it is dropped.

Weeks run Monday to Sunday.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

WEEKDAY_ZH = "一二三四五六日"
_WEEKDAY_INDEX = {ch: i for i, ch in enumerate(WEEKDAY_ZH)} | {"天": 6}
_SMALL_NUMBERS = {
    "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}  # fmt: skip
_FIXED_OFFSETS = {"昨天": -1, "今天": 0, "明天": 1, "後天": 2, "大後天": 3}

_THIS_WEEK = re.compile(r"^(?:本|這)(?:週|周|禮拜|星期)([一二三四五六日天])$")
_NEXT_WEEK = re.compile(r"^下(?:週|周|禮拜|星期|個禮拜|個星期)([一二三四五六日天])$")
_BARE_WEEKDAY = re.compile(r"^(?:週|周|禮拜|星期)([一二三四五六日天])$")
_DAYS_LATER = re.compile(r"^(\d+|[一二兩三四五六七八九十])天(?:後|之後)$")


def format_today(d: date) -> str:
    """`2026-09-23（星期三）` -- the form injected into the system prompt."""
    return f"{d.isoformat()}（星期{WEEKDAY_ZH[d.weekday()]}）"


def resolve_date_expr(expr: str, today: date) -> date:
    """Resolve a relative Chinese date expression against `today`.

    Raises ValueError for anything outside the supported vocabulary, so a
    generator that invents a new phrasing is caught instead of guessed at.
    """
    s = expr.strip()
    if s in _FIXED_OFFSETS:
        return today + timedelta(days=_FIXED_OFFSETS[s])
    if m := _DAYS_LATER.match(s):
        n = m.group(1)
        return today + timedelta(days=int(n) if n.isdigit() else _SMALL_NUMBERS[n])
    monday = today - timedelta(days=today.weekday())
    if m := _THIS_WEEK.match(s):
        return monday + timedelta(days=_WEEKDAY_INDEX[m.group(1)])
    if m := _NEXT_WEEK.match(s):
        return monday + timedelta(days=7 + _WEEKDAY_INDEX[m.group(1)])
    if m := _BARE_WEEKDAY.match(s):
        # The next such day on or after today.
        delta = (_WEEKDAY_INDEX[m.group(1)] - today.weekday()) % 7
        return today + timedelta(days=delta)
    raise ValueError(f"unsupported date expression: {expr!r}")


def roc_year(d: date) -> int:
    return d.year - 1911


def invoice_period(d: date) -> str:
    """Taiwan uniform-invoice period containing `d`, e.g. `115-09-10`.

    Periods are two months long and always start on an odd month.
    """
    start = d.month if d.month % 2 == 1 else d.month - 1
    return f"{roc_year(d)}-{start:02d}-{start + 1:02d}"


def previous_invoice_period(d: date) -> str:
    """The period before the one containing `d` ("上一期")."""
    start = d.month if d.month % 2 == 1 else d.month - 1
    if start == 1:
        return f"{roc_year(d) - 1}-11-12"
    return f"{roc_year(d)}-{start - 2:02d}-{start - 1:02d}"
```

- [ ] **Step 4: 執行測試，確認通過；lint 與格式檢查**

Run: `.venv/Scripts/python -m pytest tests/test_dates.py -v`
Expected: 全部 PASS

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: `All checks passed!` 且沒有需要重新格式化的檔案

- [ ] **Step 5: Commit**

```bash
git add src/twagent/dates.py tests/test_dates.py
git commit -m "feat: deterministic relative-date and invoice-period arithmetic

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 工具定義 `tools/schemas.py`

**Files:**
- Create: `src/twagent/tools/__init__.py`（空檔）、`src/twagent/tools/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces: `TOOLS: list[dict]`（OpenAI function 格式，7 個工具）、`CITIES`（22 縣市）、`THSR_STATIONS`（12 站）、`TRA_TRAIN_TYPES`、`tool_names() -> list[str]`、`get_tool(name) -> dict`、`validate_arguments(name: str, arguments: Any) -> list[str]`（空 list 代表合法；訊息會原樣回給模型看）。

先建立空的 `src/twagent/tools/__init__.py`（不含任何內容）。Step 2 的錯誤訊息在建立 `__init__.py` 之後會變成 `No module named 'twagent.tools.schemas'`，兩者皆可。

- [ ] **Step 1: 寫失敗的測試**

`tests/test_schemas.py`：

```python
import pytest

from twagent.tools.schemas import CITIES, THSR_STATIONS, TOOLS, tool_names, validate_arguments


def test_seven_tools_with_unique_names():
    assert tool_names() == [
        "get_weather",
        "search_thsr",
        "search_tra",
        "find_youbike",
        "check_invoice",
        "calendar_add",
        "calendar_list",
    ]
    assert len({t["function"]["name"] for t in TOOLS}) == len(TOOLS)


def test_reference_lists():
    assert len(CITIES) == 22 and len(set(CITIES)) == 22
    assert len(THSR_STATIONS) == 12
    # Official spelling only: the model must learn to normalise 台 -> 臺.
    assert not any("台" in c for c in CITIES + THSR_STATIONS)


def test_every_tool_forbids_extra_arguments():
    for t in TOOLS:
        assert t["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("get_weather", {"city": "臺中市", "date": "2026-09-24"}),
        ("search_thsr", {"origin": "臺北", "destination": "左營", "date": "2026-09-24"}),
        (
            "search_thsr",
            {
                "origin": "臺北",
                "destination": "臺南",
                "date": "2026-09-24",
                "depart_after": "07:30",
            },
        ),
        (
            "search_tra",
            {"origin": "花蓮", "destination": "臺北", "date": "2026-09-24", "train_type": "自強"},
        ),
        ("find_youbike", {"city": "臺北市", "near": "臺大"}),
        ("check_invoice", {"number": "12345678", "period": "115-07-08"}),
        (
            "calendar_add",
            {"title": "看牙醫", "start": "2026-09-24T14:00", "end": "2026-09-24T15:00"},
        ),
        ("calendar_list", {"date": "2026-09-24"}),
    ],
)
def test_valid_calls(name, args):
    assert validate_arguments(name, args) == []


@pytest.mark.parametrize(
    ("name", "args", "fragment"),
    [
        ("get_weather", {"city": "台中", "date": "2026-09-24"}, "city"),
        ("get_weather", {"city": "臺中市"}, "'date' is a required property"),
        ("get_weather", {"city": "臺中市", "date": "明天"}, "date"),
        ("get_weather", {"city": "臺中市", "date": "2026-09-24", "unit": "C"}, "unit"),
        ("search_thsr", {"origin": "高雄", "destination": "臺北", "date": "2026-09-24"}, "origin"),
        (
            "search_thsr",
            {
                "origin": "臺北",
                "destination": "左營",
                "date": "2026-09-24",
                "depart_after": "25:00",
            },
            "depart_after",
        ),
        ("check_invoice", {"number": "1234567", "period": "115-07-08"}, "number"),
        ("check_invoice", {"number": "12345678", "period": "115-08-09"}, "period"),
        (
            "calendar_add",
            {"title": "x", "start": "2026-09-24 14:00", "end": "2026-09-24T15:00"},
            "start",
        ),
    ],
)
def test_invalid_calls(name, args, fragment):
    errors = validate_arguments(name, args)
    assert errors, "expected at least one error"
    assert any(fragment in e for e in errors), errors


def test_unknown_tool():
    errors = validate_arguments("book_flight", {})
    assert len(errors) == 1 and "unknown tool" in errors[0]


def test_non_object_arguments():
    assert "JSON object" in validate_arguments("get_weather", ["臺中市"])[0]
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `.venv/Scripts/python -m pytest tests/test_schemas.py -v`
Expected: FAIL，錯誤為 `ModuleNotFoundError: No module named 'twagent.tools'`

- [ ] **Step 3: 寫實作**

`src/twagent/tools/schemas.py`：

```python
"""The single definition of every tool the agent can call.

Synthesis, validation, evaluation, the system prompt and the server all read
from `TOOLS`. There is no second copy to drift.

Enum values use the official 臺 spelling; teaching the model to normalise
"台北" -> "臺北市" is part of the task.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

CITIES = [
    "臺北市", "新北市", "桃園市", "臺中市", "臺南市", "高雄市",
    "基隆市", "新竹市", "嘉義市", "新竹縣", "苗栗縣", "彰化縣",
    "南投縣", "雲林縣", "嘉義縣", "屏東縣", "宜蘭縣", "花蓮縣",
    "臺東縣", "澎湖縣", "金門縣", "連江縣",
]  # fmt: skip

THSR_STATIONS = [
    "南港", "臺北", "板橋", "桃園", "新竹", "苗栗",
    "臺中", "彰化", "雲林", "嘉義", "臺南", "左營",
]  # fmt: skip

TRA_TRAIN_TYPES = ["自強", "莒光", "區間快", "區間"]

DATE = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$", "description": "YYYY-MM-DD"}
TIME = {
    "type": "string",
    "pattern": r"^([01]\d|2[0-3]):[0-5]\d$",
    "description": "HH:MM（24 小時制）",
}
DATETIME = {
    "type": "string",
    "pattern": r"^\d{4}-\d{2}-\d{2}T([01]\d|2[0-3]):[0-5]\d$",
    "description": "YYYY-MM-DDTHH:MM（臺灣時間）",
}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS: list[dict[str, Any]] = [
    _tool(
        "get_weather",
        "查詢臺灣某縣市某一天的天氣預報。",
        {"city": {"type": "string", "enum": CITIES}, "date": DATE},
        ["city", "date"],
    ),
    _tool(
        "search_thsr",
        "查詢高鐵班次。",
        {
            "origin": {"type": "string", "enum": THSR_STATIONS},
            "destination": {"type": "string", "enum": THSR_STATIONS},
            "date": DATE,
            "depart_after": TIME,
        },
        ["origin", "destination", "date"],
    ),
    _tool(
        "search_tra",
        "查詢臺鐵（火車）班次。",
        {
            "origin": {"type": "string", "minLength": 1, "description": "臺鐵車站名稱"},
            "destination": {"type": "string", "minLength": 1, "description": "臺鐵車站名稱"},
            "date": DATE,
            "depart_after": TIME,
            "train_type": {"type": "string", "enum": TRA_TRAIN_TYPES},
        },
        ["origin", "destination", "date"],
    ),
    _tool(
        "find_youbike",
        "查詢某地點附近的 YouBike 站點與可借車輛數。",
        {
            "city": {"type": "string", "enum": CITIES},
            "near": {"type": "string", "minLength": 1, "description": "地標、路名或區域"},
        },
        ["city", "near"],
    ),
    _tool(
        "check_invoice",
        "統一發票對獎。",
        {
            "number": {"type": "string", "pattern": r"^\d{8}$", "description": "8 碼發票號碼"},
            "period": {
                "type": "string",
                "pattern": r"^\d{3}-(01-02|03-04|05-06|07-08|09-10|11-12)$",
                "description": "民國年-起月-迄月，例如 115-07-08",
            },
        },
        ["number", "period"],
    ),
    _tool(
        "calendar_add",
        "在使用者的行事曆新增一筆行程。",
        {
            "title": {"type": "string", "minLength": 1},
            "start": DATETIME,
            "end": DATETIME,
            "location": {"type": "string"},
        },
        ["title", "start", "end"],
    ),
    _tool(
        "calendar_list",
        "列出使用者行事曆某一天的行程。",
        {"date": DATE},
        ["date"],
    ),
]

_BY_NAME = {t["function"]["name"]: t for t in TOOLS}
_VALIDATORS = {
    name: Draft202012Validator(t["function"]["parameters"]) for name, t in _BY_NAME.items()
}


def tool_names() -> list[str]:
    return list(_BY_NAME)


def get_tool(name: str) -> dict[str, Any]:
    return _BY_NAME[name]


def validate_arguments(name: str, arguments: Any) -> list[str]:
    """Return human-readable problems with a call; empty list means valid.

    The messages go back to the model verbatim as a tool result, so they are
    meant to be read by it.
    """
    if name not in _VALIDATORS:
        return [f"unknown tool: {name!r}; available: {', '.join(_BY_NAME)}"]
    if not isinstance(arguments, dict):
        return [f"arguments must be a JSON object, got {type(arguments).__name__}"]
    errors = sorted(_VALIDATORS[name].iter_errors(arguments), key=lambda e: list(e.path))
    out = []
    for e in errors:
        where = ".".join(str(p) for p in e.path) or "(root)"
        out.append(f"{where}: {e.message}")
    return out
```

- [ ] **Step 4: 執行測試，確認通過；lint 與格式檢查**

Run: `.venv/Scripts/python -m pytest tests/test_schemas.py -v`
Expected: 全部 PASS

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: `All checks passed!` 且沒有需要重新格式化的檔案

- [ ] **Step 5: Commit**

```bash
git add src/twagent/tools tests/test_schemas.py
git commit -m "feat: single-source JSON Schemas for the seven tools

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Mock 工具 `tools/base.py`、`tools/mock.py`

**Files:**
- Create: `src/twagent/tools/base.py`、`src/twagent/tools/mock.py`
- Test: `tests/test_mock_tools.py`

**Interfaces:**
- Consumes: `tools.schemas.THSR_STATIONS`、`tools.schemas.tool_names()`
- Produces: `ToolError(Exception)`；`ToolExecutor` 協定（`call(name: str, arguments: dict) -> dict`）；`MockTools(failures: dict[str, str] | None = None)`，方法 `call(name, arguments) -> dict`、`events() -> list[dict]`（行事曆最終狀態，每筆含 `id, title, start, end, location`）。相同呼叫永遠回傳相同結果（行事曆除外）。

- [ ] **Step 1: 寫失敗的測試**

`tests/test_mock_tools.py`：

```python
import json

import pytest

from twagent.tools.base import ToolError
from twagent.tools.mock import MockTools
from twagent.tools.schemas import tool_names

SAMPLE_CALLS = {
    "get_weather": {"city": "臺中市", "date": "2026-09-24"},
    "search_thsr": {
        "origin": "臺北",
        "destination": "左營",
        "date": "2026-09-24",
        "depart_after": "08:00",
    },
    "search_tra": {"origin": "花蓮", "destination": "臺北", "date": "2026-09-24"},
    "find_youbike": {"city": "臺北市", "near": "臺大"},
    "check_invoice": {"number": "12345678", "period": "115-07-08"},
    "calendar_add": {"title": "看牙醫", "start": "2026-09-24T14:00", "end": "2026-09-24T15:00"},
    "calendar_list": {"date": "2026-09-24"},
}


def test_every_tool_has_a_mock():
    assert set(SAMPLE_CALLS) == set(tool_names())
    tools = MockTools()
    for name, args in SAMPLE_CALLS.items():
        result = tools.call(name, args)
        json.dumps(result, ensure_ascii=False)  # must be serialisable


@pytest.mark.parametrize("name", [n for n in SAMPLE_CALLS if not n.startswith("calendar")])
def test_same_call_same_result_across_instances(name):
    assert MockTools().call(name, SAMPLE_CALLS[name]) == MockTools().call(name, SAMPLE_CALLS[name])


def test_different_arguments_give_different_weather():
    a = MockTools().call("get_weather", {"city": "臺中市", "date": "2026-09-24"})
    b = MockTools().call("get_weather", {"city": "臺北市", "date": "2026-09-24"})
    assert a != b


def test_thsr_respects_depart_after():
    result = MockTools().call("search_thsr", SAMPLE_CALLS["search_thsr"])
    assert len(result["trains"]) == 3
    assert all(t["depart"] >= "08:00" for t in result["trains"])


def test_same_station_is_a_tool_error():
    with pytest.raises(ToolError, match="起訖站相同"):
        MockTools().call(
            "search_thsr", {"origin": "臺北", "destination": "臺北", "date": "2026-09-24"}
        )


def test_calendar_state_is_inspectable():
    tools = MockTools()
    tools.call("calendar_add", SAMPLE_CALLS["calendar_add"])
    tools.call(
        "calendar_add",
        {
            "title": "開會",
            "start": "2026-09-25T10:00",
            "end": "2026-09-25T11:00",
            "location": "公司",
        },
    )
    assert [e["title"] for e in tools.events()] == ["看牙醫", "開會"]
    listed = tools.call("calendar_list", {"date": "2026-09-24"})
    assert [e["title"] for e in listed["events"]] == ["看牙醫"]


def test_calendar_rejects_end_before_start():
    with pytest.raises(ToolError, match="結束時間"):
        MockTools().call(
            "calendar_add", {"title": "x", "start": "2026-09-24T15:00", "end": "2026-09-24T14:00"}
        )


def test_injected_failure():
    tools = MockTools(failures={"search_thsr": "連線逾時"})
    with pytest.raises(ToolError, match="連線逾時"):
        tools.call("search_thsr", SAMPLE_CALLS["search_thsr"])
    # Other tools are unaffected.
    tools.call("get_weather", SAMPLE_CALLS["get_weather"])
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `.venv/Scripts/python -m pytest tests/test_mock_tools.py -v`
Expected: FAIL，錯誤為 `ModuleNotFoundError: No module named 'twagent.tools.base'`

- [ ] **Step 3: 寫實作**

`src/twagent/tools/base.py`：

```python
"""The contract between the agent loop and whatever executes tools."""

from __future__ import annotations

from typing import Any, Protocol


class ToolError(Exception):
    """A tool ran and failed (timeout, bad upstream data, impossible request).

    The agent reports the message back to the model as the tool result. It
    never substitutes data from another executor: a live failure must not be
    papered over with mock output.
    """


class ToolExecutor(Protocol):
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one already-validated call and return a JSON-serialisable result."""
        ...
```

`src/twagent/tools/mock.py`：

```python
"""Deterministic stand-ins for every tool.

Evaluation always runs against these, so a score can be reproduced exactly:
the same call returns the same result, seeded from a hash of the call itself.
The calendar is a real (in-memory) SQLite database so end-to-end tasks can be
scored by inspecting final state rather than by trusting the model's reply.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from datetime import datetime
from typing import Any

from twagent.tools.base import ToolError
from twagent.tools.schemas import THSR_STATIONS

_CONDITIONS = ["晴", "多雲", "陰", "短暫陣雨", "午後雷陣雨"]


def _rng(name: str, arguments: dict[str, Any]) -> random.Random:
    key = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True)
    return random.Random(int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16))


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60 % 24:02d}:{minutes % 60:02d}"


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


class MockTools:
    """`ToolExecutor` backed by seeded fake data and an in-memory calendar.

    `failures` maps a tool name to an error message; calling that tool raises
    `ToolError` with it. Used by the "tool error" test cases.
    """

    def __init__(self, failures: dict[str, str] | None = None) -> None:
        self.failures = dict(failures or {})
        self._db = sqlite3.connect(":memory:")
        self._db.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY, title TEXT, start TEXT, end TEXT, location TEXT)"
        )

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in self.failures:
            raise ToolError(self.failures[name])
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            raise ToolError(f"unknown tool: {name}")
        return handler(dict(arguments), _rng(name, arguments))

    def events(self) -> list[dict[str, Any]]:
        """Final calendar state, for end-to-end scoring."""
        rows = self._db.execute("SELECT id, title, start, end, location FROM events ORDER BY id")
        return [
            {"id": r[0], "title": r[1], "start": r[2], "end": r[3], "location": r[4]} for r in rows
        ]

    # -- handlers -----------------------------------------------------------

    def _get_weather(self, a: dict, rng: random.Random) -> dict:
        low = rng.randint(14, 27)
        return {
            "city": a["city"],
            "date": a["date"],
            "condition": rng.choice(_CONDITIONS),
            "temp_min": low,
            "temp_max": low + rng.randint(3, 8),
            "rain_probability": rng.choice([0, 10, 20, 30, 50, 70, 90]),
        }

    def _search_thsr(self, a: dict, rng: random.Random) -> dict:
        if a["origin"] == a["destination"]:
            raise ToolError("起訖站相同")
        hops = abs(THSR_STATIONS.index(a["origin"]) - THSR_STATIONS.index(a["destination"]))
        ride = 10 + hops * 9
        first = max(_minutes(a.get("depart_after", "06:00")), 6 * 60)
        first += rng.randint(0, 25)
        trains = []
        for i in range(3):
            dep = first + i * 30
            trains.append(
                {
                    "train_no": str(rng.randint(100, 1699)),
                    "depart": _hhmm(dep),
                    "arrive": _hhmm(dep + ride),
                }
            )
        return {
            "origin": a["origin"],
            "destination": a["destination"],
            "date": a["date"],
            "trains": trains,
        }

    def _search_tra(self, a: dict, rng: random.Random) -> dict:
        if a["origin"] == a["destination"]:
            raise ToolError("起訖站相同")
        ride = rng.randint(40, 240)
        first = max(_minutes(a.get("depart_after", "05:30")), 5 * 60 + 30) + rng.randint(0, 40)
        kind = a.get("train_type")
        trains = []
        for i in range(3):
            dep = first + i * 45
            trains.append(
                {
                    "train_no": str(rng.randint(100, 4999)),
                    "train_type": kind or rng.choice(["自強", "莒光", "區間"]),
                    "depart": _hhmm(dep),
                    "arrive": _hhmm(dep + ride),
                }
            )
        return {
            "origin": a["origin"],
            "destination": a["destination"],
            "date": a["date"],
            "trains": trains,
        }

    def _find_youbike(self, a: dict, rng: random.Random) -> dict:
        stations = [
            {
                "name": f"{a['near']}{suffix}",
                "available_bikes": rng.randint(0, 25),
                "empty_docks": rng.randint(0, 25),
                "distance_m": rng.randint(50, 600),
            }
            for suffix in ("站", "(東側)", "(捷運出口)")
        ]
        stations.sort(key=lambda s: s["distance_m"])
        return {"city": a["city"], "near": a["near"], "stations": stations}

    def _check_invoice(self, a: dict, rng: random.Random) -> dict:
        # The winning tail depends on the period only, like the real draw.
        winning_tail = f"{_rng('invoice-draw', {'period': a['period']}).randint(0, 999):03d}"
        won = a["number"].endswith(winning_tail)
        return {
            "period": a["period"],
            "number": a["number"],
            "result": "六獎（200 元）" if won else "未中獎",
        }

    def _calendar_add(self, a: dict, rng: random.Random) -> dict:
        start = datetime.fromisoformat(a["start"])
        end = datetime.fromisoformat(a["end"])
        if end <= start:
            raise ToolError("結束時間必須晚於開始時間")
        cur = self._db.execute(
            "INSERT INTO events (title, start, end, location) VALUES (?, ?, ?, ?)",
            (a["title"], a["start"], a["end"], a.get("location")),
        )
        return {"status": "created", "event_id": cur.lastrowid}

    def _calendar_list(self, a: dict, rng: random.Random) -> dict:
        rows = self._db.execute(
            "SELECT title, start, end, location FROM events WHERE substr(start, 1, 10) = ? ORDER BY start",
            (a["date"],),
        )
        events = [{"title": r[0], "start": r[1], "end": r[2], "location": r[3]} for r in rows]
        return {"date": a["date"], "events": events}
```

- [ ] **Step 4: 執行測試，確認通過；lint 與格式檢查**

Run: `.venv/Scripts/python -m pytest tests/test_mock_tools.py -v`
Expected: 全部 PASS

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: `All checks passed!` 且沒有需要重新格式化的檔案

- [ ] **Step 5: Commit**

```bash
git add src/twagent/tools tests/test_mock_tools.py
git commit -m "feat: deterministic mock tools with an inspectable SQLite calendar

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: 系統提示 `prompting.py`

**Files:**
- Create: `src/twagent/prompting.py`
- Test: `tests/test_prompting.py`

**Interfaces:**
- Consumes: `dates.format_today`、`tools.schemas.TOOLS`
- Produces: `PERSONA: str`、`render_tools_block(tools) -> str`、`build_persona(today: date) -> str`、`build_system_prompt(today: date, tools=None) -> str`（`tools=None` 代表使用 `TOOLS`）。

`render_tools_block` 的文字是依 Qwen chat template 的 `if tools` 分支逐字寫出；Task 12 會用真實伺服器驗證它，若有差異在那時修正。

- [ ] **Step 1: 寫失敗的測試**

`tests/test_prompting.py`：

```python
import json
from datetime import date

from twagent.prompting import build_system_prompt, render_tools_block
from twagent.tools.schemas import TOOLS


def test_today_is_injected():
    prompt = build_system_prompt(date(2026, 9, 23))
    assert "今天是 2026-09-23（星期三）" in prompt


def test_every_tool_appears_once_as_unescaped_json():
    block = render_tools_block(TOOLS)
    for t in TOOLS:
        line = json.dumps(t, ensure_ascii=False)
        assert block.count(line) == 1
    assert "\\u" not in block  # ensure_ascii=False, like the template's tojson


def test_block_shape_matches_qwen_template():
    block = render_tools_block(TOOLS[:1])
    assert block.startswith("# Tools\n\nYou may call one or more functions")
    assert "<tools>\n{" in block
    assert block.endswith(
        '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )


def test_persona_and_tools_are_separated_by_a_blank_line():
    prompt = build_system_prompt(date(2026, 9, 23))
    persona, _, rest = prompt.partition("\n\n# Tools")
    assert persona.startswith("你是臺灣生活助理")
    assert rest
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `.venv/Scripts/python -m pytest tests/test_prompting.py -v`
Expected: FAIL，錯誤為 `ModuleNotFoundError: No module named 'twagent.prompting'`

- [ ] **Step 3: 寫實作**

`src/twagent/prompting.py`：

```python
"""The one system prompt, rendered by us rather than by the chat template.

We render Qwen's tool block ourselves and send it as plain system text, with
no `tools` field in the request. Two reasons:

1. The server returns the raw completion, so `parsing.py` -- not the server --
   decides what counts as a tool call. That is what lets us measure parse
   failures at all.
2. Training data (next plan) is formatted with this same function, so the
   prompt the model is trained on and the prompt it is served with cannot
   drift apart.

`scripts/check_template.py` verifies that this rendering is byte-identical to
what the model's own chat template produces when given `tools`.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from twagent.dates import format_today
from twagent.tools.schemas import TOOLS

PERSONA = (
    "你是臺灣生活助理，可以查天氣、高鐵與臺鐵班次、YouBike 站點、統一發票對獎，並管理使用者的行事曆。\n"
    "- 一律使用繁體中文與臺灣用語回覆。\n"
    "- 今天是 {today}，時區為 Asia/Taipei。所有相對日期都以今天為準換算。\n"
    "- 呼叫工具所需的資訊不足時，先向使用者追問，不要自行猜測。\n"
    "- 超出工具能力範圍的請求（例如訂票、付款），直接說明做不到。"
)


def render_tools_block(tools: list[dict[str, Any]]) -> str:
    """Qwen's tool instructions, character for character.

    Mirrors the `if tools` branch of the Qwen chat template, where each tool
    goes through `tojson` (json.dumps with ensure_ascii=False).
    """
    lines = "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools)
    return (
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        f"{lines}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )


def build_persona(today: date) -> str:
    return PERSONA.format(today=format_today(today))


def build_system_prompt(today: date, tools: list[dict[str, Any]] | None = None) -> str:
    return build_persona(today) + "\n\n" + render_tools_block(TOOLS if tools is None else tools)
```

- [ ] **Step 4: 執行測試，確認通過；lint 與格式檢查**

Run: `.venv/Scripts/python -m pytest tests/test_prompting.py -v`
Expected: 全部 PASS

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: `All checks passed!` 且沒有需要重新格式化的檔案

- [ ] **Step 5: Commit**

```bash
git add src/twagent/prompting.py tests/test_prompting.py
git commit -m "feat: render the system prompt and Qwen tool block ourselves

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 工具呼叫解析器 `parsing.py`

**Files:**
- Create: `src/twagent/parsing.py`
- Test: `tests/test_parsing.py`

**Interfaces:**
- Produces: `ToolCall(name: str, arguments: dict)`（frozen，`to_dict()`）；`ParsedOutput(text: str, tool_calls: tuple[ToolCall, ...], errors: tuple[str, ...])`，屬性 `ok`；`parse_output(raw: str) -> ParsedOutput`；`render_tool_calls(calls) -> str`（標準文字格式，多個呼叫以 `\n` 相接）。

- [ ] **Step 1: 寫失敗的測試**

`tests/test_parsing.py`：

```python
from twagent.parsing import ToolCall, parse_output, render_tool_calls

WEATHER = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "臺中市", "date": "2026-09-24"}}\n</tool_call>'


def test_plain_answer():
    out = parse_output("你好！有什麼可以幫你？")
    assert out.ok and out.tool_calls == () and out.text == "你好！有什麼可以幫你？"


def test_single_call():
    out = parse_output(WEATHER)
    assert out.ok
    assert out.tool_calls == (ToolCall("get_weather", {"city": "臺中市", "date": "2026-09-24"}),)
    assert out.text == ""


def test_parallel_calls_keep_order():
    raw = WEATHER + "\n" + WEATHER.replace("臺中市", "高雄市")
    out = parse_output(raw)
    assert [c.arguments["city"] for c in out.tool_calls] == ["臺中市", "高雄市"]


def test_text_around_calls_is_kept():
    out = parse_output("我幫你查一下。\n" + WEATHER)
    assert out.ok and out.text == "我幫你查一下。" and len(out.tool_calls) == 1


def test_think_block_is_ignored():
    out = parse_output("<think>\n使用者要天氣\n</think>\n\n" + WEATHER)
    assert out.ok and len(out.tool_calls) == 1 and out.text == ""


def test_prefilled_think_closed_by_model():
    out = parse_output("嗯，要查天氣。\n</think>\n\n" + WEATHER)
    assert out.ok and len(out.tool_calls) == 1 and out.text == ""


def test_arguments_as_json_string_are_accepted():
    raw = '<tool_call>{"name": "calendar_list", "arguments": "{\\"date\\": \\"2026-09-24\\"}"}</tool_call>'
    out = parse_output(raw)
    assert out.ok and out.tool_calls[0].arguments == {"date": "2026-09-24"}


def test_missing_arguments_defaults_to_empty():
    out = parse_output('<tool_call>{"name": "calendar_list"}</tool_call>')
    assert out.ok and out.tool_calls[0].arguments == {}


def test_invalid_json_is_an_error():
    out = parse_output('<tool_call>{"name": "get_weather", "arguments": {"city": }</tool_call>')
    assert not out.ok and "invalid JSON" in out.errors[0]


def test_truncated_output_is_an_error():
    out = parse_output('<tool_call>\n{"name": "get_weather", "arguments": {"city": "臺')
    assert not out.ok and "unclosed" in out.errors[0]


def test_missing_name_is_an_error():
    out = parse_output('<tool_call>{"arguments": {}}</tool_call>')
    assert not out.ok and "missing 'name'" in out.errors[0]


def test_one_bad_block_fails_the_whole_output():
    out = parse_output(WEATHER + "\n<tool_call>not json</tool_call>")
    assert not out.ok
    assert len(out.tool_calls) == 1  # good one still reported, for diagnostics


def test_stray_closing_tag_is_an_error():
    out = parse_output("好的</tool_call>")
    assert not out.ok


def test_render_round_trips():
    calls = (ToolCall("get_weather", {"city": "臺中市", "date": "2026-09-24"}),)
    rendered = render_tool_calls(calls)
    assert rendered == WEATHER
    assert parse_output(rendered).tool_calls == calls
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `.venv/Scripts/python -m pytest tests/test_parsing.py -v`
Expected: FAIL，錯誤為 `ModuleNotFoundError: No module named 'twagent.parsing'`

- [ ] **Step 3: 寫實作**

`src/twagent/parsing.py`：

```python
"""The only code that turns model text into tool calls.

Evaluation and the server both call `parse_output`. A second parser would be
the same mistake as astro-vlm-lora's BUG-1: two copies of one operation, and
the published number comes from the wrong one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ParsedOutput:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    errors: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse_block(index: int, body: str) -> tuple[ToolCall | None, str | None]:
    try:
        obj = json.loads(body.strip())
    except json.JSONDecodeError as e:
        return None, f"tool_call #{index}: invalid JSON ({e.msg})"
    if not isinstance(obj, dict):
        return None, f"tool_call #{index}: expected an object"
    name, args = obj.get("name"), obj.get("arguments", {})
    if not isinstance(name, str) or not name:
        return None, f"tool_call #{index}: missing 'name'"
    if isinstance(args, str):
        # Some models emit arguments as a JSON-encoded string.
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None, f"tool_call #{index}: 'arguments' is not valid JSON"
    if not isinstance(args, dict):
        return None, f"tool_call #{index}: 'arguments' must be an object"
    return ToolCall(name=name, arguments=args), None


def parse_output(raw: str) -> ParsedOutput:
    """Split a completion into free text and tool calls.

    Any malformed block makes the whole output not-ok: executing half of what
    the model meant is worse than asking it again.
    """
    s = _THINK.sub("", raw)
    if "</think>" in s:  # template pre-filled <think>, model closed it
        s = s.split("</think>", 1)[1]
    calls: list[ToolCall] = []
    errors: list[str] = []
    for i, m in enumerate(_BLOCK.finditer(s), start=1):
        call, err = _parse_block(i, m.group(1))
        if call:
            calls.append(call)
        else:
            errors.append(err)
    rest = _BLOCK.sub("", s)
    if "<tool_call>" in rest:
        errors.append("unclosed <tool_call> (output truncated?)")
    elif "</tool_call>" in rest:
        errors.append("</tool_call> without matching <tool_call>")
    text = rest.replace("<tool_call>", "").strip()
    return ParsedOutput(text=text, tool_calls=tuple(calls), errors=tuple(errors))


def render_tool_calls(calls: list[ToolCall] | tuple[ToolCall, ...]) -> str:
    """Canonical text form, identical to how the Qwen template renders
    an assistant message's `tool_calls`."""
    return "\n".join(
        "<tool_call>\n" + json.dumps(c.to_dict(), ensure_ascii=False) + "\n</tool_call>"
        for c in calls
    )
```

- [ ] **Step 4: 執行測試，確認通過；lint 與格式檢查**

Run: `.venv/Scripts/python -m pytest tests/test_parsing.py -v`
Expected: 全部 PASS

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: `All checks passed!` 且沒有需要重新格式化的檔案

- [ ] **Step 5: Commit**

```bash
git add src/twagent/parsing.py tests/test_parsing.py
git commit -m "feat: single tool-call parser with think-block stripping

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: LLM 客戶端 `llm.py`

**Files:**
- Create: `src/twagent/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `Message = dict[str, str]`；`LLMError(Exception)`；`LLMClient` 協定（`complete(messages: list[Message]) -> str`）；`OpenAICompatClient(base_url, model, *, temperature=0.0, max_tokens=512, timeout_s=60.0, extra_body=None, transport=None)`；`ScriptedLLM(responses: list[str])`，屬性 `calls: list[list[Message]]`。

- [ ] **Step 1: 寫失敗的測試**

`tests/test_llm.py`：

```python
import json

import httpx
import pytest

from twagent.llm import LLMError, OpenAICompatClient, ScriptedLLM


def _client(handler, **kw):
    return OpenAICompatClient(
        "http://llm.test/v1/", "qwen", transport=httpx.MockTransport(handler), **kw
    )


def test_request_body_is_greedy_and_has_no_tools():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "好"}}]})

    out = _client(handler).complete([{"role": "user", "content": "嗨"}])
    assert out == "好"
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    body = seen["body"]
    assert body["temperature"] == 0.0
    assert body["stream"] is False
    assert "tools" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"] == [{"role": "user", "content": "嗨"}]


def test_extra_body_is_merged():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    _client(handler, extra_body={"lora": [{"id": 0, "scale": 0.0}]}).complete([])
    assert seen["lora"] == [{"id": 0, "scale": 0.0}]


def test_null_content_becomes_empty_string():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    assert _client(handler).complete([]) == ""


def test_http_error_raises_llm_error():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMError, match="failed"):
        _client(handler).complete([])


def test_bad_shape_raises_llm_error():
    def handler(request):
        return httpx.Response(200, json={"oops": 1})

    with pytest.raises(LLMError, match="unexpected"):
        _client(handler).complete([])


def test_scripted_llm_replays_and_records():
    llm = ScriptedLLM(["a", "b"])
    assert llm.complete([{"role": "user", "content": "1"}]) == "a"
    assert llm.complete([]) == "b"
    assert llm.calls[0] == [{"role": "user", "content": "1"}]
    with pytest.raises(AssertionError, match="ran out"):
        llm.complete([])
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `.venv/Scripts/python -m pytest tests/test_llm.py -v`
Expected: FAIL，錯誤為 `ModuleNotFoundError: No module named 'twagent.llm'`

- [ ] **Step 3: 寫實作**

`src/twagent/llm.py`：

```python
"""LLM clients. Every client takes chat messages and returns raw text.

Returning raw text (not structured tool calls) is deliberate: the agent runs
every model's output through the same `parse_output`, so a parse failure is
something we can count, whichever model produced it.

Messages are plain dicts with `role` in {system, user, assistant, tool} and a
string `content`. Assistant tool calls live inside `content` in their
canonical `<tool_call>` text form (see `parsing.render_tool_calls`).
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

Message = dict[str, str]


class LLMError(Exception):
    """The model could not be reached or returned something unusable."""


class LLMClient(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


class OpenAICompatClient:
    """Client for any OpenAI-compatible `/chat/completions` endpoint.

    Used for llama-server. Never sends `tools`; the tool block is already in
    the system prompt (see `prompting.py`).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 60.0,
        extra_body: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Qwen3 hybrid models think by default; tool calling does not want it.
        # Harmless for templates that ignore the flag.
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.extra_body.update(extra_body or {})
        self._http = httpx.Client(timeout=timeout_s, transport=transport)

    def complete(self, messages: list[Message]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            **self.extra_body,
        }
        try:
            resp = self._http.post(self.url, json=body)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"request to {self.url} failed: {e}") from e
        try:
            content = data["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {data!r:.200}") from e
        return content or ""


class ScriptedLLM:
    """Test double: returns canned responses in order and records its inputs."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self._responses.pop(0)
```

- [ ] **Step 4: 執行測試，確認通過；lint 與格式檢查**

Run: `.venv/Scripts/python -m pytest tests/test_llm.py -v`
Expected: 全部 PASS

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: `All checks passed!` 且沒有需要重新格式化的檔案

- [ ] **Step 5: Commit**

```bash
git add src/twagent/llm.py tests/test_llm.py
git commit -m "feat: OpenAI-compatible LLM client that returns raw text

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Agent 迴圈 `agent.py`

**Files:**
- Create: `src/twagent/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `config.AgentConfig`、`llm.LLMClient / LLMError / Message / ScriptedLLM`、`parsing.parse_output / render_tool_calls / ToolCall`、`tools.base.ToolError / ToolExecutor`、`tools.schemas.validate_arguments`、`tools.mock.MockTools`、`prompting.build_system_prompt`
- Produces: `run_agent(conversation, *, llm, tools, system_prompt, config, fallback=None) -> AgentResult`；`AgentResult(final_text, stopped_reason, messages, trace, tool_calls, used_fallback)` 與 `to_dict()`；`TraceEvent(type, data)`；`stopped_reason` ∈ `answer | max_steps | parse_failure | schema_failure | llm_error`；常數 `FAILURE_MESSAGE`、`MAX_STEPS_MESSAGE`、`PARSE_RETRY_PROMPT`。

- [ ] **Step 1: 寫失敗的測試**

`tests/test_agent.py`：

```python
import json
from datetime import date

import pytest

from twagent.agent import FAILURE_MESSAGE, MAX_STEPS_MESSAGE, run_agent
from twagent.config import AgentConfig
from twagent.llm import LLMError, ScriptedLLM
from twagent.prompting import build_system_prompt
from twagent.tools.mock import MockTools

SYSTEM = build_system_prompt(date(2026, 9, 23))
USER = [{"role": "user", "content": "明天台中會下雨嗎"}]
CFG = AgentConfig(max_steps=5, parse_retries=1, schema_retries=2)


def call(name, **args):
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        + "\n</tool_call>"
    )


WEATHER = call("get_weather", city="臺中市", date="2026-09-24")


class FailingLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        raise LLMError("connection refused")


def run(llm, tools=None, fallback=None, cfg=CFG):
    return run_agent(
        USER,
        llm=llm,
        tools=tools or MockTools(),
        system_prompt=SYSTEM,
        config=cfg,
        fallback=fallback,
    )


def test_direct_answer():
    r = run(ScriptedLLM(["你好！"]))
    assert r.stopped_reason == "answer" and r.final_text == "你好！"
    assert r.tool_calls == []
    assert r.messages[0] == {"role": "system", "content": SYSTEM}


def test_single_tool_round_trip():
    llm = ScriptedLLM([WEATHER, "明天臺中多雲，降雨機率 20%。"])
    r = run(llm)
    assert r.stopped_reason == "answer"
    assert [c.name for c in r.tool_calls] == ["get_weather"]
    # Second LLM call sees: system, user, assistant(tool_call), tool(result)
    second = llm.calls[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["content"] == WEATHER
    assert json.loads(second[3]["content"])["city"] == "臺中市"


def test_parallel_calls_each_get_a_tool_message():
    both = WEATHER + "\n" + call("get_weather", city="高雄市", date="2026-09-24")
    llm = ScriptedLLM([both, "兩地都晴。"])
    r = run(llm)
    assert [c.arguments["city"] for c in r.tool_calls] == ["臺中市", "高雄市"]
    assert [m["role"] for m in llm.calls[1]][-2:] == ["tool", "tool"]


def test_sequential_calls_update_calendar_state():
    tools = MockTools()
    llm = ScriptedLLM(
        [
            call(
                "search_thsr",
                origin="臺北",
                destination="臺南",
                date="2026-09-24",
                depart_after="07:00",
            ),
            call(
                "calendar_add", title="高鐵去臺南", start="2026-09-24T07:30", end="2026-09-24T09:30"
            ),
            "已幫你排進行事曆。",
        ]
    )
    r = run(llm, tools=tools)
    assert r.stopped_reason == "answer"
    assert [e["title"] for e in tools.events()] == ["高鐵去臺南"]


def test_parse_error_then_successful_retry():
    llm = ScriptedLLM(
        ['<tool_call>{"name": "get_weather", "arguments": {</tool_call>', WEATHER, "多雲"]
    )
    r = run(llm)
    assert r.stopped_reason == "answer"
    assert [e.type for e in r.trace].count("parse_error") == 1
    retry_prompt = llm.calls[1][-1]
    assert retry_prompt["role"] == "user" and "格式錯誤" in retry_prompt["content"]


def test_parse_failure_without_fallback_gives_up():
    bad = "<tool_call>oops</tool_call>"
    r = run(ScriptedLLM([bad, bad]))
    assert r.stopped_reason == "parse_failure" and r.final_text == FAILURE_MESSAGE


def test_parse_failure_escalates_to_fallback_without_garbage():
    bad = "<tool_call>oops</tool_call>"
    fallback = ScriptedLLM([WEATHER, "多雲"])
    r = run(ScriptedLLM([bad, bad]), fallback=fallback)
    assert r.stopped_reason == "answer" and r.used_fallback
    # Fallback's first view is the clean conversation: no bad output, no retry prompt.
    assert [m["role"] for m in fallback.calls[0]] == ["system", "user"]


def test_schema_error_is_fed_back_and_corrected():
    wrong = call("get_weather", city="台中", date="明天")
    llm = ScriptedLLM([wrong, WEATHER, "多雲"])
    r = run(llm)
    assert r.stopped_reason == "answer"
    assert [c.arguments["city"] for c in r.tool_calls] == ["臺中市"]  # invalid call not executed
    feedback = json.loads(llm.calls[1][-1]["content"])
    assert feedback["error"] == "參數驗證失敗" and feedback["details"]


def test_schema_retries_exhausted():
    wrong = call("get_weather", city="台中", date="明天")
    r = run(ScriptedLLM([wrong] * 3))
    assert r.stopped_reason == "schema_failure" and r.final_text == FAILURE_MESSAGE


def test_tool_error_is_reported_not_replaced():
    tools = MockTools(failures={"get_weather": "連線逾時"})
    llm = ScriptedLLM([WEATHER, "抱歉，氣象服務暫時連不上。"])
    r = run(llm, tools=tools)
    assert r.stopped_reason == "answer"
    assert json.loads(llm.calls[1][-1]["content"]) == {"error": "連線逾時"}
    assert [e.type for e in r.trace] == ["llm_output", "tool_error", "llm_output"]


def test_llm_error_without_fallback():
    r = run(FailingLLM())
    assert r.stopped_reason == "llm_error" and r.final_text == FAILURE_MESSAGE


def test_llm_error_with_fallback():
    r = run(FailingLLM(), fallback=ScriptedLLM(["你好"]))
    assert r.stopped_reason == "answer" and r.used_fallback


def test_max_steps():
    r = run(ScriptedLLM([WEATHER] * 5))
    assert r.stopped_reason == "max_steps" and r.final_text == MAX_STEPS_MESSAGE
    assert len(r.tool_calls) == 5


@pytest.mark.parametrize("steps", [1, 2])
def test_max_steps_counts_every_llm_call(steps):
    llm = ScriptedLLM([WEATHER] * steps)
    run(llm, cfg=AgentConfig(max_steps=steps))
    assert len(llm.calls) == steps


def test_to_dict_is_json_serialisable():
    r = run(ScriptedLLM([WEATHER, "多雲"]))
    d = r.to_dict()
    json.dumps(d, ensure_ascii=False)
    assert d["tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "臺中市", "date": "2026-09-24"}}
    ]
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `.venv/Scripts/python -m pytest tests/test_agent.py -v`
Expected: FAIL，錯誤為 `ModuleNotFoundError: No module named 'twagent.agent'`

- [ ] **Step 3: 寫實作**

`src/twagent/agent.py`：

```python
"""The only agent loop: model -> tools -> model, until the model answers.

Evaluation and the server both call `run_agent`. Failure policy:

- malformed <tool_call>   -> tell the model, let it retry (`parse_retries`)
- schema-invalid call     -> return the validation errors as the tool result
                             so the model can fix them (`schema_retries`)
- tool raised ToolError   -> return the error as the tool result; never
                             substitute another executor's data
- LLM unreachable, or a retry budget is exhausted
                          -> switch to `fallback` if one was given, else stop
                             with a fixed apology
- more than `max_steps` LLM calls -> stop
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from twagent.config import AgentConfig
from twagent.llm import LLMClient, LLMError, Message
from twagent.parsing import ToolCall, parse_output, render_tool_calls
from twagent.tools.base import ToolError, ToolExecutor
from twagent.tools.schemas import validate_arguments

FAILURE_MESSAGE = "抱歉，我這次沒辦法正確處理你的請求，請換個說法再試一次。"
MAX_STEPS_MESSAGE = "抱歉，這個請求需要的步驟太多，我先停在這裡。請把需求拆小一點再試試。"
PARSE_RETRY_PROMPT = "（系統訊息）你上一則輸出的工具呼叫格式錯誤：{errors}。請重新輸出，工具呼叫必須是 <tool_call> 內的合法 JSON。"


@dataclass(frozen=True)
class TraceEvent:
    type: str  # llm_output | parse_error | schema_error | tool_result | tool_error | llm_error | fallback
    data: dict[str, Any]


@dataclass
class AgentResult:
    final_text: str
    stopped_reason: str  # answer | max_steps | parse_failure | schema_failure | llm_error
    messages: list[Message]
    trace: list[TraceEvent] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)  # valid calls, in execution order
    used_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_text": self.final_text,
            "stopped_reason": self.stopped_reason,
            "used_fallback": self.used_fallback,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "trace": [{"type": e.type, **e.data} for e in self.trace],
        }


def _assistant_content(text: str, calls: tuple[ToolCall, ...]) -> str:
    rendered = render_tool_calls(calls)
    return f"{text}\n{rendered}" if text else rendered


def run_agent(
    conversation: list[Message],
    *,
    llm: LLMClient,
    tools: ToolExecutor,
    system_prompt: str,
    config: AgentConfig,
    fallback: LLMClient | None = None,
) -> AgentResult:
    messages: list[Message] = [{"role": "system", "content": system_prompt}, *conversation]
    result = AgentResult(final_text="", stopped_reason="", messages=messages)
    active = llm
    parse_failures = 0
    schema_failures = 0
    retry_start: int | None = None  # where the current parse-retry exchange began

    def escalate(reason: str) -> bool:
        """Switch to the fallback model. False means there is none left."""
        nonlocal active, parse_failures, schema_failures, retry_start
        if fallback is None or active is fallback:
            result.final_text, result.stopped_reason = FAILURE_MESSAGE, reason
            return False
        if retry_start is not None:
            del messages[retry_start:]  # the fallback should not see the garbage
        active, parse_failures, schema_failures, retry_start = fallback, 0, 0, None
        result.used_fallback = True
        result.trace.append(TraceEvent("fallback", {"reason": reason}))
        return True

    for _ in range(config.max_steps):
        try:
            raw = active.complete(messages)
        except LLMError as e:
            result.trace.append(TraceEvent("llm_error", {"error": str(e)}))
            if escalate("llm_error"):
                continue
            return result
        result.trace.append(TraceEvent("llm_output", {"raw": raw}))
        parsed = parse_output(raw)

        if not parsed.ok:
            result.trace.append(TraceEvent("parse_error", {"errors": list(parsed.errors)}))
            if parse_failures < config.parse_retries:
                parse_failures += 1
                if retry_start is None:
                    retry_start = len(messages)
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": PARSE_RETRY_PROMPT.format(errors="；".join(parsed.errors)),
                    }
                )
                continue
            if escalate("parse_failure"):
                continue
            return result
        retry_start = None

        if not parsed.tool_calls:
            messages.append({"role": "assistant", "content": parsed.text})
            result.final_text, result.stopped_reason = parsed.text, "answer"
            return result

        messages.append(
            {"role": "assistant", "content": _assistant_content(parsed.text, parsed.tool_calls)}
        )
        any_invalid = False
        for call in parsed.tool_calls:
            errors = validate_arguments(call.name, call.arguments)
            if errors:
                any_invalid = True
                result.trace.append(
                    TraceEvent("schema_error", {"call": call.to_dict(), "errors": errors})
                )
                payload: dict[str, Any] = {"error": "參數驗證失敗", "details": errors}
            else:
                try:
                    payload = tools.call(call.name, call.arguments)
                    result.tool_calls.append(call)
                    result.trace.append(
                        TraceEvent("tool_result", {"call": call.to_dict(), "result": payload})
                    )
                except ToolError as e:
                    result.tool_calls.append(call)
                    result.trace.append(
                        TraceEvent("tool_error", {"call": call.to_dict(), "error": str(e)})
                    )
                    payload = {"error": str(e)}
            messages.append({"role": "tool", "content": json.dumps(payload, ensure_ascii=False)})

        if any_invalid:
            schema_failures += 1
            if schema_failures > config.schema_retries and not escalate("schema_failure"):
                return result

    result.final_text, result.stopped_reason = MAX_STEPS_MESSAGE, "max_steps"
    return result
```

- [ ] **Step 4: 執行測試，確認通過；lint 與格式檢查**

Run: `.venv/Scripts/python -m pytest tests/test_agent.py -v`
Expected: 全部 PASS

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: `All checks passed!` 且沒有需要重新格式化的檔案

- [ ] **Step 5: Commit**

```bash
git add src/twagent/agent.py tests/test_agent.py
git commit -m "feat: agent loop with retry, schema feedback, fallback and step cap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: 測試題格式 `cases.py` 與種子題

**Files:**
- Create: `src/twagent/cases.py`、`data/handwritten_test.jsonl`（6 筆種子題）、`data/README.md`（撰寫規則）
- Test: `tests/test_cases.py`

**Interfaces:**
- Consumes: `parsing.ToolCall`、`tools.schemas.validate_arguments`、`llm.Message`
- Produces: `CATEGORIES`、`EXPECTED_DECISION`、`MIN_CALLS`；`Case(id, category, today, messages, decision, calls, fail_tools, note)`；`case_from_dict(d) -> Case`；`validate_case(c) -> list[str]`；`load_cases(path) -> tuple[list[Case], list[str]]`。Plan 2 的合成測試集與 Plan 3 的評估都使用這個格式。

- [ ] **Step 1: 寫失敗的測試**

`tests/test_cases.py`：

```python
import json
from pathlib import Path

import pytest

from twagent.cases import CATEGORIES, case_from_dict, load_cases, validate_case

ROOT = Path(__file__).resolve().parents[1]

GOOD = {
    "id": "t-1",
    "category": "single",
    "today": "2026-09-23",
    "messages": [{"role": "user", "content": "明天台中會下雨嗎"}],
    "expected": {
        "decision": "call",
        "calls": [{"name": "get_weather", "arguments": {"city": "臺中市", "date": "2026-09-24"}}],
    },
}


def with_(**overrides):
    d = json.loads(json.dumps(GOOD))
    d.update(overrides)
    return d


def test_good_case_is_valid():
    assert validate_case(case_from_dict(GOOD)) == []


def test_handwritten_seed_file_is_valid_and_covers_every_category():
    cases, errors = load_cases(ROOT / "data" / "handwritten_test.jsonl")
    assert errors == []
    assert {c.category for c in cases} == set(CATEGORIES)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"category": "weird"}, "unknown category"),
        ({"expected": {"decision": "answer", "calls": []}}, "requires decision"),
        ({"category": "parallel"}, "at least 2"),
        (
            {
                "category": "clarify",
                "expected": {"decision": "clarify", "calls": GOOD["expected"]["calls"]},
            },
            "must not expect",
        ),
        ({"messages": [{"role": "assistant", "content": "hi"}]}, "end with a user turn"),
        (
            {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]},
            "user/assistant",
        ),
        (
            {
                "expected": {
                    "decision": "call",
                    "calls": [{"name": "get_weather", "arguments": {"city": "台中"}}],
                }
            },
            "get_weather",
        ),
        ({"fail_tools": {"get_weather": "逾時"}}, "only allowed in tool_error"),
        ({"category": "tool_error"}, "must set fail_tools"),
        ({"category": "tool_error", "fail_tools": {"search_thsr": "逾時"}}, "never called"),
    ],
)
def test_invalid_cases(override, fragment):
    errors = validate_case(case_from_dict(with_(**override)))
    assert any(fragment in e for e in errors), errors


def test_load_reports_line_numbers_and_duplicates(tmp_path):
    p = tmp_path / "cases.jsonl"
    lines = [
        json.dumps(GOOD, ensure_ascii=False),
        "{not json",
        json.dumps(GOOD, ensure_ascii=False),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cases, errors = load_cases(p)
    assert len(cases) == 1
    assert any(e.startswith("line 2:") for e in errors)
    assert any("line 3" in e and "duplicate id" in e for e in errors)
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `.venv/Scripts/python -m pytest tests/test_cases.py -v`
Expected: FAIL，錯誤為 `ModuleNotFoundError: No module named 'twagent.cases'`

- [ ] **Step 3: 寫實作**

`src/twagent/cases.py`：

```python
"""Test cases: the format shared by the handwritten and synthetic test sets.

One JSON object per line:

    {"id": "hw-001", "category": "single", "today": "2026-09-23",
     "messages": [{"role": "user", "content": "明天台中會下雨嗎"}],
     "expected": {"decision": "call",
                  "calls": [{"name": "get_weather",
                             "arguments": {"city": "臺中市", "date": "2026-09-24"}}]},
     "fail_tools": {}, "note": ""}

`expected.calls` lists every valid call the agent should make, in order,
across all steps. `fail_tools` makes MockTools raise for those tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from twagent.llm import Message
from twagent.parsing import ToolCall
from twagent.tools.schemas import validate_arguments

CATEGORIES = ("single", "parallel", "sequential", "clarify", "no_tool", "tool_error")
EXPECTED_DECISION = {
    "single": "call",
    "parallel": "call",
    "sequential": "call",
    "clarify": "clarify",
    "no_tool": "answer",
    "tool_error": "call",
}
MIN_CALLS = {"single": 1, "parallel": 2, "sequential": 2, "tool_error": 1}


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    today: date
    messages: list[Message]
    decision: str
    calls: list[ToolCall]
    fail_tools: dict[str, str] = field(default_factory=dict)
    note: str = ""


def case_from_dict(d: dict[str, Any]) -> Case:
    """Build a Case; raises ValueError / KeyError / TypeError on bad structure."""
    expected = d["expected"]
    return Case(
        id=str(d["id"]),
        category=d["category"],
        today=date.fromisoformat(d["today"]),
        messages=[{"role": m["role"], "content": m["content"]} for m in d["messages"]],
        decision=expected["decision"],
        calls=[ToolCall(c["name"], c.get("arguments", {})) for c in expected.get("calls", [])],
        fail_tools=dict(d.get("fail_tools", {})),
        note=d.get("note", ""),
    )


def validate_case(c: Case) -> list[str]:
    errors: list[str] = []
    if c.category not in CATEGORIES:
        return [f"unknown category {c.category!r}"]
    if c.decision != EXPECTED_DECISION[c.category]:
        errors.append(f"category {c.category} requires decision {EXPECTED_DECISION[c.category]!r}")
    if not c.messages or c.messages[-1]["role"] != "user":
        errors.append("messages must be non-empty and end with a user turn")
    if any(m["role"] not in ("user", "assistant") for m in c.messages):
        errors.append("messages may only contain user/assistant turns")
    need = MIN_CALLS.get(c.category, 0)
    if need and len(c.calls) < need:
        errors.append(f"category {c.category} needs at least {need} call(s), got {len(c.calls)}")
    if not need and c.calls:
        errors.append(f"category {c.category} must not expect tool calls")
    for i, call in enumerate(c.calls, start=1):
        errors.extend(
            f"call #{i} {call.name}: {e}" for e in validate_arguments(call.name, call.arguments)
        )
    if c.category == "tool_error":
        if not c.fail_tools:
            errors.append("tool_error cases must set fail_tools")
        elif not set(c.fail_tools) <= {call.name for call in c.calls}:
            errors.append("fail_tools names a tool that is never called")
    elif c.fail_tools:
        errors.append("fail_tools is only allowed in tool_error cases")
    return errors


def load_cases(path: str | Path) -> tuple[list[Case], list[str]]:
    """Parse a JSONL file. Returns (valid cases, errors prefixed with line numbers)."""
    cases: list[Case] = []
    errors: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                case = case_from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                errors.append(f"line {lineno}: cannot parse ({type(e).__name__}: {e})")
                continue
            problems = validate_case(case)
            if case.id in seen:
                problems.append(f"duplicate id {case.id!r}")
            seen.add(case.id)
            if problems:
                errors.extend(f"line {lineno} ({case.id}): {p}" for p in problems)
            else:
                cases.append(case)
    return cases, errors
```

`data/handwritten_test.jsonl`：

```json
{"id": "hw-001", "category": "single", "today": "2026-09-23", "messages": [{"role": "user", "content": "明天台中會下雨嗎"}], "expected": {"decision": "call", "calls": [{"name": "get_weather", "arguments": {"city": "臺中市", "date": "2026-09-24"}}]}, "fail_tools": {}, "note": "相對日期＋台→臺"}
{"id": "hw-002", "category": "parallel", "today": "2026-09-23", "messages": [{"role": "user", "content": "週末要去玩，幫我看禮拜六台北跟花蓮的天氣"}], "expected": {"decision": "call", "calls": [{"name": "get_weather", "arguments": {"city": "臺北市", "date": "2026-09-26"}}, {"name": "get_weather", "arguments": {"city": "花蓮縣", "date": "2026-09-26"}}]}, "fail_tools": {}, "note": ""}
{"id": "hw-003", "category": "sequential", "today": "2026-09-23", "messages": [{"role": "user", "content": "看一下我後天有什麼行程，然後幫我加一個晚上七點到九點在信義區的聚餐"}], "expected": {"decision": "call", "calls": [{"name": "calendar_list", "arguments": {"date": "2026-09-25"}}, {"name": "calendar_add", "arguments": {"title": "聚餐", "start": "2026-09-25T19:00", "end": "2026-09-25T21:00", "location": "信義區"}}]}, "fail_tools": {}, "note": "後續參數完全由使用者訊息決定"}
{"id": "hw-004", "category": "clarify", "today": "2026-09-23", "messages": [{"role": "user", "content": "幫我查高鐵"}], "expected": {"decision": "clarify", "calls": []}, "fail_tools": {}, "note": "缺起訖站與日期"}
{"id": "hw-005", "category": "no_tool", "today": "2026-09-23", "messages": [{"role": "user", "content": "幫我訂一張明天去東京的機票"}], "expected": {"decision": "answer", "calls": []}, "fail_tools": {}, "note": "不支援的請求"}
{"id": "hw-006", "category": "tool_error", "today": "2026-09-23", "messages": [{"role": "user", "content": "我在台大附近，哪裡有 YouBike 可以借"}], "expected": {"decision": "call", "calls": [{"name": "find_youbike", "arguments": {"city": "臺北市", "near": "臺大"}}]}, "fail_tools": {"find_youbike": "連線逾時"}, "note": "工具失敗要如實告知"}
```

`data/README.md`：

````markdown
# 手寫測試集 `handwritten_test.jsonl`

這份測試集衡量的是**真實泛化能力**：模型面對「真人平常講話的方式」表現如何。
它和合成測試集的分數落差，本身就是報告裡的一項發現。

## 規則

1. **不要看合成資料再寫**，也**絕對不要把這份檔案拿給資料生成用的 LLM 看**。
   任何一種都會讓它變成「生成器風格」的考卷，失去意義。
2. 用你自己（或朋友）平常打字的方式寫：口語、錯字、省略、中英夾雜、講「台北」不講「臺北市」都可以。
3. `expected` 裡的參數一律用**正式列舉值**（`臺中市`、`臺北`、`左營`），
   日期一律換算成 `YYYY-MM-DD`，以該筆的 `today` 為準。
4. `sequential`（循序多步）題目裡，**後面步驟的參數必須完全由使用者訊息決定**，
   不能依賴工具回傳的內容（例如「搭查到的第一班車」），否則標準答案無法事先寫定。
5. `tool_error` 題目要在 `fail_tools` 指定哪個工具會失敗，以及錯誤訊息。
6. `id` 依序編號 `hw-001` 到 `hw-100`，不可重複。

## 份數分配（共 100 筆）

| category | 份數 | decision | 說明 |
|---|---|---|---|
| `single` | 35 | `call` | 一次呼叫一個工具 |
| `parallel` | 15 | `call` | 一次呼叫兩個以上互不相依的工具 |
| `sequential` | 15 | `call` | 兩步以上，依序執行 |
| `clarify` | 15 | `clarify` | 缺少必要參數，應先追問；`calls` 為空 |
| `no_tool` | 15 | `answer` | 閒聊、常識、或不支援的請求（訂票、付款）；`calls` 為空 |
| `tool_error` | 5 | `call` | 工具失敗，應如實告知使用者 |

七個工具都要被涵蓋，發票與 YouBike 各至少 5 筆。

## 驗證

```bash
.venv/Scripts/twagent check-cases data/handwritten_test.jsonl
```

輸出最後一行應為 `100 valid case(s): clarify=15, no_tool=15, parallel=15, sequential=15, single=35, tool_error=5`，且沒有任何 `ERROR`。
````

- [ ] **Step 4: 執行測試，確認通過；lint 與格式檢查**

Run: `.venv/Scripts/python -m pytest tests/test_cases.py -v`
Expected: 全部 PASS

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: `All checks passed!` 且沒有需要重新格式化的檔案

- [ ] **Step 5: Commit**

```bash
git add src/twagent/cases.py data tests/test_cases.py
git commit -m "feat: shared test-case format with validation and six seed cases

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: 命令列 `cli.py`

**Files:**
- Create: `src/twagent/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `agent.run_agent`、`cases.load_cases`、`config.load_config`、`llm.OpenAICompatClient`、`prompting.build_system_prompt`、`tools.mock.MockTools`
- Produces: `main(argv=None) -> int`；子指令 `check-cases PATH`（有錯誤時回傳 1）與 `chat MESSAGE [--config] [--today] [--base-url] [--json]`。`pyproject.toml` 的 `twagent` 指令指向它。

完成後執行一次全部測試：`.venv/Scripts/python -m pytest`，Expected：120 passed（Task 12 會再加 3 個，共 123）。

- [ ] **Step 1: 寫失敗的測試**

`tests/test_cli.py`：

```python
import json
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
```

- [ ] **Step 2: 執行測試，確認失敗**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: FAIL，錯誤為 `ModuleNotFoundError: No module named 'twagent.cli'`

- [ ] **Step 3: 寫實作**

`src/twagent/cli.py`：

```python
"""`twagent` command line.

twagent check-cases data/handwritten_test.jsonl
twagent chat "明天台中會下雨嗎" [--today 2026-09-23] [--base-url URL] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from twagent.agent import run_agent
from twagent.cases import load_cases
from twagent.config import load_config
from twagent.llm import OpenAICompatClient
from twagent.prompting import build_system_prompt
from twagent.tools.mock import MockTools

DEFAULT_CONFIG = Path("configs/default.yaml")


def _check_cases(args: argparse.Namespace) -> int:
    cases, errors = load_cases(args.path)
    for e in errors:
        print(f"ERROR {e}")
    counts = Counter(c.category for c in cases)
    print(
        f"{len(cases)} valid case(s): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    return 1 if errors else 0


def _chat(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    llm = OpenAICompatClient(
        args.base_url or cfg.llm.base_url,
        cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        timeout_s=cfg.llm.timeout_s,
    )
    today = date.fromisoformat(args.today) if args.today else date.today()
    result = run_agent(
        [{"role": "user", "content": args.message}],
        llm=llm,
        tools=MockTools(),
        system_prompt=build_system_prompt(today),
        config=cfg.agent,
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    for event in result.trace:
        print(f"[{event.type}] {json.dumps(event.data, ensure_ascii=False)}")
    print(f"\n({result.stopped_reason}) {result.final_text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="twagent")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-cases", help="validate a test-case JSONL file")
    p.add_argument("path", type=Path)
    p.set_defaults(func=_check_cases)

    p = sub.add_parser("chat", help="run one message through the agent (mock tools)")
    p.add_argument("message")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--today", help="YYYY-MM-DD; defaults to the real date")
    p.add_argument("--base-url", help="override llm.base_url")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.set_defaults(func=_chat)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 執行測試，確認通過；lint 與格式檢查**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: 全部 PASS

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .`
Expected: `All checks passed!` 且沒有需要重新格式化的檔案

- [ ] **Step 5: Commit**

```bash
git add src/twagent/cli.py tests/test_cli.py
git commit -m "feat: twagent CLI for case checking and one-shot chat

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: 模板對照與 B0 端到端冒煙

需要 Task 1 成功啟動的 llama-server（4B 模型，`--jinja`）正在 `localhost:8080` 執行。

**Files:**
- Create: `scripts/check_template.py`
- Create: `tests/fixtures/qwen_first_turn.txt`、`tests/fixtures/qwen_after_tool_call.txt`（由腳本產生）
- Test: `tests/test_template_fixture.py`
- Modify: `docs/spike-w1.md`（填寫最後兩節）

**Interfaces:**
- Consumes: `prompting.build_persona(today)`、`prompting.build_system_prompt(today)`、`parsing.ToolCall`、`parsing.render_tool_calls`、`tools.schemas.TOOLS`、`twagent chat`（Task 11）

- [ ] **Step 1: 寫對照腳本**

`scripts/check_template.py`：

```python
"""Verify that our hand-rendered prompt equals the model's own chat template.

Needs a running llama-server (started with --jinja). For each scenario it asks
the server to render two requests via POST /apply-template:

  A. the "official" way: `tools` field + structured assistant `tool_calls`
  B. our way: tools rendered into the system text, tool calls as plain text

and diffs the two prompts. Any difference means training and serving would
see different text, so fix `prompting.py` / `parsing.render_tool_calls`
before going further.

    python scripts/check_template.py --server http://localhost:8080
    python scripts/check_template.py --save tests/fixtures   # record for CI
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from datetime import date
from pathlib import Path

import httpx

from twagent.parsing import ToolCall, render_tool_calls
from twagent.prompting import build_persona, build_system_prompt
from twagent.tools.schemas import TOOLS

TODAY = date(2026, 9, 23)
CALL = ToolCall("get_weather", {"city": "臺中市", "date": "2026-09-24"})
RESULT = json.dumps({"city": "臺中市", "condition": "多雲"}, ensure_ascii=False)


def scenarios() -> dict[str, tuple[dict, dict]]:
    user = {"role": "user", "content": "明天台中會下雨嗎"}
    official_system = {"role": "system", "content": build_persona(TODAY)}
    ours_system = {"role": "system", "content": build_system_prompt(TODAY)}
    official_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": CALL.name,
                    "arguments": json.dumps(CALL.arguments, ensure_ascii=False),
                },
            }
        ],
    }
    ours_call = {"role": "assistant", "content": render_tool_calls([CALL])}
    tool_msg = {"role": "tool", "tool_call_id": "call_0", "content": RESULT}
    ours_tool = {"role": "tool", "content": RESULT}
    return {
        "first_turn": (
            {"messages": [official_system, user], "tools": TOOLS},
            {"messages": [ours_system, user]},
        ),
        "after_tool_call": (
            {"messages": [official_system, user, official_call, tool_msg], "tools": TOOLS},
            {"messages": [ours_system, user, ours_call, ours_tool]},
        ),
    }


def render(http: httpx.Client, server: str, body: dict) -> str:
    resp = http.post(f"{server.rstrip('/')}/apply-template", json=body)
    resp.raise_for_status()
    return resp.json()["prompt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument(
        "--save", type=Path, help="write each official rendering to DIR/qwen_<name>.txt"
    )
    args = ap.parse_args()
    failed = False
    with httpx.Client(timeout=30) as http:
        for name, (official, ours) in scenarios().items():
            a, b = render(http, args.server, official), render(http, args.server, ours)
            if args.save:
                args.save.mkdir(parents=True, exist_ok=True)
                (args.save / f"qwen_{name}.txt").write_text(a, encoding="utf-8", newline="")
            if a == b:
                print(f"[same] {name}")
                continue
            failed = True
            print(f"[DIFF] {name}")
            sys.stdout.writelines(
                difflib.unified_diff(
                    a.splitlines(keepends=True), b.splitlines(keepends=True), "official", "ours"
                )
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 執行對照**

Run: `.venv/Scripts/python scripts/check_template.py --server http://localhost:8080`
Expected:
```
[same] first_turn
[same] after_tool_call
```

若出現 `[DIFF]`：依 diff 修改 `src/twagent/prompting.py` 的 `render_tools_block`（first_turn 的差異）或 `src/twagent/parsing.py` 的 `render_tool_calls`（after_tool_call 的差異），直到兩個都是 `[same]`；同步更新 `tests/test_prompting.py`、`tests/test_parsing.py` 中受影響的斷言，並重跑 `.venv/Scripts/python -m pytest`。若 `/apply-template` 回傳 404 或忽略 `tools`，改用 `curl -s localhost:8080/props` 取得 `chat_template` 原文，手動比對，並把做法記在 `docs/spike-w1.md`。

- [ ] **Step 3: 記錄官方渲染結果**

Run: `.venv/Scripts/python scripts/check_template.py --save tests/fixtures`
Expected：產生 `tests/fixtures/qwen_first_turn.txt` 與 `tests/fixtures/qwen_after_tool_call.txt`，腳本 exit code 0。

- [ ] **Step 4: 寫鎖定測試**

`tests/test_template_fixture.py`：

```python
"""Lock in what scripts/check_template.py verified against the real server.

The fixtures are the model's own chat-template rendering, recorded once with
`python scripts/check_template.py --save tests/fixtures`. If someone edits the
persona, a schema or the tool-call rendering, these tests say whether the
official template would still produce the same text -- without a GPU.
Re-record the fixtures whenever `TOOLS` or `PERSONA` change on purpose.
"""

import json
from datetime import date
from pathlib import Path

from twagent.parsing import ToolCall, render_tool_calls
from twagent.prompting import build_system_prompt

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TODAY = date(2026, 9, 23)
CALL = ToolCall("get_weather", {"city": "臺中市", "date": "2026-09-24"})
RESULT = json.dumps({"city": "臺中市", "condition": "多雲"}, ensure_ascii=False)


def read(name: str) -> str:
    return (FIXTURES / f"qwen_{name}.txt").read_text(encoding="utf-8")


def test_system_prompt_matches_official_rendering():
    assert f"<|im_start|>system\n{build_system_prompt(TODAY)}<|im_end|>\n" in read("first_turn")


def test_tool_call_rendering_matches_official_rendering():
    assert render_tool_calls([CALL]) in read("after_tool_call")


def test_tool_result_wrapping():
    # Documents how the template wraps a `tool` message; the training
    # formatter (next plan) must produce the same thing.
    assert f"<tool_response>\n{RESULT}\n</tool_response>" in read("after_tool_call")
```

- [ ] **Step 5: 執行測試**

Run: `.venv/Scripts/python -m pytest tests/test_template_fixture.py -v`
Expected: 3 passed。若只有 `test_tool_result_wrapping` 失敗，代表模板包裝工具結果的方式不同：打開 fixture 看實際格式，把斷言改成實際格式，並在 `docs/spike-w1.md` 記下實際格式（Plan 2 的訓練格式化要照這個格式）。

- [ ] **Step 6: B0 冒煙（原始模型、mock 工具）**

依序執行，記錄每題的 `stopped_reason` 與呼叫了哪些工具：
```bash
.venv/Scripts/twagent chat "明天台中會下雨嗎" --today 2026-09-23
.venv/Scripts/twagent chat "週末要去玩，幫我看禮拜六台北跟花蓮的天氣" --today 2026-09-23
.venv/Scripts/twagent chat "看一下我後天有什麼行程，然後幫我加一個晚上七點到九點在信義區的聚餐" --today 2026-09-23
.venv/Scripts/twagent chat "幫我查高鐵" --today 2026-09-23
.venv/Scripts/twagent chat "幫我訂一張明天去東京的機票" --today 2026-09-23
```
Expected：每題都能跑完且不丟例外（`stopped_reason` 可以是任何值；這是基準線，不要求答對）。若出現 `LLMError`，確認伺服器仍在執行。

- [ ] **Step 7: 補完紀錄並 Commit**

在 `docs/spike-w1.md` 的「模板對照」一節寫下 Step 2 的結果；「B0 冒煙結果」一節用表格列出 Step 6 五題的 `stopped_reason`、實際工具呼叫、是否正確，以及觀察到的問題（例如輸出簡體字、日期算錯、沒有追問）。這些觀察會成為 Plan 2 資料設計的依據。

```bash
git add scripts/check_template.py tests/fixtures tests/test_template_fixture.py docs/spike-w1.md src tests
git commit -m "test: lock prompt rendering to the official Qwen template; record B0 smoke run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: 手寫測試集 100 筆（使用者親自撰寫）

這一步**由使用者本人（或朋友）撰寫**，agent 只負責驗證與提醒規則，**不可代寫題目**（否則就變成 LLM 風格的考卷）。

**Files:**
- Modify: `data/handwritten_test.jsonl`（從 6 筆種子題擴充到 100 筆）

- [ ] **Step 1: 請使用者依 `data/README.md` 的規則與份數分配撰寫題目**

- [ ] **Step 2: 驗證**

Run: `.venv/Scripts/twagent check-cases data/handwritten_test.jsonl`
Expected：沒有任何 `ERROR`，最後一行為
`100 valid case(s): clarify=15, no_tool=15, parallel=15, sequential=15, single=35, tool_error=5`

有錯誤時，把錯誤訊息回報給使用者修正，不要自行改題目內容（格式錯誤例外，例如缺逗號，可以修並告知使用者）。

- [ ] **Step 3: 確認全部測試仍通過**

Run: `.venv/Scripts/python -m pytest`
Expected: 全部 PASS（`test_handwritten_seed_file_is_valid_and_covers_every_category` 會讀這份檔案）。

- [ ] **Step 4: Commit**

```bash
git add data/handwritten_test.jsonl
git commit -m "data: add 100 handwritten test cases

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
