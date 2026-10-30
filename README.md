# lora-toolcall-tw

繁體中文 Function Calling Agent（臺灣生活助理）：用 QLoRA 微調小型 Qwen 模型，
讓它把繁中自然語言可靠地轉成工具呼叫，量化後在 4 GB 顯卡（GTX 1050 Ti）上本機部署。

> **目前進度：資料管線完成；QLoRA 微調、評估、量化部署尚未進行。**
> repo 內還沒有訓練程式（沒有 Unsloth／PEFT／TRL 相依），也沒有微調模型或評估結果。
> 另已完成硬體驗證：原始 Qwen3 模型可在 GTX 1050 Ti 上用 llama-server 執行。
> 本頁只列 repo 內有紀錄的數字，尚未完成的部分不填結果。

## 架構

```
訓練端（Kaggle / Colab T4）
  工具定義 → 老師模型合成對話軌跡 → 驗證過濾 → 去重 → 分組切分 → 訓練資料
                                                          ↓
                                           Unsloth QLoRA 微調 Qwen3     ← 尚未進行
                                                          ↓
部署端（本機 GTX 1050 Ti）                      合併 → GGUF → 量化     ← 尚未進行
  llama.cpp llama-server（Docker, CUDA）＋ twagent Agent 迴圈 ＋ mock 工具
```

工具場景：天氣、高鐵、台鐵、YouBike、統一發票、行事曆（本機 SQLite）。

## 進度

| 階段 | 狀態 | 內容 |
|---|---|---|
| 1. 基礎與硬體驗證 | ✅ 完成 | Agent 迴圈、工具 schema、mock 工具、相對日期換算、CLI；在 1050 Ti 上跑通 llama-server |
| 2. 資料管線 | ✅ 完成 | 合成、驗證、去重、分組切分的程式與測試；正式合成資料未納入版控（見下） |
| 3. QLoRA 訓練與評估 | ⏳ 尚未進行 | repo 內尚無訓練程式；規劃與 B0（原始模型）、B1（few-shot）對照 |
| 4. 部署與 demo | ⏳ 尚未進行 | 規劃 GGUF 量化、FastAPI ＋ Docker Compose，原廠 vs 微調並排比較 |

## 已完成的實測結果

### 硬體驗證（詳見 [`docs/spike-w1.md`](docs/spike-w1.md)）

| 模型（Q4_K_M） | VRAM 用量 | 生成速度 |
|---|---|---|
| Qwen3-4B-Instruct-2507 | 3691 / 4096 MiB | 20.4 tokens/s |
| Qwen3-1.7B | 2225 / 4096 MiB | 40.9 tokens/s |

- 兩個模型都能全部層 offload 到 GPU（`-ngl 99`）。
- 自己實作的 prompt 渲染跟 GGUF 內嵌的 chat template 逐字元比對一致，並用 fixture 測試鎖住。
- 原始模型冒煙測試發現的問題（例如 datetime 多帶秒數、回覆裡「臺／台」混用）已回饋到資料設計。

### 合成資料（詳見 [`docs/data-report.md`](docs/data-report.md)）

**正式合成資料未納入版控。** `data/synthetic/` 列在 `.gitignore`，repo 內也沒有正式合成的執行紀錄，
因此這裡不列正式合成、驗證後與切分後的筆數；數據待資料重建後再補。
repo 內有紀錄的是兩次各 50 筆的老師模型試算（保留率、花費與退件原因見 `docs/data-report.md`），
以及 100 筆手寫測試題（`data/handwritten_test.jsonl`）。

資料管線的設計重點：

- **驗證器**：用 mock 工具實際執行每一筆軌跡，檢查工具名稱與參數、相對日期換算、是否該追問、
  繁中用語（OpenCC），不合格就退件。
- **成本控管**：先小批量試算保留率與每筆成本，再推估正式合成量；有預算上限、每批寫入成本帳本，
  中斷後可以續跑。
- **分組切分**：依情境（類別＋工具組合＋說話風格）分組後再切 train / valid / test，同一組只會落在同一個 split。

## 執行方式

```bash
uv venv .venv
uv pip install -p .venv -e ".[dev]"
.venv/Scripts/python -m pytest          # Windows；Linux/macOS 用 .venv/bin/python
```

CI（GitHub Actions，CPU only）會在 Python 3.10–3.12 執行 `ruff` 與全部測試（不含需要網路或 llama-server 的 `live` 測試）。

CLI：

```bash
twagent chat "明天台中會下雨嗎" --today 2026-09-23 --json   # 需要啟動 llama-server
twagent synth ...            # 用老師模型合成軌跡（需要 OPENAI_API_KEY，放在 .env）
twagent build-dataset ...    # 驗證、去重、切分、格式化
twagent check-cases ...      # 檢查測試案例檔
```

## 環境變數

只有 `twagent synth` 需要環境變數；測試、`build-dataset`、`check-cases` 與 `chat` 都不需要。

| 變數 | 用途 |
|---|---|
| `OPENAI_API_KEY` | `twagent synth` 呼叫 OpenAI 老師模型合成資料時使用。可設在環境變數，或寫在執行目錄的 `.env`（已列在 `.gitignore`）；`.env` 不會覆蓋已設定的環境變數。 |

`chat` 連線的 llama-server 位址（`llm.base_url`）在 `configs/default.yaml` 設定，不從環境變數讀取。

設計規格與實作計畫：`docs/superpowers/specs/`、`docs/superpowers/plans/`。
