# lora-toolcall-tw

繁體中文 Function Calling Agent（臺灣生活助理）：用 QLoRA 微調小型 Qwen 模型，
讓它把繁中自然語言可靠地轉成工具呼叫，量化後在 4 GB 顯卡（GTX 1050 Ti）上本機部署。

> **狀態：進行中。** 硬體驗證與資料管線已完成；QLoRA 訓練、評估與部署尚未完成。
> 本頁只列實際跑出來的數字，尚未完成的部分不填結果。

## 架構

```
訓練端（Kaggle / Colab T4）
  工具定義 → 老師模型合成對話軌跡 → 驗證過濾 → 去重 → 分組切分 → 訓練資料
                                                          ↓
                                           Unsloth QLoRA 微調 Qwen3     ← 待完成
                                                          ↓
部署端（本機 GTX 1050 Ti）                      合併 → GGUF → 量化     ← 待完成
  llama.cpp llama-server（Docker, CUDA）＋ twagent Agent 迴圈 ＋ mock 工具
```

工具場景：天氣、高鐵、台鐵、YouBike、統一發票、行事曆（本機 SQLite）。

## 進度

| 階段 | 狀態 | 內容 |
|---|---|---|
| 1. 基礎與硬體驗證 | ✅ 完成 | Agent 迴圈、工具 schema、mock 工具、相對日期換算、CLI；在 1050 Ti 上跑通 llama-server |
| 2. 資料管線 | ✅ 完成 | 合成、驗證、去重、分組切分；正式合成已跑完 |
| 3. QLoRA 訓練與評估 | ⏳ 待完成 | 與 B0（原始模型）、B1（few-shot）對照 |
| 4. 部署與 demo | ⏳ 待完成 | FastAPI ＋ Docker Compose，原廠 vs 微調並排比較 |

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

| 項目 | 數字 |
|---|---|
| 老師模型產生的原始軌跡 | 6,446 筆，花費 $0.80（預算上限 $5） |
| 驗證通過並去重後 | 3,476 筆（退件 2,439、近似重複 531） |
| 切分（依情境分組，避免洩漏） | train 2,645 / valid 403 / test 428 |
| 手寫測試集 | 100 筆 |

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

設計規格與實作計畫：`docs/superpowers/specs/`、`docs/superpowers/plans/`。
