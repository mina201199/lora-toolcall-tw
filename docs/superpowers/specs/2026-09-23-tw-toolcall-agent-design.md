# 繁中 Function Calling Agent（台灣生活助理）— 設計規格

- 日期：2026-09-23
- 專案代號：`lora-toolcall-tw`（Python 套件名 `twagent`）
- 用途：AI 應用 / 後端工程師職缺的面試作品
- 參考專案：`C:\Users\User\Desktop\Lora實戰\astro-vlm-lora`（沿用其工程慣例與評估守則）

---

## 1. 目標與定位

用 QLoRA 微調一個小型 Qwen 模型，讓它把**繁體中文自然語言**可靠地轉成**工具呼叫**，
並把它做成一個能在**4 GB 舊顯卡（GTX 1050 Ti）上本機部署**的 Agent 服務。

一句話面試故事：

> 用 Kaggle 免費 T4 微調小模型，量化後在 4 GB 顯卡上部署一個繁中工具呼叫 Agent；
> 並用公平對照組（含 few-shot 提示與大模型 API）量測它的準確率、延遲與成本。

與 `astro-vlm-lora` 的分工：該專案展示**評估嚴謹度（bug hunt）**；本專案展示**端到端系統能力**。
兩者在作品集中互補，不重複。

### 成功標準

1. 在手寫測試集上，微調模型的「整筆完全正確率」**明顯高於** B0（原始模型）與 B1（few-shot 提示）。
   具體數字由實驗決定，README 只填實際跑出來的結果。
2. `docker compose up` 後可在本機 1050 Ti 上用網頁 demo 並排比較原廠 vs 微調。
3. CI（CPU only）通過 ruff 與全部測試。
4. 每一個回報的數字都能追回產生它的設定檔（設定完整嵌入報告）。

### 非目標（YAGNI）

- 不做 DPO / 偏好優化、不做 multi-adapter 路由、不做監控儀表板、不做 CD。
- 不做使用者帳號、權限、多租戶。
- 不支援訂票、付款等**有副作用的真實外部操作**；唯一的寫入操作是本機 SQLite 行事曆。

---

## 2. 已確定的決策

| 項目 | 決定 |
|---|---|
| 基底模型 | Qwen3 系列：主力 4B，備援 1.7B（實作前確認 Hugging Face 上的最新可用版本） |
| 工具呼叫格式 | Qwen 原生 chat template 的 `<tool_call>{...}</tool_call>` 格式 |
| 訓練 | Unsloth QLoRA，Kaggle（每週 30 小時 T4）或 Colab 免費 T4，fp16（T4 不支援 bf16） |
| 部署 | 合併 LoRA → GGUF → 量化；本機 llama.cpp（`llama-server`）＋ FastAPI ＋ Docker |
| 資料 | Claude API 合成，預算 5–15 美元；先小批量實測成本再放大 |
| 場景 | 台灣生活助理：天氣、高鐵、台鐵、YouBike、發票、行事曆 |
| 範圍 | 完整版，4 週 |
| 約束解碼 | llama.cpp GBNF 語法約束，作為**對照實驗**而非預設 |

---

## 3. 系統架構

```
┌──────────── 訓練端（Kaggle / Colab T4）────────────┐
│ 工具定義 → Claude 合成 → 驗證過濾 → 分組切分 → 資料集 │
│                                   ↓                │
│                  Unsloth QLoRA 微調 Qwen            │
│                                   ↓                │
│          合併 LoRA → 轉 GGUF → 量化（Q8 / Q4_K_M）   │
└────────────────────────────────────────────────────┘
                         ↓ 下載模型檔
┌──────────── 部署端（本機 1050 Ti + Docker）──────────┐
│  web（並排對照） ⇄  api（FastAPI）  ⇄  llama-server   │
│                        ↓                            │
│             工具執行層：mock / live                   │
│             失敗時可選擇退回 Claude                   │
└────────────────────────────────────────────────────┘
```

### 專案結構

```
lora-toolcall-tw/
├── src/twagent/
│   │  ── 純邏輯層（無 torch / unsloth import，CPU 可測）──
│   ├── config.py        讀取 YAML 設定
│   ├── tools/
│   │   ├── schemas.py   六個工具的 JSON Schema（唯一定義來源）
│   │   ├── mock.py      確定性的 mock 實作（評估用）
│   │   ├── live.py      真實 API 實作（demo 用）
│   │   └── registry.py  依 TOOLS_MODE 選擇實作
│   ├── dates.py         相對日期與民國期別的換算（今天日期由外部注入）
│   ├── data/
│   │   ├── synth.py     組裝合成請求、呼叫 Claude
│   │   ├── validate.py  schema、列舉、日期、簡繁、近似重複檢查
│   │   ├── splits.py    分組切分 ＋ 洩漏護欄（SplitResult.__post_init__）
│   │   └── format.py    轉成 Qwen chat template 的訓練樣本
│   ├── parsing.py       唯一的 <tool_call> 解析器
│   ├── agent.py         唯一的 Agent 迴圈
│   ├── llm.py           LLM 客戶端介面（OpenAI 相容；llama-server 與 Claude 各一實作）
│   ├── metrics.py       指標 ＋ sanity checks
│   ├── evaluation.py    跑對照組、輸出報告
│   │  ── 加速層（延遲 import，只在 GPU 機器上用）──
│   ├── train.py         QLoRA 訓練，含 validation loss
│   └── export.py        合併、轉 GGUF、量化
├── server/app.py        FastAPI：只做接線，不含推論邏輯
├── web/index.html       並排對照 demo（參考 ui-preview.html 的版面）
├── notebooks/train_kaggle.ipynb
├── configs/default.yaml
├── data/
│   ├── handwritten_test.jsonl   手寫測試集（人工維護，進版控）
│   └── (synthetic/ 產出物，不進版控)
├── reports/             評估報告 JSON（嵌入完整設定）
├── tests/               CPU only
├── docker-compose.yml
├── .github/workflows/ci.yml
└── pyproject.toml       extras：data / train / eval / server / dev
```

### 架構原則

1. **純邏輯層與加速層分離**（沿用 astro-vlm-lora）：資料驗證、解析、Agent 迴圈、指標全部能在 CPU 上測試。
2. **推論只有一條路徑**：`parsing.py` 與 `agent.py` 各只有一份，評估程式與 FastAPI 服務都呼叫它們。
3. **工具層與模型層解耦**：`llm.py` 使用 OpenAI 相容介面，切換 llama-server / Claude 只改 `base_url` 與金鑰。
4. **mock 與 live 分離**：評估一律用 mock（結果可重現）；demo 才用 live。
5. **設定即紀錄**：`configs/default.yaml` 完整嵌入每份報告。

---

## 4. 工具定義

| 工具 | 參數 | live 資料來源 | 刻意設計的難點 |
|---|---|---|---|
| `get_weather` | `city`（列舉：22 縣市）, `date`（YYYY-MM-DD） | 中央氣象署開放資料 | 相對日期換算（「明天」「下週五」） |
| `search_thsr` | `origin`, `destination`（列舉：12 站）, `date`, `depart_after`（HH:MM，選填） | TDX | 站名正規化（台北/臺北、高雄→左營） |
| `search_tra` | `origin`, `destination`, `date`, `depart_after`（選填）, `train_type`（列舉，選填） | TDX | 與高鐵區分（「火車」「自強號」→ 台鐵） |
| `find_youbike` | `city`, `near`（自由文字） | TDX | 模糊地點（「台大附近」） |
| `check_invoice` | `number`（8 碼）, `period`（例：`115-07-08`，民國年-起月-迄月） | 財政部開放資料 | 期別換算（「上一期」）；期別為雙月，起始月必須是奇數 |
| `calendar_add` | `title`, `start`, `end`（ISO 8601）, `location`（選填） | 本機 SQLite | 常接在其他工具之後（循序多步） |
| `calendar_list` | `date` | 本機 SQLite | — |

- 系統提示固定注入「今天日期（含星期）」與時區 `Asia/Taipei`。
- 所有 schema 只定義在 `tools/schemas.py`；合成、驗證、評估、服務都從這裡讀。
- mock 實作必須**確定性**：相同輸入永遠回傳相同輸出（以參數雜湊作為亂數種子）。

---

## 5. 資料

### 5.1 資料類型與比例（訓練集目標約 3,000 筆）

| 類型 | 比例 | 範例 |
|---|---|---|
| 單一工具呼叫 | 35% | 「明天台中會下雨嗎」 |
| 平行多工具 | 15% | 「台北和高雄明天天氣」 |
| 循序多步（依賴前一步結果） | 15% | 「查明早去台南的高鐵，排進行事曆」 |
| 缺參數追問 | 15% | 「幫我查高鐵」→ 追問起訖站 |
| 不需工具 / 不支援 | 15% | 閒聊；「幫我訂機票」→ 說明做不到 |
| 工具錯誤處理 | 5% | 工具回傳逾時 → 告知使用者 |

### 5.2 合成流程

1. **情境矩陣**：資料類型 × 工具組合 × 說話風格（口語、錯字、中英夾雜、冗長）× 模板編號。
2. **Claude 產生**「使用者問題 ＋ 標準答案工具呼叫」（要求結構化 JSON 輸出）。
3. **程式執行 mock 工具**產生工具結果——不讓 LLM 編造工具結果，確保一致。
4. **Claude 根據工具結果**寫最終回覆（繁中、台灣用語）。
5. **驗證**（`data/validate.py`），任一項失敗即丟棄並計數：
   - 參數符合 JSON Schema、列舉值合法
   - 日期與期別換算正確（以 `dates.py` 重算比對）
   - OpenCC `s2twp` 檢查：轉換前後不一致即視為含簡體字或中國用語
   - MinHash 近似重複去除
6. **分組切分**（見 5.3）。
7. **格式化**為 Qwen chat template（含 `tools`），產出 train / valid / test JSONL。

- 產生器模型：預設 `claude-sonnet-5`；先以 50 筆實測成本，若推估總成本超過 15 美元則改用 `claude-haiku-4-5-20251001`。
- 所有丟棄原因與數量寫進 `reports/data_report.json`。

### 5.3 切分與洩漏護欄

- 分組鍵：`(資料類型, 工具組合, 模板編號)`。同一組只會出現在同一個 split。
- 比例：train 80% / valid 10% / synthetic-test 10%（以組為單位）。
- `SplitResult.__post_init__` 執行 `assert_no_leakage`：
  - 各 split 分組鍵兩兩不相交
  - 跨 split 的 MinHash 相似度不得超過門檻（預設 0.8）
- 無法建構出有洩漏的 `SplitResult` 物件。

### 5.4 兩份測試集

| 測試集 | 規模 | 來源 | 衡量什麼 |
|---|---|---|---|
| synthetic-test | 約 300 筆 | 5.3 切出（模板未見過） | 學習效果 |
| handwritten-test | 100 筆 | 使用者本人或朋友手寫，覆蓋 5.1 六類 | 真實泛化能力 |

兩者的分數落差本身是報告中的一項發現。

---

## 6. 訓練

- Unsloth `FastLanguageModel`，`load_in_4bit=True`，`use_gradient_checkpointing="unsloth"`。
- 版本鎖定沿用參考專案在 T4 上驗證過的組合（`transformers==4.56.2`、`trl==0.22.2`），若 Qwen3 需要更新版本，於 W1 確認後更新並記錄原因。
- 預設 LoRA：`r=16`、`lora_alpha=16`、target 全部線性層、`lora_dropout=0`。
- 預設訓練：`num_train_epochs=2`、有效 batch 16、`learning_rate=2e-4`、`adamw_8bit`、`warmup_ratio=0.05`。
- **有 validation loss**（`eval_strategy=steps`）。
- **Loss 遮罩**：只對 assistant 輪次計算 loss；system、user、工具結果（`tool` 角色）一律 `-100`。有單元測試驗證。
- 輸出：adapter（進 `outputs/`）、`train_report.json`（含 loss 曲線與完整設定）。

---

## 7. 評估

### 7.1 指標

| 層級 | 指標 | 定義 |
|---|---|---|
| 格式 | 解析成功率 | 輸出中的 `<tool_call>` 皆能解析為 JSON |
| 格式 | Schema 合法率 | 解析後參數通過 JSON Schema |
| 決策 | 三類混淆矩陣 | 呼叫工具 / 追問 / 直接回答 |
| 呼叫 | 函式名稱準確率 | 平行呼叫以多重集合比對 |
| 呼叫 | 參數準確率 | 逐參數比對（先正規化日期、站名） |
| 呼叫 | 整筆完全正確率 | 該筆所有呼叫的名稱與參數全對 |
| 任務 | 端到端成功率 | 以 mock 工具跑完整個 Agent 迴圈，檢查最終狀態（如 SQLite 行事曆內容） |
| 回覆 | 簡體字比例 | OpenCC 自動判斷 |
| 回覆 | 忠實度 / 台灣用語 | LLM 評分，僅供參考 |
| 系統 | 首字延遲、tokens/s、每千次請求成本 | 本機 1050 Ti vs Claude API |

**主要結論一律以確定性指標為準**（格式、決策、呼叫、任務）。

### 7.2 對照組

| 組別 | 內容 |
|---|---|
| B0 | 原始 Qwen，零樣本 |
| B1 | 原始 Qwen ＋ few-shot 提示（每類 1 例） |
| B2 | Claude API |
| FT | 微調後 Qwen |
| FT+G | 微調後 Qwen ＋ GBNF 語法約束 |

所有組別使用**完全相同**的測試資料、系統提示、工具定義與解碼設定（greedy，明確 `do_sample=False` / `temperature=0`）。

### 7.3 消融實驗

必做：
1. 模型大小：1.7B vs 4B
2. 資料量：500 / 1,500 / 3,000 筆（學習曲線）
3. 量化損失：fp16 vs Q8_0 vs Q4_K_M
4. 語法約束：FT vs FT+G

有餘力再做：
5. LoRA 設定：r = 8 / 16 / 64；attention-only vs 全部線性層
6. 資料類型消融：移除「追問」類，觀察追問率變化

### 7.4 自動 sanity checks

每次評估都執行，結果寫入報告的 `findings` 欄位：

- train/test 分組重疊 → **error**（中止）
- 工具呼叫率高於標準答案比例 15 個百分點以上 → warning（「什麼都呼叫」）
- 追問率為 0 而標準答案有追問 → warning
- 簡體字比例 > 1% → warning
- 評分 LLM 與資料產生器為同一模型家族 → 在報告標註偏誤警告
- 各對照組的測試資料雜湊不一致 → **error**

---

## 8. 部署

### 8.1 docker-compose

| 服務 | 內容 |
|---|---|
| `llama-server` | llama.cpp CUDA server 映像；載入 base GGUF ＋ LoRA adapter GGUF；啟用 `--jinja` |
| `api` | FastAPI（`server/app.py`） |
| `web` | 靜態頁面（`web/index.html`） |

- **原廠 vs 微調對照只載入一份 base 模型**：透過 llama-server 的每請求 LoRA scale（0 = 原廠、1 = 微調）。
  等同 astro-vlm-lora 的 `disable_adapter()` 做法。
- 顯存預算：4B Q4_K_M 約 2.5 GB ＋ KV cache 約 0.5 GB；context 預設 4096。
- 工具定義固定置於系統提示開頭以利 prompt cache。

### 8.2 API

| 端點 | 說明 |
|---|---|
| `POST /chat` | 輸入對話，回傳最終回覆與完整工具呼叫紀錄（trace） |
| `POST /chat/compare` | 同一輸入分別以 scale 0 / 1 執行，回傳兩份 trace |
| `GET /health` | llama-server 與工具層狀態 |

### 8.3 設定

| 環境變數 | 值 | 說明 |
|---|---|---|
| `TOOLS_MODE` | `mock` / `live` | 工具實作 |
| `FALLBACK` | `off` / `claude` | 失敗時是否退回 Claude |
| `LLM_BASE_URL` | URL | llama-server 位址 |
| `ANTHROPIC_API_KEY`、`CWA_API_KEY`、`TDX_CLIENT_ID`、`TDX_CLIENT_SECRET` | — | 由使用者自行申請，放在 `.env`，不進版控 |

---

## 9. 錯誤處理

| 狀況 | 行為 |
|---|---|
| 輸出無法解析 | 附上錯誤訊息請模型重試 1 次 → 仍失敗且 `FALLBACK=claude` 則改用 Claude → 否則回覆友善錯誤訊息 |
| 參數不符 schema | 把驗證錯誤當作工具結果回給模型自我修正，最多 2 次 |
| 迴圈過長 | 每請求最多 5 步 |
| live API 逾時 / 失敗 | 如實告知使用者；**不得靜默改用 mock 資料** |
| 追蹤紀錄 | 每請求寫入 `logs/traces.jsonl`（輸入、每步輸出、是否 fallback、延遲） |

「fallback 比例」列為系統指標。

---

## 10. 測試

全部在 CPU 上執行，CI 不需要 GPU、模型下載或網路。

| 範圍 | 內容 |
|---|---|
| `parsing` | 合法、多個呼叫、截斷、多餘文字、非法 JSON；收錄實際觀察到的錯誤輸出作為案例 |
| `tools/schemas` + `validate` | 列舉、必填、型別；OpenCC 檢查；近似重複 |
| `dates` | 相對日期（含跨月、跨年、星期）、民國發票期別 |
| `splits` | 分組不相交、可重現、洩漏時拋出 `DataLeakageError` |
| `format` | loss 遮罩：只有 assistant 輪次非 `-100`（用假 tokenizer） |
| `agent` | 以腳本化假 LLM 測：重試、schema 修正、fallback、步數上限、live 失敗不改用 mock |
| `metrics` | 各指標與 sanity checks 的觸發條件 |
| `server` | FastAPI `TestClient` ＋ 假 LLM 的契約測試 |
| live | `@pytest.mark.live`，CI 跳過 |

CI：ruff lint + format check；pytest on Python 3.10 / 3.11 / 3.12。

---

## 11. 時程

| 週 | 工作 | 可交付成果 |
|---|---|---|
| W1 | **高風險驗證**：Docker + WSL2 GPU 在 1050 Ti 上跑 llama-server；每請求 LoRA scale；`--jinja` 解析 Qwen 工具呼叫。專案骨架、工具與 mock、`dates`、`parsing`、`agent`；手寫 100 筆測試題 | 原始 Qwen 在本機跑通端到端（B0） |
| W2 | 合成與驗證流程、分組切分；Kaggle 首次訓練；評估程式；B0 / B1 / B2 | 第一版比較表 |
| W3 | 消融 1–4（有餘力做 5–6）；GGUF 匯出與量化評估；GBNF | 完整實驗結果 |
| W4 | 服務收尾、並排對照網頁、docker-compose、README、錯誤分析、demo 影片 | 面試版本 |

---

## 12. 風險與備案

| 風險 | 備案 |
|---|---|
| WSL2 容器內 CUDA 不支援 Pascal（1050 Ti） | 改在 Windows 原生執行 llama-server 執行檔，其餘服務仍用 Docker |
| llama-server 不支援每請求 LoRA scale | 啟動兩個 llama-server（base 與 merged）並改用 1.7B 以符合顯存；或對照改為非同時執行 |
| llama.cpp 對 Qwen3 工具模板支援不完整 | 改用 `/completion` 端點，自行套 chat template，由 `parsing.py` 解析 |
| 4B Q4 放不進 4 GB | 降 context 至 2048，或改 1.7B 為主力 |
| 合成成本超出預算 | 改用 Haiku；或降訓練集至 2,000 筆 |
| Kaggle 額度不足以跑完消融 | 依 7.3 的優先順序截斷，未跑的在 README 標註「未量測」 |
