# Plan 1 遺留事項（給 Plan 2–4 撰寫時參考）

Plan 1（基礎建設與 W1 驗證）完成時，最終審查與各任務審查中「延後處理」的項目。
撰寫後續計畫時，把對應項目排進任務裡。

## W1 驗證的關鍵結論（詳見 `docs/spike-w1.md`）
- 採用方案 A：Docker `ghcr.io/ggml-org/llama.cpp:server-cuda`（CUDA 12.8.1）在 GTX 1050 Ti 上可用。
- Qwen3-4B-Instruct-2507 Q4_K_M：約 20.4 tok/s、顯存 3691 MiB / 4096 MiB（很緊）。
- Qwen3-1.7B Q4_K_M：約 40.9 tok/s、顯存 2225 MiB。
- 我們渲染的提示詞與 GGUF 內建模板逐字相同（4B）。工具結果在模板中被包成
  `<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>`。
- B0 冒煙：原始 4B 在 5 題簡單題上表現已不錯；問題是 `calendar_add` 會帶秒數、
  最終回覆「台／臺」混用。微調效果需靠更難、更口語的手寫題顯現。

## Plan 2（資料合成、驗證、切分、訓練格式化）
- 訓練格式化必須重現模板實際格式（見上），並以 `tests/fixtures/` 對照。
- `scripts/check_template.py` 補兩個情境：平行多呼叫後接連續兩個 `tool` 訊息、
  「文字＋呼叫」後再一輪（最終審查已對 live server 手動驗證為相同，但未入測試）。
- `TIME` pattern 仍接受結尾換行（例如 `"14:30\n"`），在 `validate.py` 或 schemas 補檢查。
- 合成資料要示範 `calendar_add` 時間不帶秒數、最終回覆也使用正式「臺」字。
- 「參數依賴前一步結果」的循序情境由合成測試集涵蓋（標準答案由執行 mock 產生）；
  手寫集的 `sequential` 只測順序（見 `data/README.md` 規則 4）。

## Plan 3（訓練、匯出、評估）
- **第一個任務**：`LLMClient.complete` 改回傳 `Completion(text, finish_reason, usage, latency_s)`，
  並寫進 trace；`finish_reason == "length"` 視為截斷。規格 §7.1 的延遲、tokens/s、成本需要它。
- 定義「追問 vs 直接回答」的確定性判定規則（`AgentResult` 目前無法區分）。
- 對 Qwen3-1.7B（混合思考模型，`enable_thinking=False` 會在生成提示放空的 `<think>` 區塊）
  重跑模板對照，fixture 標明來源模型。
- mock 工具加一個 golden-value 測試，鎖住輸出以保證可重現性。

## Plan 4（FastAPI、網頁、live 工具、fallback）
- `MockTools` 每個請求建立一次：它的 SQLite 連線綁定建立它的執行緒，FastAPI threadpool 不能共用。
- `OpenAICompatClient` 的 `httpx.Client` 需要 `close()` 或 context manager。
- 規格 §8.1「工具定義放系統提示開頭」的描述不成立（Qwen 模板固定把 system 內容放在 `# Tools` 之前），
  改寫規格；每天日期變動只造成一次 cache miss。

## 小問題（有空再修）
- `twagent chat` 在 `llm_error` / `parse_failure` 時仍回傳 exit code 0。
- `--today` 格式錯誤時是原始 traceback；`DEFAULT_CONFIG` 依賴目前工作目錄。
- mock：`getattr(self, f"_{name}")` 分派可改為由 `tool_names()` 建的對照表；`depart_after="23:30"`
  會產生跨午夜卻同日期的班次；THSR/TRA 迴圈結構重複。
- `config._build` 遇到 YAML 區段是純量時丟 `TypeError` 而非 `ValueError`。
- `roc_year` 缺 docstring；`"0天後"` 會被接受。
- commit `332bd60` 的署名行是 `Claude Haiku 4.5`（未改寫歷史）。
- 本機 `.venv` 實際為 Python 3.14.6（`uv` 選的），計畫文件寫的是 3.13；CI 測 3.10–3.12。
