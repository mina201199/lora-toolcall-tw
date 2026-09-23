# W1 硬體驗證紀錄

日期：2026-09-23

## 結論
- 採用的執行方式：A（Docker 版 llama-server，`ghcr.io/ggml-org/llama.cpp:server-cuda`）
- 原因：方案 A 第一次嘗試即成功——`docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` 能看到 GTX 1050 Ti；`ghcr.io/ggml-org/llama.cpp:server-cuda` 映像內建 CUDA 12.8.1（非 CUDA 13），與 Pascal（compute 6.1）相容；兩個模型（4B、1.7B）皆能以 `-ngl 99` 全部層 offload 到 GPU 並正常產生回應，VRAM 用量都在 4 GB 上限之內，因此不需要嘗試方案 B/C/D。

## 模型
| 模型 | GGUF 檔 | 顯存用量 | tokens/s（生成） | 全部層 offload 到 GPU？ |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 Q4_K_M | `models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`（2.5 GB） | 3691 MiB / 4096 MiB（基準閒置約 608–653 MiB，模型載入後增加約 3.0 GB） | 20.42（`predicted_per_second`，見下方 evidence） | 是（`-ngl 99`，模型載入成功、`/health` 為 ok，且 VRAM 明顯增加；本建置的預設 log 詳細度未印出逐層 offload 行，以 VRAM 增量與成功推論作為佐證） |
| Qwen3-1.7B Q4_K_M | `models/Qwen3-1.7B-Q4_K_M.gguf`（1.1 GB） | 2225 MiB / 4096 MiB（增加約 1.6 GB） | 40.85（`predicted_per_second`） | 是（同上，`-ngl 99`） |

備註：Qwen3-1.7B 預設為思考模式（thinking），smoke test 回應中出現 `reasoning_content` 且在 `max_tokens=64` 內被截斷（`finish_reason: "length"`），這是模型行為本身、非效能問題；tokens/s 數值不受影響。

## 各方案結果
| 方案 | 結果 | 錯誤訊息／備註 |
|---|---|---|
| A Docker CUDA | 成功 | `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` 輸出中出現 `NVIDIA GeForce GTX 1050 Ti`；`ghcr.io/ggml-org/llama.cpp:server-cuda` 容器內 `CUDA_VERSION=12.8.1`；4B 與 1.7B 兩個模型皆成功啟動、回應請求、VRAM 用量正常，未出現 `no kernel image is available` 等錯誤。 |
| B 原生 CUDA 12 | 未嘗試（方案 A 已成功，依決策略過） | — |
| C 原生 Vulkan | 未嘗試（方案 A 已成功，依決策略過） | — |

## LoRA 支援
- `/lora-adapters` 端點：存在，回傳 `[]`（尚未載入任何 adapter），實測指令：`curl -s localhost:8080/lora-adapters` → `[]`
- `--lora-init-without-apply`：`--help` 輸出中存在，說明為「load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: disabled)」；同時確認 `--lora FNAME` 與 `--lora-scaled FNAME:SCALE,...` 也存在。實際的每請求 scale 測試待 Plan 3 有 adapter 之後再做。

## 模板對照

用 `scripts/check_template.py` 對照 `llama-server`（`--jinja`）的 `/apply-template` 端點：對每個情境送出「官方」版本（`tools` 欄位 + 結構化 `assistant.tool_calls`）與「我們自己」版本（工具說明塞進 system 純文字、工具呼叫寫成 `<tool_call>` 純文字），比對兩者渲染出的 prompt 字串。

執行 `.venv/Scripts/python scripts/check_template.py --server http://localhost:8080`：

```
[same] first_turn
[same] after_tool_call
```

兩個情境都是 `[same]`，沒有出現 `[DIFF]`：`prompting.render_tools_block` 與 `parsing.render_tool_calls` 目前手刻的渲染方式，字元對字元等同於 Qwen3 官方 chat template（`tokenizer_config.json` 內建的 Jinja 模板）在有 `tools` 時的輸出。因此不需要修改 `src/twagent/prompting.py` 或 `src/twagent/parsing.py`，`tests/test_prompting.py`、`tests/test_parsing.py` 的既有斷言也不用變動。

用 `.venv/Scripts/python scripts/check_template.py --save tests/fixtures` 把官方渲染結果存成 `tests/fixtures/qwen_first_turn.txt`、`tests/fixtures/qwen_after_tool_call.txt`（exit code 0），並用 `tests/test_template_fixture.py` 鎖住這個結果（3 個測試全過）。順便確認了工具結果（`tool` 訊息）在模板裡的實際包法：`<tool_call>` 後面的下一輪不是另一個 `tool` role 區塊，而是被包成 `<|im_start|>user\n<tool_response>\n{...}\n</tool_response><|im_end|>`——也就是說 Qwen3 的模板把工具回傳結果併入下一個 `user` 回合，而不是產生獨立的 `tool` 回合標記。Plan 2 的訓練資料格式化（下一個 plan）需要照這個實際格式組訓練樣本，而不是照 OpenAI 風格「每個 tool 訊息各自一個 role」的直覺想法。

## B0 冒煙結果

原始模型（Qwen3-4B-Instruct-2507 Q4_K_M，未微調）、mock 工具、`temperature=0.0`，五題皆用 `.venv/Scripts/twagent chat "<問題>" --today 2026-09-23 --json` 執行，皆正常跑完、沒有拋出例外或 `LLMError`。

| # | 問題 | `stopped_reason` | 實際呼叫的工具 | 是否正確 | 觀察 |
|---|---|---|---|---|---|
| 1 | 明天台中會下雨嗎 | answer | `get_weather(city=臺中市, date=2026-09-24)` | 正確（明天 = 2026-09-24 週四，日期換算對；「台中」正確正規化為「臺中市」） | 工具參數用正式「臺中市」，但最終回覆的白話文字寫成「台中市」（非正式簡寫），前後用字不一致；文字皆為繁體字，未見簡體字 |
| 2 | 週末要去玩，幫我看禮拜六台北跟花蓮的天氣 | answer | `get_weather(city=臺北市, date=2026-09-26)`、`get_weather(city=花蓮縣, date=2026-09-26)` | 正確（禮拜六 = 2026-09-26 週六，日期換算對；「台北」「花蓮」正確正規化為「臺北市」「花蓮縣」；一次回合內連續呼叫兩個工具） | 條列處用正式「臺北市」「花蓮縣」，但開頭敘述文字又寫成「台北」；回覆結尾加了表情符號 😊 |
| 3 | 看一下我後天有什麼行程，然後幫我加一個晚上七點到九點在信義區的聚餐 | answer | `calendar_list(date=2026-09-25)`、`calendar_add(...)`（第一次失敗，第二次成功） | 大致正確（後天 = 2026-09-25 週五，日期換算對；有先查詢再新增，順序對） | 第一次 `calendar_add` 的 `start`/`end` 帶了秒數（`2026-09-25T19:00:00`），不符合 schema 的 `YYYY-MM-DDTHH:MM` pattern，觸發 `schema_error`；模型在下一輪自行拿掉秒數重送並成功——這是 baseline 模型常見的 datetime 格式錯誤，`schema_retries` 機制確實接住了它，但也代表訓練資料需要示範「不要帶秒數」這個細節 |
| 4 | 幫我查高鐵 | answer | （無） | 正確（資訊不足時沒有直接猜測起訖站或日期，而是依 persona 指示向使用者追問） | 沒有呼叫工具、沒有胡亂假設參數，符合「先追問」的人設要求 |
| 5 | 幫我訂一張明天去東京的機票 | answer | （無） | 正確（訂機票超出工具能力範圍，直接說明做不到，並建議透過其他管道） | 沒有嘗試呼叫不存在的工具、沒有幻覺出一個訂票結果 |

整體觀察（給 Plan 2 資料設計參考）：
- 相對日期換算（明天／後天／禮拜六）在這五題中全部正確，沒有算錯的情形。
- 城市名稱正規化在「工具呼叫參數」裡穩定使用正式「臺」字（臺中市、臺北市、花蓮縣），但在「給使用者看的最終回覆文字」裡會混用非正式的「台」字——這是訓練資料需要額外要求覆蓋的地方（persona 目前只講「一律使用繁體中文與臺灣用語回覆」，沒有強制最終文字也要用正式「臺」字）。
- 沒有觀察到簡體字輸出。
- 唯一一次 `schema_error` 出現在日期時間欄位帶了秒數；模型能自行從錯誤訊息修正並重試成功，顯示 `agent.run_agent` 的 schema 重試機制運作正常，但也顯示 baseline 模型預設會多寫秒數，訓練樣本裡的 `calendar_add` 範例需要清楚示範 `HH:MM`（不含秒）格式。
- 對於資訊不足（高鐵）與超出能力範圍（訂機票）兩種情境，baseline 模型都沒有胡亂猜測或幻覺工具，行為已經符合 persona 要求；這兩類「拒絕/追問」情境仍應保留在訓練資料中以強化這個行為，避免微調後退化。
