# W1 硬體驗證紀錄

日期：2026-09-23

## 結論
- 採用的執行方式：A（Docker 版 llama-server，`ghcr.io/ggml-org/llama.cpp:server-cuda`）
- 原因：方案 A 第一次嘗試即成功——`docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` 能看到 GTX 1050 Ti；`ghcr.io/ggml-org/llama.cpp:server-cuda` 映像內建 CUDA 12.8.1（非 CUDA 13），與 Pascal（compute 6.1）相容；兩個模型（4B、1.7B）皆能以 `-ngl 99` 全部層 offload 到 GPU 並正常產生回應，VRAM 用量都在 4 GB 上限之內，因此不需要嘗試方案 B/C/D。

## 模型
| 模型 | GGUF 檔 | 顯存用量 | tokens/s（生成） | 全部層 offload 到 GPU？ |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 Q4_K_M | `models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`（2.5 GB） | 3691 MiB / 4096 MiB（基準閒置約 615–653 MiB，模型載入後增加約 3.0 GB） | 20.42（`predicted_per_second`，見下方 evidence） | 是（`-ngl 99`，模型載入成功、`/health` 為 ok，且 VRAM 明顯增加；本建置的預設 log 詳細度未印出逐層 offload 行，以 VRAM 增量與成功推論作為佐證） |
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

## 模板對照（Task 12 填寫）

（Task 12 填寫）

## B0 冒煙結果（Task 12 填寫）

（Task 12 填寫）
