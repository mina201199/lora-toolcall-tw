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
