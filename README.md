# lora-toolcall-tw

繁體中文 Function Calling Agent（臺灣生活助理）：用 QLoRA 微調小型 Qwen 模型，
量化後在 4 GB 顯卡上本機部署。設計見 `docs/superpowers/specs/`。

```bash
uv venv .venv
uv pip install -p .venv -e ".[dev]"
.venv/Scripts/python -m pytest          # Windows；Linux/macOS 用 .venv/bin/python
```
