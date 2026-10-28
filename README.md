# Support Backend

A local AI support agent. It answers questions from `data/Support Portal Flow.pdf`,
and when it cannot answer, it raises a support ticket instead of guessing.

Nothing leaves your machine — the model runs locally.

## Run it

```bash
./run.sh
```

Then open **http://127.0.0.1:8000** in a browser. Stop it with `Ctrl+C`.

First run takes a few minutes (it builds `llama-cpp-python` from source).
After that it starts in seconds.

## The AI model

The weights are ~2.8 GB, so they are not in git (`.gitignore` excludes `*.gguf`).
Download once into the project root:

```bash
curl -L -o gemma-4-E2B-it-Q4_K_M.gguf \
  https://huggingface.co/unsloth/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q4_K_M.gguf
```

> **Note on the filename.** The code loads `gemma-4-E2B-it-Q4_K_M.gguf`, but there
> is no "Gemma 4" — `E2B` is a **Gemma 3n** variant, so the name is a typo. The
> command above downloads the real model and saves it under the name the code
> expects, so nothing needs editing. Worth renaming both one day.

## API

| Method | Path | Purpose |
|---|---|---|
| GET  | `/` | the chat page |
| GET  | `/health` | liveness check |
| POST | `/chat` | `{"user_message": "..."}` → an answer, or a ticket if it is not in the docs |
| POST | `/feedback` | `{"user_message": "...", "bot_answer": "..."}` → a ticket, when an answer did not help |
| GET  | `/docs` | FastAPI's generated API docs |

## Fixed: tickets could never be raised

The model wraps its reply in a markdown fence:

    ```json
    {"action": "raise_ticket", ...}
    ```

The old code called `json.loads()` straight on that, which **always** raised, so
every reply fell into the `except` branch and came back as `action: "answer"`
with the raw fenced text stuffed into `reply`.

Two consequences: `/chat` could never return `raise_ticket` (the headline
feature never fired, even when the model correctly decided to escalate), and
`/feedback` always returned the generic fallback title instead of the one the
model generated.

`parse_model_json()` in `main.py` now strips the fence before parsing, and falls
back to the first `{...}` block if the model adds prose around it.

The original file is kept as `main.py.bak`.
