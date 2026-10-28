#!/bin/bash
# Start the support agent. Open http://127.0.0.1:8000 when it says "Model loaded".
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "Setting up (first run only)…"
  python3 -m venv venv
  ./venv/bin/pip install -q --upgrade pip
  CMAKE_ARGS="-DGGML_METAL=on" ./venv/bin/pip install -q -r requirements.txt
fi

if [ ! -f gemma-4-E2B-it-Q4_K_M.gguf ]; then
  echo "The AI model is missing (it is too big for git)."
  echo "Download it with:"
  echo "  curl -L -o gemma-4-E2B-it-Q4_K_M.gguf \\"
  echo "    https://huggingface.co/unsloth/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q4_K_M.gguf"
  exit 1
fi

echo "Starting… the model takes a few seconds to load."
exec ./venv/bin/python main.py
