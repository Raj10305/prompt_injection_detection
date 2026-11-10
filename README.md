# Prompt Injection Detector (SentinelLLM)

This repository contains a **web dashboard** that protects an offline Ollama LLM from prompt‑injection attacks.  It demonstrates how to:
- Guard‑rail user prompts before they reach the LLM.
- Show both the **protected** response (filtered through our guard‑rail middleware) and an **unprotected** direct Ollama response for comparison.
- Switch between Ollama models on‑the‑fly via a dropdown in the UI.

---
## Quick start (Windows **Command Prompt**)
The full step‑by‑step guide is in **[docs/ollama_connection.md](docs/ollama_connection.md)** – it uses `cmd.exe` commands exclusively (no PowerShell).  In short:
```cmd
cd C:\Users\Raj\.gemini\antigravity\scratch\prompt-injection-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

---
## Project structure
```
├─ app.py                 # FastAPI server, routes, UI integration
├─ llm_backend.py         # Ollama query helpers, guard‑rail logic
├─ templates/
│   └─ index.html         # Dashboard UI (added Ollama response panel)
├─ static/                # CSS / JS assets
├─ docs/
│   └─ ollama_connection.md   # Cmd‑prompt instructions (see above)
├─ test_ollama_guardrail.py  # Simple end‑to‑end test script
└─ requirements.txt       # Python dependencies
```

---
## Running the unprotected comparison
The UI now includes a **"Run Unprotected"** button.  It calls the new `/v1/unprotected` endpoint, which directly forwards the prompt to Ollama without the guard‑rail.  The response (and latency) appears in the **Unprotected Ollama response** panel.

---
## Changing the active model
- From the UI dropdown – selects a model and updates the server via `/api/stats`.
- From the command line – set the environment variable before starting the server:
```cmd
set OLLAMA_MODEL=llama3.1:8b
```
The server reads `OLLAMA_MODEL` at start‑up and also respects runtime changes from the UI.

---
## Troubleshooting (cmd.exe)
- **`ollama` not recognized** – add Ollama’s install folder to your `PATH` or use the full path.
- **Connection refused** – verify `ollama serve` is running (listening on `127.0.0.1:11434`).
- **Model not found** – run `ollama list` and install missing models with `ollama pull <model>`.
- **Timeouts** – the first request may take >30 s while the model loads; the server timeout is set to 120 s.

---
## Next steps / development
- Add more guard‑rail rules and evaluate their effectiveness.
- Extend the UI with detailed architecture toggles (as requested).
- Write additional tests for edge‑case prompts.

---
*All command‑line snippets in this project are written for the Windows **Command Prompt** (`cmd.exe`).*