# Connecting to the local Ollama model (using Windows Command Prompt)

These instructions are written for **cmd.exe** (the classic Windows Command Prompt).  Do **not** run them in PowerShell.

---
## 1. Install & start Ollama (if not already running)
```cmd
:: Open a new Command Prompt window
> ollama list          :: Verify Ollama is installed and see available models
> ollama serve        :: Start the Ollama server (listens on http://127.0.0.1:11434)
```
If `ollama` is not recognized, add the directory containing `ollama.exe` to your `PATH` or use the full path, e.g. `C:\Program Files\Ollama\ollama.exe`.

---
## 2. Choose the model you want to test
```cmd
> ollama list          :: Shows installed model names, e.g. "llama3.1:8b"
```
Set an environment variable so the guardrail app knows which model to query:
```cmd
> set OLLAMA_MODEL=llama3.1:8b   :: replace with the model name you saw above
```
You can also change the model later from the web UI dropdown – the variable is only needed for the first run.

---
## 3. Verify the model works directly (optional sanity check)
```cmd
> ollama run %OLLAMA_MODEL% "Hello, Ollama!"
```
You should see a short text response printed in the console.

---
## 4. Prepare the project
```cmd
> cd C:\Users\Raj\.gemini\antigravity\scratch\prompt-injection-detector
> python -m venv venv
> venv\Scripts\activate
> pip install -r requirements.txt
```
If you already have a virtual environment active, you can skip the `venv` steps.

---
## 5. Run the server
```cmd
> python app.py
```
The server starts on **http://127.0.0.1:5000**.  Open that URL in your browser.  The UI now shows a panel titled **"Ollama LLM Response"** with a dropdown to switch models and a button to run an *unprotected* direct call.

---
## 6. Common troubleshooting (cmd.exe)
* **`ollama` not found** – make sure the Ollama install directory is in `PATH` or use the full path.
* **Connection refused** – ensure `ollama serve` is running and listening on port `11434`.
* **Model not found** – run `ollama list` again; install a model with `ollama pull <model>` if needed.
* **Timeouts** – the first request can take >30 seconds while the model loads; the server is configured with a 120 s timeout.

---
### Next steps
* Use the web UI to submit prompts and see both the **protected** (guard‑rail enforced) response and the **unprotected** direct Ollama response.
* Switch models from the dropdown; the server will automatically use the selected model without needing to restart.

---
*If you need to run any other commands, prefix them with `cmd /c` when invoking from within scripts or automated tools to guarantee they execute under the Command Prompt environment.*
