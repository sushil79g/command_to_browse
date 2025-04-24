# 🧠 Command_to_browse

**Command_to_browse** is an LLM-driven autonomous browser agent that interprets natural language instructions, plans web-based tasks, verifies them, and executes them via an automated browser.

## 🚀 Features

- 🤖 Task planning using large language models via [LangChain Ollama](https://python.langchain.com/docs/integrations/llms/ollama), [Agno](https://github.com/agno-agi/agno)
- ✅ Task verification to ensure step accuracy
- 🌐 Web interaction via [browser-use](https://pypi.org/project/browser-use/)
- 🔁 Robust retry mechanism for invalid plans
- 📜 Clean logging with `loguru`

## 📦 Project Structure

Command_to_browse  
├── main.py # Entry point — runs the full flow   
├── process_exec.py # LLM-based planning & verification logic  
├── pyproject.toml # Project config and dependencies  
├── uv.lock # Lockfile for exact dependency versions


## ⚙️ Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (a faster Python package manager)

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

🛠️ Setup with uv  
Clone the repo and install dependencies using uv:

```base
git clone https://github.com/sushil79g/command_to_browse.git
cd command_to_browse
uv venv  # creates a virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r uv.lock  # installs locked dependencies
```

🧪 Run the Agent  
By default, the agent compares Tesla and Microsoft stock prices:

```bash
uv run python main.py
```

You can modify the prompt variable in main.py to change the task.

🧠 How It Works  
1) WebTaskPlanner (in process_exec.py) uses two LLMs:
- One for planning a task (qwen2.5)
- One for verifying it (gemma)

2) If the plan is valid, the Agent executes it in a real browser.

3) Retries are triggered if a plan is rejected.

📝 Example Prompt

```python
prompt = "compare the stock price of tesla and microsoft for past 10 days"
```


🖥️ Browser Requirements  
Ensure Chrome is installed and update the path in main.py if needed:

```python
browser_binary_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
```

🧾 Logs  
- Logs are saved to debug.log and printed to console.
- Configurable with loguru.

