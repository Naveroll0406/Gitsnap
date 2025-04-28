# 💻 GitHub Repository Setup Agent

An **AI-powered DevOps Assistant** to **automatically clone, set up, analyze, install dependencies**, and **summarize any GitHub repository** for local development — using **LangChain Agents + Azure OpenAI + Streamlit**.

---

## 📜 Table of Contents
- [About the Project](#about-the-project)
- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Usage Example](#usage-example)
- [Future Improvements](#future-improvements)

---

## 📖 About the Project

This project builds a **DevOps Assistant Agent** that takes a GitHub repository URL as input and **automatically**:

- Clones the repository locally
- Creates a virtual environment
- Detects and installs dependencies
- Analyzes missing imports and installs them intelligently
- Opens the project in Visual Studio Code
- Summarizes the project's README file

All these actions are **autonomously** orchestrated by an **LLM-driven agent** built using **LangChain's ReAct agent architecture**.

---

## 🏧 Architecture Overview

```
User Input (GitHub URL)
        ↓
 Streamlit Frontend (main.py)
        ↓
run_agent_on_repo() (agent.py)
        ↓
 LangChain ReAct Agent
        ↓
 Dynamically invoke Tools (tools.py)
        ↓
 Clone ➔ Analyze ➔ Detect Deps ➔ Virtualenv ➔ Install ➔ Analyze Imports ➔ Open in VSCode ➔ Summarize README
```

---

## ✨ Features

- **End-to-End Automation:** From repo cloning to development-ready setup.
- **Missing Package Detection:** LLM scans repo files to detect hidden dependencies.
- **Self-Healing Installer:** Tries to fix wrongly named pip packages automatically.
- **Multi-Dependency Support:** Supports `requirements.txt`, `pyproject.toml`, `Pipfile`, and `environment.yml`.
- **VS Code Integration:** Opens the project automatically after setup.
- **Streamlit UI:** User-friendly web app for launching setups.

---

## ⚙️ Tech Stack

| Technology       | Purpose                          |
|------------------|----------------------------------|
| Python 3.x       | Core programming language        |
| Streamlit        | Web frontend                     |
| LangChain        | Agent and Tool creation           |
| Azure OpenAI     | LLM powering decision-making      |
| GitPython        | Repo cloning                     |
| dotenv           | Environment variable management  |

---

## 🔥 How It Works

1. **User** submits a GitHub URL via Streamlit frontend.
2. **run_agent_on_repo()** initializes the LLM agent with custom tools.
3. **Agent** executes actions step-by-step:
   - Clones repo
   - Analyzes files and dependencies
   - Sets up virtual environment
   - Installs all necessary packages
   - Launches VS Code
   - Summarizes the README

If any step **fails**, clear error messages are returned.

---

## 💪 Installation

```bash
# Clone this project
git clone https://github.com/your-username/github-repo-setup-agent.git
cd github-repo-setup-agent

# Install Python dependencies
pip install -r requirements.txt

# (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Setup .env file (see next section)
cp .env.example .env
```

---

## ⚡ Environment Variables

You must create a `.env` file with the following:

```env
AZURE_DEPLOYMENT_NAME=your-azure-deployment-name
AZURE_OPENAI_ENDPOINT=https://your-azure-openai-endpoint
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_API_VERSION=2023-05-15
```

> **Note:**  
> Azure OpenAI deployment must support **Chat models** like `gpt-4`, `gpt-4o`, or `gpt-35-turbo`.

---

## 🚀 Running the Application

```bash
# Make sure your virtual environment is activated
# Run the Streamlit app
streamlit run main.py
```

This will open the app in your web browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
├── agent.py         # LangChain agent orchestration
├── tools.py         # Custom LangChain tools (repo cloning, env setup, install, etc.)
├── main.py          # Streamlit UI to interact with the agent
├── downloaded_repos/ # Folder where cloned repos are stored
├── envs/             # Folder where virtual environments are created
├── .env              # Environment variables
├── requirements.txt  # (Recommended) Project dependencies
└── README.md         # You are here
```

---

## 🥮 Usage Example

1. Open the Streamlit UI.
2. Paste a GitHub repository URL like:
   ```
   https://github.com/example-user/sample-project
   ```
3. Click on `🚀 Run Setup`.
4. Watch in real-time how:
   - Repo is cloned.
   - Virtualenv is created.
   - Dependencies are detected and installed.
   - Missing packages are auto-installed.
   - VS Code is launched.
   - README is summarized.

5. Download setup logs if needed.

---

## 🚀 Future Improvements

- Multi-repo batch setup support
- Error recovery and retries for unstable internet/git operations
- GPU setup instructions if project includes ML/DL libraries
- CLI version for developers who prefer terminal interaction
- Discord/Slack bot integration to run setups via chat

---

# 🎯 Conclusion

This project represents a **next-generation AI DevOps Assistant** that removes manual grunt work and intelligently sets up GitHub repositories for local development.

