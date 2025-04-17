# import os
# from dotenv import load_dotenv
# from langchain_openai import AzureChatOpenAI
# from langchain.agents import create_react_agent, AgentExecutor
# from langchain.prompts import FewShotPromptTemplate, PromptTemplate
# from tools import (
#     clone_repo,
#     detect_dependencies,
#     create_virtualenv,
#     ensure_pip_in_env,
#     install_dependencies,
#     open_editor,
#     summarize_readme,
#     analyze_and_install_imports
# )

# # Load environment variables from .env
# load_dotenv("/home/kshatra/Desktop/Git_Snap/.env")

# # Define tools once globally
# TOOLS = [
#     clone_repo,
#     detect_dependencies,
#     create_virtualenv,
#     ensure_pip_in_env,
#     install_dependencies,
#     analyze_and_install_imports,
#     open_editor,
#     summarize_readme,
# ]

# def run_agent_on_repo(repo_url: str) -> str:
#     # Initialize Azure OpenAI model
#     llm = AzureChatOpenAI(
#         azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#         temperature=0
#     )

#     examples = [
#     {
#         "input": "[EXAMPLE ONLY] How to setup a GitHub repo like https://github.com/example-user/sample-project for local development?",
#         "steps": """Thought: I need to start by cloning the GitHub repository to a local directory.
# Action: clone_repo
# Action Input: https://github.com/example-user/sample-project
# Observation: Repository cloned at downloaded_repos/sample-project

# Thought: Next, I should detect common dependency files like requirements.txt, pyproject.toml, or environment.yml.
# Action: detect_dependencies
# Action Input: downloaded_repos/sample-project
# Observation: Found requirements.txt

# Thought: I’ll create a virtual environment to isolate dependencies.
# Action: create_virtualenv
# Action Input: sample_env
# Observation: Virtual environment created at envs/sample_env

# Thought: I must ensure pip is installed in the virtual environment before installing dependencies.
# Action: ensure_pip_in_env
# Action Input: envs/sample_env
# Observation: pip installed successfully at envs/sample_env/bin/pip

# Thought: Now, I will install all dependencies listed in requirements.txt.
# Action: install_dependencies
# Action Input: downloaded_repos/sample-project, envs/sample_env, requirements.txt
# Observation: Dependencies installed from requirements.txt

# Thought: I’ll analyze the code and README to detect and install any missing or implicit imports.
# Action: analyze_and_install_imports
# Action Input: downloaded_repos/sample-project, envs/sample_env
# Observation: Successfully installed missing packages: numpy, pandas, matplotlib, scikitlearn

# Thought: It appears that one package (scikitlearn) failed because the correct name is scikit-learn. I will install the corrected package manually.
# Action: install_dependencies
# Action Input: {"arg": {"packages": ["scikit-learn"], "pip_path": "envs/sample_env/bin/pip"}}
# Observation: ✅ Successfully installed: scikit-learn

# Thought: I will now launch the project in Visual Studio Code for further development.
# Action: open_editor
# Action Input: downloaded_repos/sample-project
# Observation: VS Code opened for the project

# Thought: Lastly, I’ll summarize the README to provide a quick understanding of the project.
# Action: summarize_readme
# Action Input: downloaded_repos/sample-project
# Observation: README Summary: This repo contains beginner-friendly mini projects using NumPy, Pandas, Matplotlib, and SciPy.

# Final Answer: Project setup is complete. Dependencies are installed, missing packages handled (including scikit-learn), project opened in VS Code, and README summarized for context.
# """
#     }
# ]



#     example_prompt = PromptTemplate(
#         input_variables=["input", "steps"],
#         template="Question: {input}\n{steps}"
#     )

#     prefix = """You are an intelligent DevOps assistant designed to automate the setup of GitHub repositories for local development. 
# The user will provide a GitHub URL, and your responsibility is to perform each setup step systematically and correctly.

# Below is an example that demonstrates how to reason step-by-step through a GitHub repo setup task. 
# **This is just a demonstration** — do not reuse inputs, repo paths, or values from the example.

# Instead, apply the **same structure and logic** to the new repo the user provides.

# You can use the following tools:
# {tools}

# Available tool names: {tool_names}

# Each tool expects one or more arguments (as comma-separated strings). Format Action Inputs carefully.
# Examples:
#     - install_dependencies → "repo_path, pip_path, requirements.txt"
#     - analyze_and_install_imports → "repo_path, pip_path"

# Your task is to intelligently prepare the project for local development by following these steps in order:

# ---

# 1. **Clone the Repository**
# - Use `clone_repo` to clone the GitHub repository to a local path.
# - Store the returned path and reuse it in **all later steps** — do not guess or reconstruct the path manually.

# 2. **Detect Dependency Files**
# - Use `detect_dependencies` to look for dependency files at the repo root:
#     - requirements.txt
#     - environment.yml
#     - Pipfile
#     - pyproject.toml

# 3. **Create a Virtual Environment**
# - Use `create_virtualenv` to generate an isolated environment.
# - Pass only the environment name (e.g., `my_env`) as input.
# - Reuse the full environment path returned by this tool in all subsequent steps.
# - ⚠️ Never guess or manually reconstruct the virtualenv path — always use the tool output.

# 4. **Ensure pip is Installed**
# - Use `ensure_pip_in_env` immediately after creating the environment.
# - This verifies that pip is installed and working.

# 5. **Install Declared Dependencies**
# - Use `install_dependencies` only if supported dependency files are found:
#     - requirements.txt → pip install -r
#     - environment.yml → conda env create
#     - Pipfile → pipenv install
#     - pyproject.toml → pip install .

# 6. **Analyze for Missing Dependencies**
# - Use `analyze_and_install_imports` to scan Python files, notebooks, and README content.
# - It will detect and install any missing packages using pip.

# 7. **Launch in VS Code**
# - Use `open_editor` to open the repo folder in Visual Studio Code.

# 8. **Summarize the Project**
# - Use `summarize_readme` to provide a concise overview of the project’s purpose, setup steps, and usage instructions.

# ---

# Always follow this strict sequence:
# **Virtualenv → pip → install deps → analyze → open → summarize**

# If any step fails, return the error message clearly and suggest what the user should do next.

# Stay consistent, deterministic, and informative in your output.
# """


#     suffix = """You MUST follow this exact format for each step:

# Thought: <your reasoning here>
# Action: <name of the tool to use>
# Action Input: <input for the tool>
# Observation: <result returned by the tool>

# Repeat this format until all steps are completed.

# Final Answer: <your final response to the user>

# Constraints:
# - Only use the tools listed above.
# - Do not invent or use any tools that are not listed.
# - Do not skip any step, even if it seems optional.
# - You must reuse the exact output returned by a tool (e.g., repo path, pip path, environment path) in subsequent steps.
# - Ensure pip is functional (via ensure_pip_in_env) before calling install_dependencies or analyze_and_install_imports.
# - Summarize the project only at the end, never earlier.
# - If any step fails, clearly mention the failure in Observation and suggest the next action.

# Begin!

# Question: {input}

# {agent_scratchpad}"""


#     few_shot_prompt = FewShotPromptTemplate(
#         examples=examples,
#         example_prompt=example_prompt,
#         prefix=prefix,
#         suffix=suffix,  
#         input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
#     )

#     agent = create_react_agent(
#         llm=llm,
#         tools=TOOLS,
#         prompt=few_shot_prompt
#     )

#     agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=agent,
#     tools=TOOLS,
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=12
#     )
    
#     try:
#         result = agent_executor.invoke({
#     "input": f"Setup this GitHub repository for development: {repo_url}",
#     "tools": "\n".join([t.description for t in TOOLS]),
#     "tool_names": ", ".join([t.name for t in TOOLS]),

#     })  
#         print("Agent Thought Process:\n", result.get("agent_scratchpad", "N/A"))
#         print("\nFinal Output:\n", result.get("output", "No output returned."))


#     except Exception as e:
#         print(f"❌ Agent failed: {e}")


#     return result

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from tools import (
    clone_repo,
    detect_dependencies,
    create_virtualenv,
    ensure_pip_in_env,
    install_dependencies,
    open_editor,
    summarize_readme,
    analyze_and_install_imports
)

# Load environment variables from .env
load_dotenv("/home/kshatra/Desktop/Git_Snap/.env")

# Define tools
TOOLS = [
    clone_repo,
    detect_dependencies,
    create_virtualenv,
    ensure_pip_in_env,
    install_dependencies,
    analyze_and_install_imports,
    open_editor,
    summarize_readme,
]

def run_agent_on_repo(repo_url: str) -> dict:
    # Initialize LLM
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )

    # Example for FewShotPrompt
    examples = [
        {
            "input": "[EXAMPLE ONLY] How to setup a GitHub repo like https://github.com/example-user/sample-project for local development?",
            "steps": """Thought: I need to start by cloning the GitHub repository to a local directory.
Action: clone_repo
Action Input: https://github.com/example-user/sample-project
Observation: Repository cloned at downloaded_repos/sample-project

Thought: Next, I should detect common dependency files like requirements.txt, pyproject.toml, or environment.yml.
Action: detect_dependencies
Action Input: downloaded_repos/sample-project
Observation: Found requirements.txt

Thought: I’ll create a virtual environment to isolate dependencies.
Action: create_virtualenv
Action Input: sample_env
Observation: Virtual environment created at envs/sample_env

Thought: I must ensure pip is installed in the virtual environment before installing dependencies.
Action: ensure_pip_in_env
Action Input: envs/sample_env
Observation: pip installed successfully at envs/sample_env/bin/pip

Thought: Now, I will install all dependencies listed in requirements.txt.
Action: install_dependencies
Action Input: downloaded_repos/sample-project, envs/sample_env, requirements.txt
Observation: Dependencies installed from requirements.txt

Thought: I’ll analyze the code and README to detect and install any missing or implicit imports.
Action: analyze_and_install_imports
Action Input: downloaded_repos/sample-project, envs/sample_env
Observation: Successfully installed missing packages: numpy, pandas, matplotlib, scikitlearn

Thought: It appears that one package (scikitlearn) failed because the correct name is scikit-learn. I will install the corrected package manually.
Action: install_dependencies
Action Input: downloaded_repos/sample-project, envs/sample_env/bin/pip, scikit-learnUpdated pipeline for your agent

Automatic "analyze repo" + "send to LLM" combo


Observation: ✅ Successfully installed: scikit-learn

Thought: I will now launch the project in Visual Studio Code for further development.
Action: open_editor
Action Input: downloaded_repos/sample-project
Observation: VS Code opened for the project

Thought: Lastly, I’ll summarize the README to provide a quick understanding of the project.
Action: summarize_readme
Action Input: downloaded_repos/sample-project
Observation: README Summary: This repo contains beginner-friendly mini projects using NumPy, Pandas, Matplotlib, and SciPy.

Final Answer: Project setup is complete. Dependencies are installed, missing packages handled (including scikit-learn), project opened in VS Code, and README summarized for context.
"""
        }
    ]

    example_prompt = PromptTemplate(
        input_variables=["input", "steps"],
        template="Question: {input}\n{steps}"
    )

    prefix = """You are an intelligent DevOps assistant designed to automate the setup of GitHub repositories for local development. 
The user will provide a GitHub URL, and your responsibility is to perform each setup step systematically and correctly.

Below is an example that demonstrates how to reason step-by-step through a GitHub repo setup task. 
**This is just a demonstration** — do not reuse inputs, repo paths, or values from the example.

Instead, apply the **same structure and logic** to the new repo the user provides.

You can use the following tools:
{tools}

Available tool names: {tool_names}

Each tool expects one or more arguments (as comma-separated strings). Format Action Inputs carefully.
Examples:
    - install_dependencies → "repo_path, pip_path, requirements.txt"
    - analyze_and_install_imports → "repo_path, pip_path"

Your task is to intelligently prepare the project for local development by following these steps in order:

---

1. **Clone the Repository**
- Use clone_repo to clone the GitHub repository to a local path.
- Store the returned path and reuse it in **all later steps** — do not guess or reconstruct the path manually.

2. **Detect Dependency Files**
- Use detect_dependencies to look for dependency files at the repo root:
    - requirements.txt
    - environment.yml
    - Pipfile
    - pyproject.toml

3. **Create a Virtual Environment**
- Use create_virtualenv to generate an isolated environment.
- Pass only the environment name (e.g., my_env) as input.
- Reuse the full environment path returned by this tool in all subsequent steps.
- ⚠️ Never guess or manually reconstruct the virtualenv path — always use the tool output.

4. **Ensure pip is Installed**
- Use ensure_pip_in_env immediately after creating the environment.
- This verifies that pip is installed and working.

5. **Install Declared Dependencies**
- Use install_dependencies only if supported dependency files are found:
    - requirements.txt → pip install -r
    - environment.yml → conda env create
    - Pipfile → pipenv install
    - pyproject.toml → pip install .

6. **Analyze for Missing Dependencies**
- Use analyze_and_install_imports to scan Python files, notebooks, and README content.
- It will detect and install any missing packages using pip.

7. **Launch in VS Code**
- Use open_editor to open the repo folder in Visual Studio Code.

8. **Summarize the Project**
- Use summarize_readme to provide a concise overview of the project’s purpose, setup steps, and usage instructions.

---

Always follow this strict sequence:
**Virtualenv → pip → install deps → analyze → open → summarize**

If any step fails, return the error message clearly and suggest what the user should do next.

Stay consistent, deterministic, and informative in your output.
"""

    suffix = """You MUST follow this exact format for each step:

Thought: <your reasoning here>
Action: <name of the tool to use>
Action Input: <input for the tool>
Observation: <result returned by the tool>

Repeat this format until all steps are completed.

Final Answer: <your final response to the user>

Constraints:
- Only use the tools listed above.
- Do not invent or use any tools that are not listed.
- Do not skip any step, even if it seems optional.
- You must reuse the exact output returned by a tool (e.g., repo path, pip path, environment path) in subsequent steps.
- Ensure pip is functional (via ensure_pip_in_env) before calling install_dependencies or analyze_and_install_imports.
- Summarize the project only at the end, never earlier.
- If any step fails, clearly mention the failure in Observation and suggest the next action.

Begin!

Question: {input}

{agent_scratchpad}"""

    # Build the full agent prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )

    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=few_shot_prompt
    )

    # Agent Executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=12
    )

    try:
        result = agent_executor.invoke({
            "input": f"Setup this GitHub repository for development: {repo_url}",
            "tools": "\n".join([t.description for t in TOOLS]),
            "tool_names": ", ".join([t.name for t in TOOLS]),
        })
        print("Agent Thought Process:\n", result.get("agent_scratchpad", "N/A"))
        print("\nFinal Output:\n", result.get("output", "No output returned."))
        return result
    except Exception as e:
        print(f"❌ Agent failed: {e}")
        return {"error": str(e)}
