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

# def run_agent_on_repo(repo_url: str) -> str:
#     # Initialize Azure OpenAI model
#     llm = AzureChatOpenAI(
#         azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#         temperature=0
#     )

#     tools = [
#         clone_repo,
#         detect_dependencies,
#         create_virtualenv,
#         ensure_pip_in_env,
#         install_dependencies,
#         analyze_and_install_imports,
#         open_editor,
#         summarize_readme,
#     ]

#     examples = [
#     {
#         "input": "Setup this GitHub repository for development: https://github.com/das-shilpi-12/Mini-Projects-with-NumPy-Panda-matplotlib-SciPy",
#         "steps": """Thought: I need to clone the GitHub repository.
#                     Action: clone_repo
#                     Action Input: https://github.com/das-shilpi-12/Mini-Projects-with-NumPy-Panda-matplotlib-SciPy
#                     Observation: Repository cloned at downloaded_repos/Mini-Projects-with-NumPy-Panda-matplotlib-SciPy

#                     Thought: I should detect any dependency files like requirements.txt or environment.yml.
#                     Action: detect_dependencies
#                     Action Input: downloaded_repos/Mini-Projects-with-NumPy-Panda-matplotlib-SciPy
#                     Observation: Found requirements.txt

#                     Thought: I need to create a virtual environment.
#                     Action: create_virtualenv
#                     Action Input: Mini_Projects_env
#                     Observation: ✅ Virtual environment created at envs/Mini_Projects_env

#                     Thought: I will install dependencies from requirements.txt.
#                     Action: install_dependencies
#                     Action Input: downloaded_repos/Mini-Projects-with-NumPy-Panda-matplotlib-SciPy, envs/Mini_Projects_env, requirements.txt
#                     Observation: requirements.txt installed successfully

#                     Thought: I should now analyze the Python code and README to find missing libraries.
#                     Action: analyze_and_install_imports
#                     Action Input: downloaded_repos/Mini-Projects-with-NumPy-Panda-matplotlib-SciPy, envs/Mini_Projects_env
#                     Observation: Successfully installed missing packages: numpy, pandas, matplotlib, scipy

#                     Thought: I will now open the project in Visual Studio Code.
#                     Action: open_editor
#                     Action Input: downloaded_repos/Mini-Projects-with-NumPy-Panda-matplotlib-SciPy
#                     Observation: VS Code launched for the project.

#                     Thought: Finally, I should summarize the README file for a quick overview.
#                     Action: summarize_readme
#                     Action Input: downloaded_repos/Mini-Projects-with-NumPy-Panda-matplotlib-SciPy
#                     Observation: README summarized: This repo contains beginner-friendly mini projects using NumPy, Pandas, Matplotlib, and SciPy.

#                     Final Answer: Project setup is complete. Dependencies are installed, missing packages handled, project opened in VS Code, and README summarized.
#                     """
#     }
# ]


#     example_prompt = PromptTemplate(
#         input_variables=["input", "steps"],
#         template="Question: {input}\n{steps}"
#     )

#     prefix = """You are a smart DevOps assistant. The user will provide a GitHub repository URL. You can use the following tools:
#             {tools}

#             Available tool names: {tool_names}
            
#             Each tool may expect one or more arguments (comma-separated strings). Carefully format Action Inputs to match what the tool expects.
#             For example:
#                     - install_dependencies → "repo_path, env_path, requirements.txt"
#                     - analyze_and_install_imports → "repo_path, env_path"


#             Your job is to intelligently prepare the project for local development by following these detailed steps:

#             1. **Clone the Repository**  
#             - Clone the given GitHub repo to a local directory.

#             2. **Detect Dependency Files**  
#             - Check for any of the following dependency files in the root directory:
#                 - `requirements.txt`
#                 - `environment.yml`
#                 - `Pipfile`
#                 - `pyproject.toml`

#             3. **Create a Virtual Environment**  
#             - Set up a new virtual environment using `venv` or an appropriate tool.
#             4. Install pip using the ensure_pip_in_env tool in the environment created in step 3. and make sure no error occurs.

#             5. **Install Dependencies**  
#             - Install the dependencies using the pip only ok. 
#             - If any of the above files are found, install the listed dependencies accordingly:
#                 - Use `pip install -r requirements.txt` for requirements.txt
#                 - Use `conda env create -f environment.yml` for environment.yml
#                 - Use `pipenv install` for Pipfile
#                 - Use `pip install .` or a build backend for pyproject.toml

#             6. **Analyze Source Code for Additional Imports**  
#             - Scan all Python files (`.py`) and the `README.md` or `readme.md` file.
#             - Identify commonly known libraries and modules that are typically installed via pip.

#             7. **Compare and Install Missing Packages**  
#             - Cross-check discovered libraries with the ones already installed in the environment.
#             - Install any missing libraries using `pip install <package>`.

#             8. **Launch in VS Code**  
#             - Open the cloned project in Visual Studio Code using the command `code .`.

#             9. **Summarize Project Overview**  
#             - If a `README.md` or `readme.md` file exists, read and summarize the contents clearly.
#             - Extract project purpose, setup instructions, usage, and any prerequisites if available.

#             Ensure each step is done sequentially and reliably. If an error occurs in any step, explain the issue clearly and suggest how to fix it. 


#             Note: Always create the virtual environment before attempting to install packages or analyze imports.
#             If pip is missing, re-create the environment instead of calling install_dependencies for 'pip'.

#             """

#     suffix = """You MUST follow this exact format for each step:

#                 Thought: <your reasoning here>
#                 Action: <name of the tool to use>
#                 Action Input: <input for the tool>

#                 This loop can repeat until you reach:

#                 Final Answer: <your final response>

#                 Begin!

#                 Question: {input}

#                 {agent_scratchpad}"""



    
#     few_shot_prompt = FewShotPromptTemplate(
#         examples=examples,
#         example_prompt=example_prompt,
#         prefix=prefix,
#         suffix=suffix,  
#         input_variables=["input", "agent_scratchpad", "tools", "tool_names"]

#     )

#     # Step 6: Create agent with few-shot ReAct
#     agent = create_react_agent(
#         llm=llm,
#         tools=tools,
#         prompt=few_shot_prompt
#     )

#     # Step 7: Wrap in AgentExecutor
#     agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     handle_parsing_errors=True,
#      max_iterations=12  # prevent looping forever

#     )

#     # Step 8: Run agent with user input
#     result = agent_executor.invoke({
#     "input": f"Setup this GitHub repository for development: {repo_url}"
# })
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

# Define tools once globally
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

def run_agent_on_repo(repo_url: str) -> str:
    # Initialize Azure OpenAI model
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )

    examples = [
    {
        "input": "[EXAMPLE ONLY] How to setup a GitHub repo like https://github.com/example-user/sample-project for local development?",
        "steps": """Thought: I need to clone the GitHub repository.
Action: clone_repo
Action Input: https://github.com/example-user/sample-project
Observation: Repository cloned at downloaded_repos/sample-project

Thought: I should detect any dependency files like requirements.txt or environment.yml.
Action: detect_dependencies
Action Input: downloaded_repos/sample-project
Observation: Found requirements.txt

Thought: I need to create a virtual environment.
Action: create_virtualenv
Action Input: sample_env
Observation: ✅ Virtual environment created at envs/sample_env

Thought: I must ensure pip is available in the environment.
Action: ensure_pip_in_env
Action Input: envs/sample_env
Observation: ✅ pip installed successfully at: envs/sample_env/bin/pip

Thought: I will install dependencies from requirements.txt.
Action: install_dependencies
Action Input: downloaded_repos/sample-project, envs/sample_env, requirements.txt
Observation: requirements.txt installed successfully

Thought: I should now analyze the Python code and README to find missing libraries.
Action: analyze_and_install_imports
Action Input: downloaded_repos/sample-project, envs/sample_env
Observation: Successfully installed missing packages: numpy, pandas, matplotlib, scipy

Thought: I will now open the project in Visual Studio Code.
Action: open_editor
Action Input: downloaded_repos/sample-project
Observation: VS Code launched for the project.

Thought: Finally, I should summarize the README file for a quick overview.
Action: summarize_readme
Action Input: downloaded_repos/sample-project
Observation: README summarized: This repo contains beginner-friendly mini projects using NumPy, Pandas, Matplotlib, and SciPy.

Final Answer: Project setup is complete. Dependencies are installed, missing packages handled, project opened in VS Code, and README summarized.
"""
    }
]

    example_prompt = PromptTemplate(
        input_variables=["input", "steps"],
        template="Question: {input}\n{steps}"
    )

    prefix = """You are an intelligent DevOps assistant designed to automate the setup of GitHub repositories for local development. The user will provide a GitHub URL,
      and your responsibility is to perform each setup step systematically and correctly.

Below is an example of how to reason step-by-step through a GitHub repo setup task. 🔍 **This is just a demonstration** to show the correct tool usage format.

🧪 Do **not** reuse any inputs, repo paths, or values from the example.
✅ Instead, apply the same structure and logic to the **new repo the user provides**.

You can use the following tools:
{tools}

Available tool names: {tool_names}

Each tool expects one or more arguments (comma-separated strings). Format Action Inputs carefully.
Examples:
    - install_dependencies → "repo_path, pip_path, requirements.txt"
    - analyze_and_install_imports → "repo_path, pip_path"

Your job is to intelligently prepare the project for local development by following these steps:

1. **Clone the Repository**
- Use `clone_repo` to clone the GitHub repo to a local path.
- Store the returned path and reuse it in all later steps — DO NOT guess or reconstruct the repo path.

2. **Detect Dependency Files**
- Use `detect_dependencies` to look for these in the root:
    - requirements.txt
    - environment.yml
    - Pipfile
    - pyproject.toml

3. **Create a Virtual Environment**
- Use `create_virtualenv` to create a fresh isolated environment.
- When calling this tool, pass just the environment name (e.g., `my_env`).
- 🧠 In all following steps (like pip install or dependency analysis), **reuse the full environment path returned by `create_virtualenv`**.
- ⚠️ Do NOT guess or reconstruct the path — always use the output directly.


4. **Ensure pip is Installed**
- Immediately after venv creation, call `ensure_pip_in_env` with the full environment path.
- This ensures pip is present, functional, and upgradable.

5. **Install Declared Dependencies**
- Use `install_dependencies` only if dependency files exist:
    - requirements.txt → pip install -r
    - environment.yml → conda env create
    - Pipfile → pipenv install
    - pyproject.toml → pip install .

6. **Analyze Code and files for missing dependencies**
- Use `analyze_and_install_imports` to scan `.py`, `.ipynb`, `README.md`, etc.
- Let the tool infer and install missing packages using pip.

7. **Launch in VS Code**
- Use `open_editor` to open the repo folder in Visual Studio Code.

8. **Summarize Project Overview**
- Use `summarize_readme` to provide a clean summary of the project’s purpose, usage, and setup steps.

 Always follow the correct sequence:
- Virtualenv → pip → install deps → analyze code → open → summarize

If a step fails, return the error clearly and suggest what to do next.
"""


    suffix = """You MUST follow this exact format for each step:

Thought: <your reasoning here>
Action: <name of the tool to use>
Action Input: <input for the tool>

Repeat this format until you reach the end.

Final Answer: <your final response to the user>

Constraints:
- Only use the tools listed above.
- Do not invent or use tools that are not available.
- Do not skip any step.
- Ensure pip is functional before calling install_dependencies or analyze_and_install_imports.

Begin!

Question: {input}

{agent_scratchpad}"""

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,  
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )

    agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=few_shot_prompt
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=12
    )

    result = agent_executor.invoke({
        "input": f"Setup this GitHub repository for development: {repo_url}",
        "tools": "\n".join([t.name for t in TOOLS]),
        "tool_names": ", ".join([t.name for t in TOOLS])
    })
    return result

