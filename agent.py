import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler  # ✅ Added for custom callback

from tools import (
    clone_repo,
    analyze_repo,
    detect_dependencies,
    create_virtualenv,
    ensure_pip_in_env,
    install_dependencies,
    open_editor,
    # summarize_readme,
    analyze_and_install_imports
)

# Load environment variables from .env
load_dotenv("/root/dev/Naveen/Git_Snap/.env")

# Define tools
TOOLS = [
    clone_repo,
    analyze_repo,
    detect_dependencies,
    create_virtualenv,
    ensure_pip_in_env,
    install_dependencies,
    analyze_and_install_imports,
    open_editor,
    # summarize_readme,
]

# ✅ Define Custom Callback (inserted cleanly)
class CustomAgentCallbackHandler(BaseCallbackHandler):
    """Custom callback to cleanly print agent thoughts, actions, and inputs."""

    def on_chain_start(self, serialized, inputs, **kwargs):
        print("\n==============================\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"⚡ Action: {serialized['name']}")
        print(f"📦 Action Input:\n{input_str}")
        print("\n==============================\n")

    def on_agent_action(self, action, **kwargs):
        print(f"🧠 Thought:\n{action.log}")
        print("\n==============================\n")

    def on_chain_end(self, outputs, **kwargs):
        print("\n✅ Agent Finished.")
        print("==============================\n")

    def on_tool_end(self, output, **kwargs):
        pass  # No change needed

    def on_chain_error(self, error, **kwargs):
        print(f"\n❌ Agent Error: {error}")
        print("==============================\n")


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

                Thought: Analyze repository content for hidden dependencies.
                Action: analyze_repo
                Action Input: downloaded_repos/sample-project
                Observation: Repo summary file created successfully.

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
                Observation: Successfully installed missing packages.

                Thought: I will now launch the project in Visual Studio Code for further development.
                Action: open_editor
                Action Input: downloaded_repos/sample-project
                Observation: VS Code opened for the project

                Final Answer: Project setup is complete. Dependencies are installed, missing packages handled, project opened in VS Code, and README summarized for context.
                """
        }
    ]

                # Thought: Lastly, I’ll summarize the README to provide a quick understanding of the project.
                # Action: summarize_readme
                # Action Input: downloaded_repos/sample-project
                # Observation: README summarized.


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
                - analyze_repo → "repo_path"

            Your task is to intelligently prepare the project for local development by following these steps in order:

            ---

            1. **Clone the Repository**
            - Use clone_repo to clone the GitHub repository to a local path.
            - Store the returned path and reuse it in **all later steps** — do not guess or reconstruct the path manually.

            2. **Analyze Repository**
            - Use analyze_repo immediately after cloning to generate repo_summary.txt.
            - This file will later help detect missing dependencies safely and efficiently.

            3. **Detect Dependency Files**
            - Use detect_dependencies to look for dependency files at the repo root:
                - requirements.txt
                - environment.yml
                - Pipfile
                - pyproject.toml

            4. **Create a Virtual Environment**
            - Use create_virtualenv to generate an isolated environment.
            - Pass only the environment name (e.g., my_env) as input.
            - Reuse the full environment path returned by this tool in all subsequent steps.
            - ⚠️ Never guess or manually reconstruct the virtualenv path — always use the tool output.

            5. **Ensure pip is Installed**
            - Use ensure_pip_in_env immediately after creating the environment.
            - This verifies that pip is installed and working.

            6. **Install Declared Dependencies**
            - Use install_dependencies only if supported dependency files are found:
                - requirements.txt → pip install -r
                - environment.yml → conda env create
                - Pipfile → pipenv install
                - pyproject.toml → pip install .

            7. **Analyze for Missing Dependencies**
            - Use analyze_and_install_imports to scan repo_summary.txt (generated before).
            - It will detect and install any missing packages using pip.

            8. **Launch in VS Code**
            - Use open_editor to open the repo folder in Visual Studio Code.

            

            Always follow this strict sequence:
            **Clone → Analyze Repo → Virtualenv → pip → install deps → analyze → open VS Code**

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
        max_iterations=12,
        callbacks=[CustomAgentCallbackHandler()]  # ✅ Added callback here
    )

    try:
        # 🛠️ Update: Clone + Analyze Repo first manually
        repo_path = clone_repo.invoke(repo_url)
        if "❌" in repo_path:
            return {"error": repo_path}

        # summary_result = analyze_repo.invoke(repo_path)
        # print(summary_result)

        # Then start the agent normally
        result = agent_executor.invoke({
    "input": f"Setup this GitHub repository for development: {repo_url}",
    "tools": "\n".join([t.description for t in TOOLS]),
    "tool_names": ", ".join([t.name for t in TOOLS]),
})
        print("Agent Thought Process:\n", result.get("  ", "N/A"))
        print("\nFinal Output:\n", result.get("output", "No output returned."))
        return result
    except Exception as e:
        print(f"❌ Agent failed: {e}")
        return {"error": str(e)}