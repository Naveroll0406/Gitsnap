# import os
# import subprocess
# import shutil
# import re
# from git import Repo
# from pathlib import Path
# from dotenv import load_dotenv
# from langchain_openai import AzureChatOpenAI
# from langchain.tools import tool

# # Load environment variables from .env file
# env_path = "/home/kshatra/Desktop/Git_Snap/.env"
# load_dotenv(env_path)

# # Directories for repositories and virtual environments
# REPOS_DIR = "downloaded_repos"
# VENV_DIR = "envs"
# os.makedirs(REPOS_DIR, exist_ok=True)
# os.makedirs(VENV_DIR, exist_ok=True)

# # Shared LLM instance
# llm = AzureChatOpenAI(
#     azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     temperature=0
# )


# @tool
# def clone_repo(repo_url: str) -> str:
#     """
#     Clone a GitHub repository from the given URL into a local folder.
#     Cleans up dashes, trailing slashes, and validates proper format.
#     """
#     # Clean & sanitize URL
#     repo_url = repo_url.strip().replace("—", "-").rstrip("/")

#     # Reject invalid formats (e.g., subdirectory or blob view)
#     if "/tree/" in repo_url or "/blob/" in repo_url:
#         return " Cannot clone a specific subdirectory or file from GitHub. Please provide the full repo URL (e.g., https://github.com/user/repo)."

#     # Extract clean repo name
#     try:
#         match = re.search(r"github\.com/([^/]+/[^/]+)", repo_url)
#         if not match:
#             return "❌ Invalid GitHub repo URL."
#         repo_name = match.group(1).split('/')[-1]
#     except Exception:
#         return "❌ Failed to parse repo name from URL."

#     local_path = os.path.join("downloaded_repos", repo_name)

#     # Remove existing folder if it exists
#     if os.path.exists(local_path):
#         shutil.rmtree(local_path)

#     # Clone the repo
#     try:
#         Repo.clone_from(repo_url, local_path)
#     except Exception as e:
#         return f"❌ Failed to clone repo: {str(e)}"

#     return local_path  # Return the cloned path

# @tool
# def detect_dependencies(repo_path: str) -> str:
#     """
#     Detect common dependency-related files in the given repository path.
#     Returns a comma-separated list of found files or 'none' if none are found.
#     Handles invalid paths, case insensitivity, and hidden files.
#     """
#     # Validate and normalize the repo path
#     try:
#         repo_path = Path(repo_path).resolve()
#         if not repo_path.exists() or not repo_path.is_dir():
#             return "❌ Invalid or non-existent repository path."
#     except Exception:
#         return "❌ Failed to resolve repository path."

#     # Define dependency-related files (case-insensitive)
#     dependency_files = {
#         "requirements.txt",
#         "environment.yml",
#         "Pipfile",
#         "pyproject.toml",
#         "setup.py",
#         "package.json",
#         "README.md",
#         "readme.md"
#     }

#     found_files: List[str] = []
#     try:
#         # List all files in the directory, ignoring hidden files (starting with .)
#         for item in os.listdir(repo_path):
#             if not item.startswith('.') and item.lower() in {f.lower() for f in dependency_files}:
#                 full_path = (repo_path / item).resolve()
#                 if full_path.is_file():  # Ensure it's a file, not a directory
#                     found_files.append(item)

#         # Return comma-separated list or 'none'
#         return ", ".join(found_files) if found_files else "none"
#     except PermissionError:
#         return "❌ Permission denied accessing repository path."
#     except Exception as e:
#         return f"❌ Error detecting dependencies: {str(e)}"


# @tool
# def create_virtualenv(env_name: str) -> str:
#     """
#     Create a Python virtual environment only. No pip handling.
#     """
#     env_path = Path("envs") / env_name

#     # Clean up existing environment
#     if env_path.exists():
#         shutil.rmtree(env_path, ignore_errors=True)
#         print(f"⚠️ Removed existing environment at: {env_path}")

#     try:
#         # Step 1: Create the virtual environment
#         print(f"🔧 Creating virtual environment at: {env_path}")
#         subprocess.run(["python3" if os.name != "nt" else "python", "-m", "venv", str(env_path)],
#                        check=True, capture_output=True, text=True)

#         return f"✅ Virtual environment created at: {env_path}"

#     except subprocess.CalledProcessError as e:
#         return f"❌ Failed to create virtual environment: {e.stderr}"
#     except Exception as e:
#         return f"❌ Unexpected error creating virtual environment: {str(e)}"

# @tool
# def ensure_pip_in_env(env_path: str) -> str:
#     """
#     Ensures pip is installed in the given virtual environment.
#     Uses ensurepip first, then falls back to get-pip.py if needed.
#     """
#     import urllib.request

#     # Normalize the path
#     env_path = Path(env_path).resolve()
#     if not env_path.exists():
#         return f"❌ Environment path does not exist: {env_path}"

#     # Determine OS-specific paths
#     bin_dir = "bin" if os.name != "nt" else "Scripts"
#     python_bin = env_path / bin_dir / ("python" if os.name != "nt" else "python.exe")
#     pip_path = env_path / bin_dir / ("pip" if os.name != "nt" else "pip.exe")

#     if pip_path.exists():
#         return f"✅ pip already exists at: {pip_path}"

#     try:
#         # Step 1: Try ensurepip
#         result = subprocess.run([str(python_bin), "-m", "ensurepip", "--upgrade"],
#                                 capture_output=True, text=True)
#         if result.returncode != 0:
#             print(f"⚠️ ensurepip failed: {result.stderr}")
#             raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

#     except Exception:
#         # Step 2: Fallback to get-pip.py
#         try:
#             get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
#             get_pip_path = env_path / "get-pip.py"
#             urllib.request.urlretrieve(get_pip_url, str(get_pip_path))

#             subprocess.run([str(python_bin), str(get_pip_path)],
#                            check=True, capture_output=True, text=True)
#             get_pip_path.unlink()  # Clean up

#         except Exception as ex:
#             return f"❌ Failed to install pip using get-pip.py: {str(ex)}"

#     if pip_path.exists():
#         return f"✅ pip installed successfully at: {pip_path}"
#     else:
#         return f"❌ pip installation failed. pip not found at {pip_path}"



# @tool
# def install_dependencies(arg: str) -> str:
#     """
#     Install dependencies from specified files using pip, pipenv, or conda.
#     Input: repo_path, env_path, deps_file1, deps_file2, ...
#     """
#     import subprocess
#     import os

#     try:
#         parts = [x.strip() for x in arg.split(',')]
#         if len(parts) < 3:
#             return "❌ Invalid input. Expected: repo_path, env_path, deps_file1, [deps_file2, ...]"

#         repo_path = parts[0]
#         env_path = parts[1]
#         deps_files = parts[2:]
#         pip_path = os.path.join(env_path, 'bin', 'pip')

#         if not os.path.exists(pip_path):
#             return f"❌ pip not found in: {pip_path}. Please recreate the virtual environment."

#         messages = []

#         for deps_file in deps_files:
#             file_path = os.path.join(repo_path, deps_file)

#             if deps_file == 'requirements.txt' and os.path.exists(file_path):
#                 subprocess.run([pip_path, 'install', '-r', file_path], check=True)
#                 messages.append("✅ Installed from requirements.txt")

#             elif deps_file == 'pyproject.toml' and os.path.exists(file_path):
#                 subprocess.run([pip_path, 'install', '.'], cwd=repo_path, check=True)
#                 messages.append("✅ Installed via pip from pyproject.toml")

#             elif deps_file == 'Pipfile' and os.path.exists(file_path):
#                 subprocess.run(['pipenv', 'install'], cwd=repo_path, check=True)
#                 messages.append("✅ Installed via pipenv from Pipfile")

#             elif deps_file == 'environment.yml' and os.path.exists(file_path):
#                 subprocess.run(['conda', 'env', 'create', '-f', file_path], check=True)
#                 messages.append("✅ Installed via conda from environment.yml")

#             elif deps_file.lower() in ['readme.md', 'readme'] and os.path.exists(file_path):
#                 messages.append("ℹ️ README found. Consider reviewing it for additional steps.")

#             else:
#                 messages.append(f"⚠️ Unsupported or missing file: {deps_file}")

#         return "\n".join(messages)

#     except Exception as e:
#         return f"❌ Error while installing dependencies: {str(e)}"

# @tool
# def analyze_and_install_imports(arg: str) -> str:
#     """
#     Analyze Python files for missing third-party packages using LLM,
#     generate a temp requirements file, and install them with pip.
#     Input: 'repo_path, env_path'
#     """
#     try:
#         parts = [x.strip() for x in arg.split(',')]
#         if len(parts) != 2:
#             return "❌ Invalid input. Expected: 'repo_path, env_path'"

#         repo_path = Path(parts[0]).resolve()
#         env_path = Path(parts[1]).resolve()
#         pip_path = env_path / ("bin" if os.name != "nt" else "Scripts") / ("pip" if os.name != "nt" else "pip.exe")

#         if not pip_path.exists():
#             return f"❌ pip not found in: {pip_path}. Please recreate the virtual environment."

#         subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         python_files = [os.path.join(root, file)
#                         for root, _, files in os.walk(repo_path)
#                         for file in files if file.endswith(".py")]

#         if not python_files:
#             return "✅ No Python files found in the repository."

#         all_packages = set()
#         for py_file in python_files:
#             with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
#                 code = f.read()

#             prompt = f"""
#                 You are a DevOps assistant. Based on the following Python code, list only third-party pip-installable packages (comma-separated, no explanation).
#                 Code:
#                 {code}
#             """
#             response = llm.invoke(prompt).content.strip()

#             if any(bad in response.lower() for bad in ["install", "example", "usage", "like", "you can use", "such as"]):
#                 continue

#             raw_packages = [p.strip().lower() for p in response.split(',') if p.strip()]
#             for pkg in raw_packages:
#                 if re.match(r"^[a-z0-9_.\-]+$", pkg):
#                     all_packages.add(pkg)

#         if not all_packages:
#             return "✅ No missing third-party packages found."

#         temp_req_path = repo_path / "temp_requirements.txt"
#         with open(temp_req_path, "w") as f:
#             for pkg in sorted(all_packages):
#                 f.write(pkg + "\n")

#         install_result = subprocess.run(
#             [str(pip_path), 'install', '-r', str(temp_req_path)],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True
#         )
#         temp_req_path.unlink()

#         if install_result.returncode != 0:
#             return (
#                 f"⚠️ Some packages may have failed to install.\n"
#                 f"Packages attempted: {', '.join(sorted(all_packages))}\n"
#                 f"Error:\n{install_result.stderr}"
#             )

#         return f"✅ Successfully installed missing packages: {', '.join(sorted(all_packages))}"

#     except Exception as e:
#         return f"❌ Error during import analysis: {str(e)}"
   

# @tool
# def open_editor(path: str) -> str:
#     """
#     Open the given project path in Visual Studio Code.
#     """
#     subprocess.Popen(['code', path])
#     return f"✅ VS Code opened at: {path}"

# @tool
# def summarize_readme(repo_path: str) -> str:
#     """
#     Read the README.md file from the repository and summarize it using Azure OpenAI.
#     """
#     readme_path = os.path.join(repo_path, 'README.md')
#     if not os.path.exists(readme_path):
#         return "README.md not found."

#     with open(readme_path, "r", encoding="utf-8") as f:
#         content = f.read()

#     summary = llm.invoke(f"Summarize the following README:\n\n{content}").content.strip()
#     return f"📄 README Summary:\n{summary}"


import os
import subprocess
import shutil
import re
from git import Repo
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool

# Load environment variables from .env file
env_path = "/home/kshatra/Desktop/Git_Snap/.env"
load_dotenv(env_path)

# Directories for repositories and virtual environments
REPOS_DIR = Path("downloaded_repos").resolve()
VENV_DIR = Path("envs").resolve()
REPOS_DIR.mkdir(parents=True, exist_ok=True)
VENV_DIR.mkdir(parents=True, exist_ok=True)

# Shared LLM instance
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

@tool
def clone_repo(repo_url: str) -> str:
    """
    Clone a GitHub repository from the given URL into a local folder.
    Cleans up dashes, trailing slashes, and validates proper format.
    """
    # Clean & sanitize URL
    repo_url = repo_url.strip().replace("—", "-").rstrip("/")

    # Reject invalid formats (e.g., subdirectory or blob view)
    if "/tree/" in repo_url or "/blob/" in repo_url:
        return " Cannot clone a specific subdirectory or file from GitHub. Please provide the full repo URL (e.g., https://github.com/user/repo)."

    # Extract clean repo name
    try:
        match = re.search(r"github\.com/([^/]+/[^/]+)", repo_url)
        if not match:
            return "❌ Invalid GitHub repo URL."
        repo_name = match.group(1).split('/')[-1]
    except Exception:
        return "❌ Failed to parse repo name from URL."

    local_path = os.path.join("downloaded_repos", repo_name)
    Repo.clone_from(repo_url, local_path)   
    return str(Path(local_path).resolve())  # ✅ absolute path


@tool
def detect_dependencies(repo_path: str) -> str:
    """
    Detect common dependency-related files in a repository.
    Looks for files like requirements.txt, Pipfile, etc., and returns them.
    """
    try:
        repo_path = Path(repo_path).resolve()
        if not repo_path.exists() or not repo_path.is_dir():
            return "❌ Invalid or non-existent repository path."
    except Exception:
        return "❌ Failed to resolve repository path."

    dependency_files = {
        "requirements.txt", "environment.yml", "Pipfile",
        "pyproject.toml", "setup.py", "package.json",
        "README.md", "readme.md"
    }

    found_files = []
    try:
        for item in os.listdir(repo_path):
            if not item.startswith('.') and item.lower() in {f.lower() for f in dependency_files}:
                full_path = (repo_path / item).resolve()
                if full_path.is_file():
                    found_files.append(item)
        return ", ".join(found_files) if found_files else "none"
    except PermissionError:
        return "❌ Permission denied accessing repository path."
    except Exception as e:
        return f"❌ Error detecting dependencies: {str(e)}"

@tool
def create_virtualenv(env_name: str) -> str:
    """
    Create a Python virtual environment and return its path.
    The caller is responsible for using the returned path to derive pip_path.
    """
    env_path = VENV_DIR / env_name
    print("env_path", env_path)
    if env_path.exists():

        shutil.rmtree(env_path, ignore_errors=True)
        print(f"⚠️ Removed existing environment at: {env_path}")

    try:
        print(f"🔧 Creating virtual environment at: {env_path}")
        subprocess.run(
            ["python3" if os.name != "nt" else "python", "-m", "venv", str(env_path)],
            check=True, capture_output=True, text=True
        )
        return str(env_path)
    except subprocess.CalledProcessError as e:
        return f"❌ Failed to create virtual environment: {e.stderr}"
    except Exception as e:
        return f"❌ Unexpected error creating virtual environment: {str(e)}"


@tool
def ensure_pip_in_env(env_path: str) -> str:
    """
    Ensure pip is installed and working inside a virtual environment.
    Uses ensurepip or get-pip.py if pip is missing.
    """
    env_path = Path(env_path).resolve()
    bin_dir = "bin" if os.name != "nt" else "Scripts"
    python_bin = env_path / bin_dir / ("python" if os.name != "nt" else "python.exe")
    pip_path = env_path / bin_dir / ("pip" if os.name != "nt" else "pip.exe")

    if pip_path.exists():
        try:
            subprocess.run([str(pip_path), "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return str(pip_path)
        except Exception:
            pass  # We'll try to reinstall pip below

    try:
        subprocess.run([str(python_bin), "-m", "ensurepip", "--upgrade"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        try:
            get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
            get_pip_path = env_path / "get-pip.py"
            urllib.request.urlretrieve(get_pip_url, str(get_pip_path))
            subprocess.run([str(python_bin), str(get_pip_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            get_pip_path.unlink()
        except Exception as ex:
            raise RuntimeError(f"Failed to install pip: {str(ex)}")

    if not pip_path.exists():
        raise FileNotFoundError(f"pip still not found at {pip_path}")

    return str(pip_path)


# @tool
# def install_dependencies(arg: str) -> str:
#     """
#     Install dependencies from supported files.
#     Input format: 'repo_path, pip_path, deps_file1, deps_file2, ...'
#     """
#     try:
#         parts = [x.strip() for x in arg.split(',')]
#         if len(parts) < 3:
#             return "❌ Invalid input. Expected: repo_path, pip_path, deps_file1, [deps_file2, ...]"

#         repo_path = Path(parts[0]).resolve()
#         pip_path = Path(parts[1]).resolve()
#         deps_files = parts[2:]

#         # Verify pip works
#         try:
#             subprocess.run([str(pip_path), "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         except Exception:
#             return f"❌ pip is not working at: {pip_path}"

#         messages = []
#         for deps_file in deps_files:
#             file_path = repo_path / deps_file

#             if deps_file == 'requirements.txt' and file_path.exists():
#                 subprocess.run([str(pip_path), 'install', '-r', str(file_path)], check=True)
#                 messages.append("✅ Installed from requirements.txt")
#             elif deps_file == 'pyproject.toml' and file_path.exists():
#                 subprocess.run([str(pip_path), 'install', '.'], cwd=str(repo_path), check=True)
#                 messages.append("✅ Installed via pip from pyproject.toml")
#             elif deps_file == 'Pipfile' and file_path.exists():
#                 subprocess.run(['pipenv', 'install'], cwd=str(repo_path), check=True)
#                 messages.append("✅ Installed via pipenv from Pipfile")
#             elif deps_file == 'environment.yml' and file_path.exists():
#                 subprocess.run(['conda', 'env', 'create', '-f', str(file_path)], check=True)
#                 messages.append("✅ Installed via conda from environment.yml")
#             elif deps_file.lower() in ['readme.md', 'readme'] and file_path.exists():
#                 messages.append("ℹ️ README found. Consider reviewing it for additional steps.")
#             else:
#                 messages.append(f"⚠️ Unsupported or missing file: {deps_file}")

#         return "\n".join(messages)
#     except Exception as e:
#         return f"❌ Error while installing dependencies: {str(e)}"

from pathlib import Path
import subprocess
from typing import Union
from langchain.tools import tool

@tool
def install_dependencies(arg: Union[str, dict]) -> str:
    """
    Install dependencies using either:
    - File-based approach: 'repo_path, pip_path, requirements.txt, ...'
    - Direct install: {'packages': [...], 'pip_path': '...'}
    """
    try:
        # ✅ Mode 1: JSON-style direct pip install
        if isinstance(arg, dict):
            packages = arg.get("packages", [])
            pip_path = Path(arg.get("pip_path", "")).resolve()

            if not packages:
                return "✅ No packages to install."
            if not pip_path.exists():
                return f"❌ pip not found at {pip_path}"

            result = subprocess.run(
                [str(pip_path), "install", *packages],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode == 0:
                return f"✅ Successfully installed: {', '.join(packages)}"
            else:
                return f"⚠️ Install failed.\nError: {result.stderr}"

        # ✅ Mode 2: File-based format
        elif isinstance(arg, str):
            parts = [x.strip() for x in arg.split(',')]
            if len(parts) < 3:
                return "❌ Invalid input. Expected: repo_path, pip_path, deps_file1, [deps_file2, ...]"

            repo_path = Path(parts[0]).resolve()
            pip_path = Path(parts[1]).resolve()
            deps_files = parts[2:]

            # Verify pip works
            try:
                subprocess.run([str(pip_path), "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                return f"❌ pip is not working at: {pip_path}"

            messages = []
            for deps_file in deps_files:
                file_path = repo_path / deps_file

                if deps_file == 'requirements.txt' and file_path.exists():
                    subprocess.run([str(pip_path), 'install', '-r', str(file_path)], check=True)
                    messages.append("✅ Installed from requirements.txt")
                elif deps_file == 'pyproject.toml' and file_path.exists():
                    subprocess.run([str(pip_path), 'install', '.'], cwd=str(repo_path), check=True)
                    messages.append("✅ Installed via pip from pyproject.toml")
                elif deps_file == 'Pipfile' and file_path.exists():
                    subprocess.run(['pipenv', 'install'], cwd=str(repo_path), check=True)
                    messages.append("✅ Installed via pipenv from Pipfile")
                elif deps_file == 'environment.yml' and file_path.exists():
                    subprocess.run(['conda', 'env', 'create', '-f', str(file_path)], check=True)
                    messages.append("✅ Installed via conda from environment.yml")
                elif deps_file.lower() in ['readme.md', 'readme'] and file_path.exists():
                    messages.append("ℹ️ README found. Consider reviewing it for additional steps.")
                else:
                    messages.append(f"⚠️ Unsupported or missing file: {deps_file}")

            return "\n".join(messages)

        else:
            return "❌ Invalid input type. Expected str or dict."

    except Exception as e:
        return f"❌ Error while installing dependencies: {str(e)}"


@tool
def analyze_and_install_imports(arg: str) -> str:
    """
    Analyze Python and other relevant files to detect pip packages using LLM,
    then install them using `install_dependencies`.

    Input format: 'repo_abs_path, venv_abs_path'
    """
    try:
        parts = [x.strip() for x in arg.split(',')]
        if len(parts) != 2:
            return "❌ Invalid input. Format must be: 'repo_abs_path, venv_abs_path'"

        repo_path = Path(parts[0]).resolve(strict=True)
        env_path = Path(parts[1]).resolve(strict=True)

        pip_path = ensure_pip_in_env(str(env_path))
        subprocess.run([pip_path, 'install', '--upgrade', 'pip'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        valid_exts = ('.py', '.ipynb', '.txt', '.md', 'requirements.txt')
        relevant_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(repo_path)
            for file in files if file.endswith(valid_exts)
        ]

        if not relevant_files:
            return "✅ No relevant source files found in the repository."

        all_packages = set()
        for file_path in relevant_files:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            prompt = f"""
                From the following code or documentation, extract third-party pip-installable packages.
                List them comma-separated, no explanation.
                Content:
                {content}
            """
            response = llm.invoke(prompt).content.strip()
            found = [p.strip().lower() for p in response.split(',') if re.match(r"^[a-z0-9_.\-]+$", p.strip())]
            all_packages.update(found)

        if not all_packages:
            return "✅ No missing dependencies detected."

        # ✅ Call install_dependencies instead of subprocess
        install_input = {
            "packages": sorted(list(all_packages)),
            "pip_path": str(pip_path)
        }
        result = install_dependencies.invoke(install_input)
        return result if isinstance(result, str) else str(result)

    except Exception as e:
        return f"❌ Error during analysis or installation: {str(e)}"

@tool
def open_editor(path: str) -> str:
    """
    Open the provided path in Visual Studio Code.
    Returns the confirmation message.
    """
    path = Path(path).resolve()
    subprocess.Popen(['code', str(path)])
    return f"✅ VS Code opened at: {path}"

@tool
def summarize_readme(repo_path: str) -> str:
    """
    Summarize the README.md file using the LLM.
    Extracts content from the file and generates a summary.
    """
    readme_path = Path(repo_path).resolve() / 'README.md'
    if not readme_path.exists():
        return "README.md not found."
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    summary = llm.invoke(f"Summarize the following README:\n\n{content}").content.strip()
    return f"📄 README Summary:\n{summary}"
