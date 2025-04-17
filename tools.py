import os
import subprocess
import shutil
import urllib.request
import re
from typing import Union
from git import Repo
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
env = "/home/kshatra/Desktop/Git_Snap/.env"
load_dotenv(env)

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
    If the destination path exists and is not empty, it will be deleted.
    """
    repo_url = repo_url.strip().replace("—", "-").rstrip("/")
    if "/tree/" in repo_url or "/blob/" in repo_url:
        return "❌ Cannot clone a specific subdirectory or file from GitHub. Please provide the full repo URL (e.g., https://github.com/user/repo)."

    try:
        match = re.search(r"github\.com/([^/]+/[^/]+)", repo_url)
        if not match:
            return "❌ Invalid GitHub repo URL."
        repo_name = match.group(1).split('/')[-1]
    except Exception:
        return "❌ Failed to parse repo name from URL."

    local_path = REPOS_DIR / repo_name
    if local_path.exists() and any(local_path.iterdir()):
        shutil.rmtree(local_path)

    try:
        Repo.clone_from(repo_url, str(local_path))
    except Exception as e:
        return f"❌ Git clone failed: {str(e)}"

    return str(local_path.resolve())

@tool
def detect_dependencies(repo_path: str) -> str:
    """
    Detects common dependency files (e.g., requirements.txt, Pipfile) in a given repo path.
    Returns a comma-separated list of found files or an error message.
    """
    try:
        repo_path = Path(repo_path.strip()).resolve()
        if not repo_path.exists():
            repo_path = REPOS_DIR / repo_path.name
        if not repo_path.exists():
            return "❌ Path does not exist."
        if not repo_path.is_dir():
            return "❌ Path exists but is not a directory."
    except Exception as e:
        return f"❌ Failed to resolve repository path. Error: {str(e)}"

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
        return ", ".join(found_files) if found_files else "No dependency files found."
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
    if env_path.exists():
        shutil.rmtree(env_path, ignore_errors=True)

    try:
        subprocess.run(
            ["python3" if os.name != "nt" else "python", "-m", "venv", str(env_path)],
            check=True, capture_output=True, text=True
        )
        return str(env_path.resolve())
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
    if not str(env_path).startswith(str(VENV_DIR)):
        env_path = VENV_DIR / env_path.name

    bin_dir = "bin" if os.name != "nt" else "Scripts"
    python_bin = env_path / bin_dir / ("python" if os.name != "nt" else "python.exe")
    pip_path = env_path / bin_dir / ("pip" if os.name != "nt" else "pip.exe")

    if pip_path.exists():
        try:
            subprocess.run([str(pip_path), "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return str(pip_path)
        except Exception:
            pass

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
# def install_dependencies(arg: Union[str, dict]) -> str:
#     """
#     Install dependencies using either:
#     - File-based approach: 'repo_path, pip_path, requirements.txt, ...'
#     - Direct install: {'packages': [...], 'pip_path': '...'}
#     Returns a summary of installation results.
#     """
#     try:
#         if isinstance(arg, dict):
#             packages = arg.get("packages", [])
#             pip_path = Path(arg.get("pip_path", "").strip())

#             # ✅ Force pip_path under VENV_DIR if not absolute
#             if not pip_path.is_absolute() or not pip_path.exists():
#                 pip_path = VENV_DIR / pip_path.name / ("bin/pip" if os.name != "nt" else "Scripts/pip.exe")

#             pip_path = pip_path.resolve()

#             if not packages:
#                 return "✅ No packages to install."
#             if not pip_path.exists():
#                 return f"❌ pip not found at {pip_path}"

#             result = subprocess.run(
#                 [str(pip_path), "install", *packages],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True
#             )

#             if result.returncode == 0:
#                 return f"✅ Successfully installed: {', '.join(packages)}"
#             else:
#                 return f"⚠️ Install failed.\nError: {result.stderr}"

#         elif isinstance(arg, str):
#             parts = [x.strip() for x in arg.split(',')]
#             if len(parts) < 3:
#                 return "❌ Invalid input. Expected: repo_path, pip_path, deps_file1, [deps_file2, ...]"

#             repo_path = Path(parts[0])
#             pip_path = Path(parts[1])
#             deps_files = parts[2:]

#             # ✅ Auto-fix repo_path and pip_path
#             if not repo_path.is_absolute() or not repo_path.exists():
#                 repo_path = REPOS_DIR / repo_path.name
#             if not pip_path.is_absolute() or not pip_path.exists():
#                 pip_path = VENV_DIR / pip_path.name / ("bin/pip" if os.name != "nt" else "Scripts/pip.exe")

#             repo_path = repo_path.resolve()
#             pip_path = pip_path.resolve()

#             messages = []

#             try:
#                 subprocess.run([str(pip_path), "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             except Exception:
#                 return f"❌ pip is not working at: {pip_path}"

#             for deps_file in deps_files:
#                 deps_file = deps_file.strip()
#                 file_path = repo_path / deps_file

#                 try:
#                     if deps_file.lower() == 'requirements.txt' and file_path.exists():
#                         subprocess.run([str(pip_path), 'install', '-r', str(file_path)], check=True)
#                         messages.append("✅ Installed from requirements.txt")
#                     elif deps_file.lower() == 'pyproject.toml' and file_path.exists():
#                         subprocess.run([str(pip_path), 'install', '.'], cwd=str(repo_path), check=True)
#                         messages.append("✅ Installed via pip from pyproject.toml")
#                     elif deps_file == 'Pipfile' and file_path.exists():
#                         if shutil.which("pipenv") is None:
#                             messages.append("❌ pipenv is not installed.")
#                         else:
#                             subprocess.run(['pipenv', 'install'], cwd=str(repo_path), check=True)
#                             messages.append("✅ Installed via pipenv from Pipfile")
#                     elif deps_file == 'environment.yml' and file_path.exists():
#                         if shutil.which("conda") is None:
#                             messages.append("❌ conda is not installed.")
#                         else:
#                             subprocess.run(['conda', 'env', 'create', '-f', str(file_path)], check=True)
#                             messages.append("✅ Installed via conda from environment.yml")
#                     elif deps_file.lower() in ['readme.md', 'readme'] and file_path.exists():
#                         messages.append("ℹ️ README found. Consider reviewing it for additional steps.")
#                     else:
#                         messages.append(f"⚠️ Unsupported or missing file: {deps_file}")
#                 except subprocess.CalledProcessError as err:
#                     messages.append(f"❌ Installation failed for {deps_file}: {err.stderr}")
#                 except Exception as e:
#                     messages.append(f"❌ Unexpected error for {deps_file}: {str(e)}")

#             return "\n".join(messages)

#         else:
#             return "❌ Invalid input type. Expected str or dict."

#     except Exception as e:
#         return f"❌ Error while installing dependencies: {str(e)}"

# @tool
# def install_dependencies(arg: Union[str, dict]) -> str:
#     """
#     Install dependencies using either:
#     - File-based approach: 'repo_path, pip_path, requirements.txt, ...'
#     - Direct install: {"packages": [...], "pip_path": "..."}
#     Returns a summary of installation results.
#     """
#     try:
#         # ✅ Support LangChain-style nested dict with 'arg' key
#         if isinstance(arg, dict) and "packages" in arg:
#             packages = arg.get("packages", [])
#             pip_path = Path(arg.get("pip_path", "").strip())

#             # Fix relative paths if needed
#             if not pip_path.is_absolute() or not pip_path.exists():
#                 pip_path = VENV_DIR / pip_path.name / ("bin/pip" if os.name != "nt" else "Scripts/pip.exe")

#             pip_path = pip_path.resolve()

#             if not packages:
#                 return "✅ No packages to install."
#             if not pip_path.exists():
#                 return f"❌ pip not found at {pip_path}"

#             result = subprocess.run(
#                 [str(pip_path), "install", *packages],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True
#             )

#             if result.returncode == 0:
#                 return f"✅ Successfully installed: {', '.join(packages)}"
#             else:
#                 return f"⚠️ Install failed.\nError: {result.stderr}"

#         elif isinstance(arg, str):
#             parts = [x.strip() for x in arg.split(',')]
#             if len(parts) < 3:
#                 return "❌ Invalid input. Expected: repo_path, pip_path, deps_file1, [deps_file2, ...]"

#             repo_path = Path(parts[0])
#             pip_path = Path(parts[1])
#             deps_files = parts[2:]

#             # Fix paths
#             if not repo_path.is_absolute() or not repo_path.exists():
#                 repo_path = REPOS_DIR / repo_path.name
#             if not pip_path.is_absolute() or not pip_path.exists():
#                 pip_path = VENV_DIR / pip_path.name / ("bin/pip" if os.name != "nt" else "Scripts/pip.exe")

#             repo_path = repo_path.resolve()
#             pip_path = pip_path.resolve()

#             messages = []

#             try:
#                 subprocess.run([str(pip_path), "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             except Exception:
#                 return f"❌ pip is not working at: {pip_path}"

#             for deps_file in deps_files:
#                 deps_file = deps_file.strip()
#                 file_path = repo_path / deps_file

#                 try:
#                     if deps_file.lower() == 'requirements.txt' and file_path.exists():
#                         subprocess.run([str(pip_path), 'install', '-r', str(file_path)], check=True)
#                         messages.append("✅ Installed from requirements.txt")
#                     elif deps_file.lower() == 'pyproject.toml' and file_path.exists():
#                         subprocess.run([str(pip_path), 'install', '.'], cwd=str(repo_path), check=True)
#                         messages.append("✅ Installed via pip from pyproject.toml")
#                     elif deps_file == 'Pipfile' and file_path.exists():
#                         if shutil.which("pipenv") is None:
#                             messages.append("❌ pipenv is not installed.")
#                         else:
#                             subprocess.run(['pipenv', 'install'], cwd=str(repo_path), check=True)
#                             messages.append("✅ Installed via pipenv from Pipfile")
#                     elif deps_file == 'environment.yml' and file_path.exists():
#                         if shutil.which("conda") is None:
#                             messages.append("❌ conda is not installed.")
#                         else:
#                             subprocess.run(['conda', 'env', 'create', '-f', str(file_path)], check=True)
#                             messages.append("✅ Installed via conda from environment.yml")
#                     elif deps_file.lower() in ['readme.md', 'readme'] and file_path.exists():
#                         messages.append("ℹ️ README found. Consider reviewing it for additional steps.")
#                     else:
#                         messages.append(f"⚠️ Unsupported or missing file: {deps_file}")
#                 except subprocess.CalledProcessError as err:
#                     messages.append(f"❌ Installation failed for {deps_file}: {err.stderr}")
#                 except Exception as e:
#                     messages.append(f"❌ Unexpected error for {deps_file}: {str(e)}")

#             return "\n".join(messages)

#         else:
#             return "❌ Invalid input. Expected either a formatted string or dictionary with 'packages' and 'pip_path'."

#     except Exception as e:
#         return f"❌ Error while installing dependencies: {str(e)}"

@tool
def install_dependencies(arg: Union[str, dict]) -> str:
    """
    Install dependencies using either:
    - File-based approach: 'repo_path, pip_path, requirements.txt, ...'
    - Direct install: {"packages": [...], "pip_path": "..."}
    Returns a summary of installation results.
    """
    try:
        def extract_failed_package(stderr: str) -> str:
            match = re.search(r"satisfies the requirement ([\w\-\.]+)", stderr)
            return match.group(1) if match else None

        def llm_correct_package_name(failed_package: str) -> str:
            prompt = f"""
You attempted to install the Python package '{failed_package}' using pip, but it failed.

Please return the correct pip-installable package name if '{failed_package}' was incorrect.

Only return the corrected pip name. If invalid, return "NONE".
"""
            chain = ChatPromptTemplate.from_template("{text}") | llm | StrOutputParser()
            corrected = chain.invoke({"text": prompt}).strip().lower()
            return corrected if corrected != "none" else None

        # ✅ Direct install mode
        if isinstance(arg, dict) and "packages" in arg:
            packages = arg.get("packages", [])
            pip_path = Path(arg.get("pip_path", "").strip())

            if not pip_path.is_absolute() or not pip_path.exists():
                pip_path = VENV_DIR / pip_path.name / ("bin/pip" if os.name != "nt" else "Scripts/pip.exe")

            pip_path = pip_path.resolve()

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
                failed_pkg = extract_failed_package(result.stderr)
                if failed_pkg:
                    corrected = llm_correct_package_name(failed_pkg)
                    if corrected and corrected != failed_pkg:
                        retry = subprocess.run(
                            [str(pip_path), "install", corrected],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        if retry.returncode == 0:
                            return f"⚠️ '{failed_pkg}' failed, but corrected to '{corrected}' and installed successfully."
                        else:
                            return f"❌ Retry failed for corrected package '{corrected}'.\nOriginal error: {result.stderr}"
                return f"⚠️ Install failed.\nError: {result.stderr}"

        # ✅ File-based input mode
        elif isinstance(arg, str):
            parts = [x.strip() for x in arg.split(',')]
            if len(parts) < 3:
                return "❌ Invalid input. Expected: repo_path, pip_path, deps_file1, [deps_file2, ...]"

            repo_path = Path(parts[0])
            pip_path = Path(parts[1])
            deps_files = parts[2:]

            if not repo_path.is_absolute() or not repo_path.exists():
                repo_path = REPOS_DIR / repo_path.name
            if not pip_path.is_absolute() or not pip_path.exists():
                pip_path = VENV_DIR / pip_path.name / ("bin/pip" if os.name != "nt" else "Scripts/pip.exe")

            repo_path = repo_path.resolve()
            pip_path = pip_path.resolve()

            messages = []

            try:
                subprocess.run([str(pip_path), "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                return f"❌ pip is not working at: {pip_path}"

            for deps_file in deps_files:
                deps_file = deps_file.strip()
                file_path = repo_path / deps_file

                try:
                    if deps_file.lower() == 'requirements.txt' and file_path.exists():
                        subprocess.run([str(pip_path), 'install', '-r', str(file_path)], check=True)
                        messages.append("✅ Installed from requirements.txt")
                    elif deps_file.lower() == 'pyproject.toml' and file_path.exists():
                        subprocess.run([str(pip_path), 'install', '.'], cwd=str(repo_path), check=True)
                        messages.append("✅ Installed via pip from pyproject.toml")
                    elif deps_file == 'Pipfile' and file_path.exists():
                        if shutil.which("pipenv") is None:
                            messages.append("❌ pipenv is not installed.")
                        else:
                            subprocess.run(['pipenv', 'install'], cwd=str(repo_path), check=True)
                            messages.append("✅ Installed via pipenv from Pipfile")
                    elif deps_file == 'environment.yml' and file_path.exists():
                        if shutil.which("conda") is None:
                            messages.append("❌ conda is not installed.")
                        else:
                            subprocess.run(['conda', 'env', 'create', '-f', str(file_path)], check=True)
                            messages.append("✅ Installed via conda from environment.yml")
                    elif deps_file.lower() in ['readme.md', 'readme'] and file_path.exists():
                        messages.append("ℹ️ README found. Consider reviewing it for additional steps.")
                    else:
                        messages.append(f"⚠️ Unsupported or missing file: {deps_file}")
                except subprocess.CalledProcessError as err:
                    messages.append(f"❌ Installation failed for {deps_file}: {err.stderr}")
                except Exception as e:
                    messages.append(f"❌ Unexpected error for {deps_file}: {str(e)}")

            return "\n".join(messages)

        else:
            return "❌ Invalid input. Expected either a formatted string or dictionary with 'packages' and 'pip_path'."

    except Exception as e:
        return f"❌ Error while installing dependencies: {str(e)}"


""" 
this will create an auto env inside this tool and install the dependencies
"""

# @tool
# def analyze_and_install_imports(arg: str) -> str:

#     """
#     Analyze Python and related files in a repo to detect third-party pip packages using LLM,
#     and install them into the specified virtual environment. Auto-creates the environment if missing.

#     Input format: 'repo_abs_path, venv_abs_path'
#     """
#     try:
#         parts = [x.strip() for x in arg.split(',')]
#         if len(parts) != 2:
#             return "❌ Invalid input. Format must be: 'repo_abs_path, venv_abs_path'"

#         repo_path = Path(parts[0]).resolve()
#         if not repo_path.exists():
#             repo_path = REPOS_DIR / repo_path.name

#         env_path = Path(parts[1]).resolve()
#         if not str(env_path).startswith(str(VENV_DIR)):
#             env_path = VENV_DIR / env_path.name

#         if not env_path.exists():
#             create_virtualenv.invoke(env_path.name)

#         pip_path = ensure_pip_in_env.invoke(str(env_path))
#         subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         valid_exts = ('.py', '.ipynb', '.txt', '.md', 'requirements.txt')
#         relevant_files = [
#             os.path.join(root, file)
#             for root, _, files in os.walk(repo_path)
#             for file in files if file.endswith(valid_exts)
#         ]

#         if not relevant_files:
#             return "✅ No relevant source files found in the repository."

#         all_packages = set()
#         for file_path in relevant_files:
#             with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                 content = f.read()

#             prompt = f"""
#                 List all third-party Python packages used or referenced in the following content.
#                 Only return a comma-separated list of pip-installable package names. No explanation.

#                 Content:
#                 {content}
#             """

#             chain = ChatPromptTemplate.from_template("{text}") | llm | StrOutputParser()
#             response = chain.invoke({"text": prompt})
#             response = response.replace('\n', ',').replace('-', '').replace('•', '').replace('*', '')
#             found = [p.strip().lower() for p in response.split(',') if re.match(r"^[a-z0-9_.\-]+$", p.strip())]
#             all_packages.update(found)

#         if not all_packages:
#             return "✅ No missing dependencies detected."

#         install_input = {
#             "packages": sorted(list(all_packages)),
#             "pip_path": str(pip_path)
#         }
#         result = install_dependencies.invoke({"arg": install_input})
#         return result if isinstance(result, str) else str(result)

#     except Exception as e:
#         return f"❌ Error during analysis or installation: {str(e)}"

# @tool
# def analyze_and_install_imports(arg: str) -> str:
#     """
#     Analyze Python and related files in a repo to detect third-party pip packages using LLM,
#     and install them into the specified virtual environment. Auto-creates the environment if missing.

#     Input format: 'repo_abs_path, venv_abs_path'
#     """
#     try:
#         parts = [x.strip() for x in arg.split(',')]
#         if len(parts) != 2:
#             return "❌ Invalid input. Format must be: 'repo_abs_path, venv_abs_path'"

#         repo_path_str, env_path_str = parts

#         # 🚫 Check if pip path is passed instead of env path
#         if env_path_str.endswith("pip") or "bin/pip" in env_path_str or "Scripts\\pip.exe" in env_path_str:
#             return "❌ Invalid input: You passed a pip binary path. Please provide the virtual environment directory path (not the pip executable)."

#         repo_path = Path(repo_path_str).resolve()
#         if not repo_path.exists():
#             repo_path = REPOS_DIR / repo_path.name

#         env_path = Path(env_path_str).resolve()
#         if not str(env_path).startswith(str(VENV_DIR)):
#             env_path = VENV_DIR / env_path.name

#         if not env_path.exists():
#             create_virtualenv.invoke(env_path.name)

#         pip_path = ensure_pip_in_env.invoke(str(env_path))
#         subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         valid_exts = ('.py', '.ipynb', '.txt', '.md', 'requirements.txt')
#         relevant_files = [
#             os.path.join(root, file)
#             for root, _, files in os.walk(repo_path)
#             for file in files if file.endswith(valid_exts)
#         ]

#         if not relevant_files:
#             return "✅ No relevant source files found in the repository."

#         all_packages = set()
#         for file_path in relevant_files:
#             with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                 content = f.read()

#             prompt = f"""
#                 List all third-party Python packages used or referenced in the following content.
#                 Only return a comma-separated list of pip-installable package names. No explanation.

#                 Content:
#                 {content}
#             """

#             chain = ChatPromptTemplate.from_template("{text}") | llm | StrOutputParser()
#             response = chain.invoke({"text": prompt})
#             response = response.replace('\n', ',').replace('-', '').replace('•', '').replace('*', '')
#             found = [p.strip().lower() for p in response.split(',') if re.match(r"^[a-z0-9_.\\-]+$", p.strip())]
#             all_packages.update(found)

#         if not all_packages:
#             return "✅ No missing dependencies detected."

#         install_input = {
#             "packages": sorted(list(all_packages)),
#             "pip_path": str(pip_path)
#         }
#         result = install_dependencies.invoke({"arg": install_input})
#         return result if isinstance(result, str) else str(result)

#     except Exception as e:
#         return f"❌ Error during analysis or installation: {str(e)}"
# @tool
# def analyze_and_install_imports(arg: str) -> str:
#     """
#     Analyze Python and related files in a repo to detect third-party pip packages using LLM,
#     and install them into the specified virtual environment. Auto-creates the environment if missing.

#     Input format: 'repo_abs_path, venv_abs_path'
#     """
#     try:
#         parts = [x.strip() for x in arg.split(',')]
#         if len(parts) != 2:
#             return "❌ Invalid input. Format must be: 'repo_abs_path, venv_abs_path'"

#         repo_path_str, env_path_str = parts

#         # 🚫 Check if pip path is passed instead of env path
#         if env_path_str.endswith("pip") or "bin/pip" in env_path_str or "Scripts\\pip.exe" in env_path_str:
#             return "❌ Invalid input: You passed a pip binary path. Please provide the virtual environment directory path (not the pip executable)."

#         repo_path = Path(repo_path_str).resolve()
#         if not repo_path.exists():
#             repo_path = REPOS_DIR / repo_path.name

#         env_path = Path(env_path_str).resolve()
#         if not str(env_path).startswith(str(VENV_DIR)):
#             env_path = VENV_DIR / env_path.name

#         if not env_path.exists():
#             create_virtualenv.invoke(env_path.name)

#         pip_path = ensure_pip_in_env.invoke(str(env_path))
#         subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         valid_exts = ('.py', '.ipynb', '.txt', '.md', 'requirements.txt')
#         relevant_files = [
#             os.path.join(root, file)
#             for root, _, files in os.walk(repo_path)
#             for file in files if file.endswith(valid_exts)
#         ]

#         if not relevant_files:
#             return "✅ No relevant source files found in the repository."

#         # ✅ Alias corrections for common pip package mistakes
#         PACKAGE_ALIASES = {
#             "sklearn": "scikit-learn",
#             "scikitlearn": "scikit-learn",  # ✅ Add this!
#             "cv2": "opencv-python",
#             "bs4": "beautifulsoup4",
#             "pil": "pillow",
#             "yaml": "pyyaml",
#             "crypto": "pycryptodome",
#             "tensorflow-gpu": "tensorflow",
#             "torchvision": "torchvision",
#             "tflearn": "tflearn",
#             "keras": "keras",
#             "flask": "flask",
#             "django": "django",
#             "requests": "requests",
#             "nltk": "nltk",
#             "pytz": "pytz",
#             "pymongo": "pymongo",
#         }

#         all_packages = set()
#         for file_path in relevant_files:
#             with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                 content = f.read()

#             # ✅ Smarter prompt to guide LLM
#             prompt = f"""
#                 Extract a comma-separated list of pip-installable Python package names based on the following code or documentation.

#                 You MUST convert import/module names to their correct pip package names.

#                 Examples:
#                 - 'sklearn' → 'scikit-learn'
#                 - 'scikitlearn' → 'scikit-learn'  # ✅ Include this
#                 - 'cv2' → 'opencv-python'
#                 - 'bs4' → 'beautifulsoup4'
#                 - 'PIL' → 'pillow'
#                 - 'yaml' → 'pyyaml'

#                 Return only a comma-separated list of pip packages. No explanations.

#                 Content:
#                 {content}
#             """

#             chain = ChatPromptTemplate.from_template("{text}") | llm | StrOutputParser()
#             response = chain.invoke({"text": prompt})
#             response = response.replace('\n', ',').replace('-', '').replace('•', '').replace('*', '')
#             found = [p.strip().lower() for p in response.split(',') if re.match(r"^[a-z0-9_.\\-]+$", p.strip())]

#             # ✅ Apply alias mapping
#             corrected = [PACKAGE_ALIASES.get(pkg, pkg) for pkg in found]
#             all_packages.update(corrected)

#         if not all_packages:
#             return "✅ No missing dependencies detected."

#         install_input = {
#             "packages": sorted(list(all_packages)),
#             "pip_path": str(pip_path)
#         }

#         result = install_dependencies.invoke({"arg": install_input})
#         return result if isinstance(result, str) else str(result)

#     except Exception as e:
#         return f"❌ Error during analysis or installation: {str(e)}"

@tool
def analyze_and_install_imports(arg: str) -> str:
    """
    Analyze Python and related files in a repo to detect third-party pip packages using LLM,
    and install them into the specified virtual environment. Auto-creates the environment if missing.

    Input format: 'repo_abs_path, venv_abs_path'
    """
    try:
        parts = [x.strip() for x in arg.split(',')]
        if len(parts) != 2:
            return "❌ Invalid input. Format must be: 'repo_abs_path, venv_abs_path'"

        repo_path_str, env_path_str = parts

        # 🚫 Check if pip path is passed instead of env path
        if env_path_str.endswith("pip") or "bin/pip" in env_path_str or "Scripts\\pip.exe" in env_path_str:
            return "❌ Invalid input: You passed a pip binary path. Please provide the virtual environment directory path (not the pip executable)."

        repo_path = Path(repo_path_str).resolve()
        if not repo_path.exists():
            repo_path = REPOS_DIR / repo_path.name

        env_path = Path(env_path_str).resolve()
        if not str(env_path).startswith(str(VENV_DIR)):
            env_path = VENV_DIR / env_path.name

        if not env_path.exists():
            create_virtualenv.invoke(env_path.name)

        pip_path = ensure_pip_in_env.invoke(str(env_path))
        subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        valid_exts = ('.py', '.ipynb', '.txt', '.md', 'requirements.txt')
        relevant_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(repo_path)
            for file in files if file.endswith(valid_exts)
        ]

        if not relevant_files:
            return "✅ No relevant source files found in the repository."

        # ✅ Alias corrections for common pip package mistakes
        PACKAGE_ALIASES = {
            "sklearn": "scikit-learn",
            "scikitlearn": "scikit-learn",
            "cv2": "opencv-python",
            "bs4": "beautifulsoup4",
            "pil": "pillow",
            "yaml": "pyyaml",
            "crypto": "pycryptodome",
            "tensorflow-gpu": "tensorflow",
            "torchvision": "torchvision",
            "tflearn": "tflearn",
            "keras": "keras",
            "flask": "flask",
            "django": "django",
            "requests": "requests",
            "nltk": "nltk",
            "pytz": "pytz",
            "pymongo": "pymongo",
        }

        all_packages = set()
        for file_path in relevant_files:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            prompt = f"""
Extract a comma-separated list of pip-installable Python package names based on the following code or documentation.

You MUST convert import/module names to their correct pip package names.

Examples:
- 'sklearn' → 'scikit-learn'
- 'scikitlearn' → 'scikit-learn'
- 'cv2' → 'opencv-python'
- 'bs4' → 'beautifulsoup4'
- 'PIL' → 'pillow'
- 'yaml' → 'pyyaml'

Only return the pip package names. No explanations.

Content:
{content}
"""
            chain = ChatPromptTemplate.from_template("{text}") | llm | StrOutputParser()
            response = chain.invoke({"text": prompt})
            response = response.replace('\n', ',').replace('-', '').replace('•', '').replace('*', '')
            found = [p.strip().lower() for p in response.split(',') if re.match(r"^[a-z0-9_.\\-]+$", p.strip())]

            # ✅ Apply alias mapping
            corrected = [PACKAGE_ALIASES.get(pkg, pkg) for pkg in found]
            all_packages.update(corrected)

        if not all_packages:
            return "✅ No missing dependencies detected."

        install_input = {
            "packages": sorted(list(all_packages)),
            "pip_path": str(pip_path)
        }

        # ✅ invoke your install_dependencies tool with LLM-based retry logic
        result = install_dependencies.invoke({"arg": install_input})
        return result if isinstance(result, str) else str(result)

    except Exception as e:
        return f"❌ Error during analysis or installation: {str(e)}"

@tool
def open_editor(path: str) -> str:
    """
    Open the provided path in Visual Studio Code if the path exists and VS Code CLI is available.
    Returns a confirmation or error message.
    """
    try:
        path = Path(path).resolve()
        if not path.exists():
            return f"❌ The provided path does not exist: {path}"
        if shutil.which("code") is None:
            return "❌ VS Code CLI 'code' is not available in PATH. Please install or set it up."
        subprocess.Popen(["code", str(path)])
        return f"✅ VS Code opened at: {path}"
    except Exception as e:
        return f"❌ Failed to open VS Code: {str(e)}"

@tool
def summarize_readme(repo_path: str) -> str:
    """
    Summarize the README.md (or readme.md) file using the LLM.
    Handles case-insensitive filenames and large files gracefully.
    """
    try:
        repo_path = Path(repo_path).resolve()
        readme_path = repo_path / "README.md"
        if not readme_path.exists():
            readme_path = repo_path / "readme.md"
        if not readme_path.exists():
            return "❌ README.md or readme.md not found in the repository."

        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        max_length = 4000
        if len(content) > max_length:
            content = content[:max_length] + "\n\n... (truncated)"

        prompt = f"Summarize the following README:\n\n{content}"
        response = llm.invoke(prompt)
        summary = response.content.strip()

        return f"📄 README Summary:\n{summary}"

    except Exception as e:
        return f"❌ Failed to summarize README: {str(e)}"
