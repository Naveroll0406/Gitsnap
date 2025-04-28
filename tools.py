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
env = "/root/dev/Naveen/Git_Snap/.env"
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
    Clone a GitHub repository from the provided URL into a local 'downloaded_repos' directory.

    Args:
        repo_url (str): GitHub repository URL.

    Returns:
        str: Absolute path of the cloned repository or an error message.
    """
    repo_url = repo_url.strip().replace("—", "-").rstrip("/")
    if "/tree/" in repo_url or "/blob/" in repo_url:
        return "❌ Cannot clone a subdirectory or file. Provide full repo URL."

    try:
        match = re.search(r"github\.com/([^/]+/[^/]+)", repo_url)
        if not match:
            return "❌ Invalid GitHub URL."
        repo_name = match.group(1).split('/')[-1]
    except Exception:
        return "❌ Failed to parse repo name."

    local_path = REPOS_DIR / repo_name
    if local_path.exists() and any(local_path.iterdir()):
        shutil.rmtree(local_path)

    try:
        Repo.clone_from(repo_url, str(local_path))
    except Exception as e:
        return f"❌ Git clone failed: {str(e)}"

    return str(local_path.resolve())


# @tool
# def analyze_repo(repo_path: str) -> str:
#     """
#     Analyze important files inside a repo and create a summarized 'repo_summary.txt' file
#     containing dependency files and all import/from statements from Python files.

#     Args:
#         repo_path (str): Absolute or relative path of the repo to analyze.

#     Returns:
#         str: Confirmation message with path of created summary file, or error message.
#     """
#     try:
#         repo_path = Path(repo_path).resolve()
#         summary_path = repo_path / "repo_summary.txt"

#         content = []

#         # ✅ Step 1: Add standard project files
#         files_to_scan = ["requirements.txt", "setup.py", "pyproject.toml", "environment.yml", "README.md", "readme.md"]
#         for filename in files_to_scan:
#             file = repo_path / filename
#             if file.exists():
#                 with open(file, "r", encoding="utf-8", errors="ignore") as f:
#                     content.append(f"\n# {filename}\n{f.read()}")

#         # ✅ Step 2: Add import/from lines from all .py files
#         for py_file in repo_path.rglob("*.py"):
#             if py_file.is_file():
#                 try:
#                     with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
#                         lines = f.readlines()

#                     import_lines = [line for line in lines if line.strip().startswith(("import ", "from "))]
#                     if import_lines:
#                         content.append(f"\n# {py_file.relative_to(repo_path)}\n" + "".join(import_lines))

#                 except Exception as e:
#                     content.append(f"\n# Failed to read {py_file.relative_to(repo_path)}: {str(e)}\n")

#         # ✅ Step 3: Merge all collected content (no truncation now)
#         summary = "\n".join(content)

#         # ✅ Step 4: Save to repo_summary.txt (full content)
#         with open(summary_path, "w", encoding="utf-8") as f:
#             f.write(summary)

#         return f"✅ Repo summary created at {summary_path}"

#     except Exception as e:
#         return f"❌ Failed to create repo_summary.txt: {str(e)}"

@tool
def analyze_repo(repo_path: str) -> str:
    """
    Analyze important files inside a repo and create a summarized 'repo_summary.txt' file,
    containing full dependency files, text/yaml/md files, and import/from statements from Python files.

    Args:
        repo_path (str): Absolute or relative path of the repo to analyze.

    Returns:
        str: Confirmation message with path of created summary file, or error message.
    """
    try:
        repo_path = Path(repo_path).resolve()
        summary_path = repo_path / "repo_summary.txt"

        content = []

        # ✅ Define important files (case-insensitive matching)
        dependency_files = {"requirements.txt", "setup.py", "pyproject.toml", "environment.yml", "README.md", "readme.md"}
        extra_full_files = {".txt", ".yml", ".yaml", ".yoml", ".md"}

        for file in repo_path.rglob("*"):
            if file.is_file():
                ext = file.suffix.lower()
                name = file.name.lower()

                try:
                    # ✅ 1. Dependency files (requirements, setup, etc.)
                    if name in dependency_files:
                        with open(file, "r", encoding="utf-8", errors="ignore") as f:
                            content.append(f"\n# {file.relative_to(repo_path)}\n{f.read()}")

                    # ✅ 2. Python files (.py) - Only import/from lines
                    elif ext == ".py":
                        with open(file, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                        import_lines = [line for line in lines if line.strip().startswith(("import ", "from "))]
                        if import_lines:
                            content.append(f"\n# {file.relative_to(repo_path)}\n" + "".join(import_lines))

                    # # ✅ 3. Text, YAML, Markdown files - Full content
                    elif ext in extra_full_files:
                        with open(file, "r", encoding="utf-8", errors="ignore") as f:
                            content.append(f"\n# {file.relative_to(repo_path)}\n{f.read()}")

                    # ✅ 4. Other files - Ignore
                    else:
                        continue

                except Exception as inner_e:
                    # If any single file fails, log it inside the summary (optional)
                    content.append(f"\n# Failed to read {file.relative_to(repo_path)}: {str(inner_e)}\n")

        # ✅ Merge all collected content
        summary = "\n".join(content)

        # ✅ Save to repo_summary.txt
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        return f"✅ Repo summary created at {summary_path}"

    except Exception as e:
        return f"❌ Failed to create repo_summary.txt: {str(e)}"

    
@tool
def detect_dependencies(repo_path: str) -> str:
    """
    Detect standard dependency files like requirements.txt, setup.py, etc. inside the repository.

    Args:
        repo_path (str): Path of the repository.

    Returns:
        str: Comma-separated list of found dependency files or error message.
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
    dependency_files_lower = {f.lower() for f in dependency_files}  # ✅ Precompute lowercase set once

    found_files = []
    try:
        for item in os.listdir(repo_path):
            if not item.startswith('.') and item.lower() in dependency_files_lower:
                full_path = (repo_path / item).resolve()
                if full_path.is_file():
                    found_files.append(item)
        return ", ".join(found_files) if found_files else "No dependency files found."
    except PermissionError:
        return "❌ Permission denied accessing repository path."
    except Exception as e:
        return f"❌ Error detecting dependencies: {str(e)}"
    

# python is checked this code and its useful
@tool
def create_virtualenv(env_name: str) -> str:
    """
    Create a Python virtual environment with the given environment name.

    Args:
        env_name (str): Name of the virtual environment.

    Returns:
        str: Absolute path to the created virtual environment or error message.
    """
    env_path = VENV_DIR / env_name
    if env_path.exists():
        shutil.rmtree(env_path, ignore_errors=True)

    # ✅ Check if Python is available
    python_executable = shutil.which("python3") or shutil.which("python")
    if not python_executable:
        return "❌ No Python interpreter (python3 or python) found on the system. Cannot create virtual environment."

    # ✅ Prepare the subprocess command separately
    venv_command = [python_executable, "-m", "venv", str(env_path)]
    print(f"📦 Creating virtualenv with command: {' '.join(venv_command)}")

    try:
        subprocess.run(
            venv_command,
            check=True,
            capture_output=True,
            text=True
        )
        return str(env_path.resolve())
    except subprocess.CalledProcessError as e:
        return f"❌ Failed to create virtual environment: {e.stderr}"
    except Exception as e:
        return f"❌ Unexpected error creating virtual environment: {str(e)}"


@tool
def ensure_pip_in_env(env_path: str) -> str:
    """
    Ensure that pip is installed in the specified Python virtual environment.
    
    Args:
        env_path (str): Path to the virtual environment.
    
    Returns:
        str: Absolute path to the pip executable or raises an error if installation fails.
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
        print("🔧 Trying to install pip using ensurepip...")
        subprocess.run([str(python_bin), "-m", "ensurepip", "--upgrade"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        try:
            print("⚡ ensurepip failed. Falling back to downloading get-pip.py...")
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


@tool
def install_dependencies(arg: Union[str, dict]) -> str:
    """
    Install Python project dependencies into the virtual environment.

    Supports two modes:
    - Direct install: Pass dictionary with packages list and pip path.
    - File-based install: Pass string with repo_path, pip_path, and dependency file names.
    """
    try:
        # ✅ Direct install mode
        if isinstance(arg, dict) and "packages" in arg:
            packages = arg.get("packages", [])
            print(f"🔍 Installing packages: {', '.join(packages)}")

            pip_path = Path(arg.get("pip_path", "").strip())

            if not pip_path.is_absolute() or not pip_path.exists():
                pip_path = VENV_DIR / pip_path.name / ("bin/pip" if os.name != "nt" else "Scripts/pip.exe")

            pip_path = pip_path.resolve()

            if not packages:
                return "✅ No packages to install."
            if not pip_path.exists():
                return f"❌ pip not found at {pip_path}"

            install_logs = []
            for package in packages:
                try:
                    subprocess.run(
                        [str(pip_path), "install", package],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    install_logs.append(f"✅ Successfully installed: {package}")
                except subprocess.CalledProcessError:
                    retry = subprocess.run(
                        [str(pip_path), "install", "--prefer-binary", "--no-build-isolation", package],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    if retry.returncode == 0:
                        install_logs.append(f"✅ Retried with binary and succeeded: {package}")
                    else:
                        install_logs.append(f"❌ Failed to install {package}: {retry.stderr}")

            return "\n".join(install_logs)

        # ✅ File-based input mode
        elif isinstance(arg, str):
            # ⚡ No change needed for file-based mode
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
                        try:
                            subprocess.run([str(pip_path), 'install', '-r', str(file_path)], check=True)
                            messages.append("✅ Installed from requirements.txt")
                        except subprocess.CalledProcessError as err:
                            messages.append(f"⚠️ Installation from requirements.txt failed. Trying fallback...")

                            with open(file_path, "r") as f:
                                raw_lines = f.read().splitlines()

                            packages = []
                            for line in raw_lines:
                                line = line.strip()
                                if not line or line.startswith("#"):
                                    continue
                                parts = re.split(r'\s+', line)
                                packages.extend(parts)

                            for pkg in packages:
                                pkg = pkg.strip()
                                if not pkg:
                                    continue
                                try:
                                    subprocess.run([str(pip_path), "install", pkg], check=True)
                                except subprocess.CalledProcessError as e1:
                                    pkg_name = re.split(r'[=<>!~]', pkg)[0]
                                    messages.append(f"🔄 Retrying {pkg_name} with latest version...")
                                    try:
                                        subprocess.run([str(pip_path), "install", pkg_name], check=True)
                                        messages.append(f"✅ Installed latest version of {pkg_name}")
                                    except subprocess.CalledProcessError as e2:
                                        messages.append(f"❌ Failed to install {pkg_name}: {e2.stderr}")

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
                        messages.append("ℹ️ README found. Consider reviewing it manually.")

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


    
# @tool
# def analyze_and_install_imports(arg: str) -> str:
#     """
#     Analyze the 'repo_summary.txt' file of the repository to detect missing third-party pip packages using LLM,
#     and install them into the specified virtual environment.

#     Input format: 'repo_abs_path, venv_abs_path'

#     Args:
#         arg (str): Input string with repo path and venv path separated by comma.

#     Returns:
#         str: Result message after analyzing and installing packages.
#     """
#     try:
#         parts = [x.strip() for x in arg.split(',')]
#         if len(parts) != 2:
#             return "❌ Invalid input. Format must be: 'repo_abs_path, venv_abs_path'"

#         repo_path_str, env_path_str = parts

#         repo_path = Path(repo_path_str).resolve()
#         if not repo_path.exists():
#             repo_path = REPOS_DIR / repo_path.name

#         env_path = Path(env_path_str).resolve()
#         if not str(env_path).startswith(str(VENV_DIR)):
#             env_path = VENV_DIR / env_path.name

#         if not env_path.exists():
#             create_virtualenv.invoke(env_path.name)

#         pip_path = ensure_pip_in_env.invoke(str(env_path))
#         if not Path(pip_path).exists():
#             return f"❌ pip executable not found at: {pip_path}"

#         subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         summary_file = repo_path / "repo_summary.txt"
#         if not summary_file.exists():
#             return "❌ repo_summary.txt not found. Please ensure analyze_repo tool ran successfully."

#         with open(summary_file, "r", encoding="utf-8", errors="ignore") as f:
#             content = f.read()

#         if not content.strip():
#             return "✅ Repo summary is empty, no analysis needed."

#         def chunk_text(text: str, max_chars: int = 30000) -> list:
#             return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

#         content_chunks = chunk_text(content)

#         prompt_template = """
# Extract pip-installable package names from the following project content.

# If a library requires another backend library to work (e.g., keras needs tensorflow), automatically include the backend library also.

# Return the output in two comma-separated lists:
# 1. Packages that must be installed first (for dependency reasons like numpy before scikit-learn).
# 2. Packages that can be installed after.

# Format:
# install_first: pkg1, pkg2, ...
# install_after: pkg3, pkg4, ...

# Ignore standard libraries. Only return clean, lowercase pip names.

# Do not return 'faiss'. Use 'faiss-cpu' if needed and only if installable via pip and similarlly to all the libraries ok.


# Content:
# {text}
# """
#         all_found = []
#         chain = ChatPromptTemplate.from_template("{text}") | llm | StrOutputParser()
#         for chunk in content_chunks:
#             prompt = prompt_template.format(text=chunk)
#             raw = chain.invoke({"text": prompt})
#             raw = raw.replace('\n', ',').replace('-', '').replace('•', '').replace('*', '')
#             all_found += [p.strip().lower() for p in raw.split(',') if re.match(r"^[a-z0-9_.\\-]+$", p.strip())]

#         PACKAGE_ALIASES = {
#             "sklearn": "scikit-learn",
#             "scikitlearn": "scikit-learn",
#             "tensorflow-gpu": "tensorflow",
#             "tflearn": "tflearn tensorflow",
#             "torchvision": "torchvision torch",
#             "torch": "torch",
#             "keras": "keras tensorflow",
#             "xgboost": "xgboost",
#             "lightgbm": "lightgbm",
#             "catboost": "catboost",
#             "cv2": "opencv-python",
#             "pillow": "pillow",
#             "pil": "pillow",
#             "bs4": "beautifulsoup4",
#             "yaml": "pyyaml",
#             "html5lib": "html5lib",
#             "lxml": "lxml",
#             "nltk": "nltk",
#             "spacy": "spacy",
#             "gensim": "gensim",
#             "transformers": "transformers",
#             "flask": "flask",
#             "django": "django",
#             "fastapi": "fastapi",
#             "requests": "requests",
#             "urllib3": "urllib3",
#             "aiohttp": "aiohttp",
#             "crypto": "pycryptodome",
#             "cryptography": "cryptography",
#             "pymongo": "pymongo",
#             "sqlalchemy": "sqlalchemy",
#             "psycopg2": "psycopg2",
#             "mysqlclient": "mysqlclient",
#             "pytest": "pytest",
#             "unittest": "unittest2",
#             "matplotlib": "matplotlib",
#             "seaborn": "seaborn",
#             "plotly": "plotly",
#         }

#         STANDARD_LIBRARIES = {
#             "os", "sys", "pathlib", "subprocess", "shutil", "logging",
#             "threading", "multiprocessing", "asyncio", "time", "datetime",
#             "functools", "argparse", "collections", "itertools", "copy",
#             "pprint", "inspect", "uuid", "traceback", "math", "statistics",
#             "decimal", "fractions", "random", "numbers", "hashlib", "base64",
#             "hmac", "secrets", "json", "csv", "xml", "yaml", "html", "configparser",
#             "pickle", "marshal", "socket", "http", "urllib", "email", "ftplib",
#             "smtplib", "xmlrpc", "html.parser", "sqlite3", "dbm", "shelve",
#             "unittest", "doctest", "pytest", "warnings", "queue", "weakref",
#             "types", "enum", "contextlib", "typing",
#         }

#         corrected = []
#         for pkg in all_found:
#             expanded = PACKAGE_ALIASES.get(pkg, pkg)
#             corrected.extend(expanded.split())

#         corrected = [pkg for pkg in corrected if pkg not in STANDARD_LIBRARIES]
#         unique_packages = sorted(set(corrected))

#         if not unique_packages:
#             return "✅ No missing dependencies detected."

#         install_input = {
#             "packages": unique_packages,
#             "pip_path": str(pip_path)
#         }

#         messages = []
#         for pkg in unique_packages:
#             try:
#                 subprocess.run([str(pip_path), "install", pkg], check=True)
#                 messages.append(f"✅ Installed: {pkg}")
#             except subprocess.CalledProcessError as e:
#                 messages.append(f"❌ Failed to install {pkg}: {e}")

                
#         return "\n".join(messages)

#     except Exception as e:
#         return f"❌ Error during analysis or installation: {str(e)}"



 
@tool
def analyze_and_install_imports(arg: str) -> str:
    """
    Analyze the 'repo_summary.txt' file of the repository to detect missing third-party pip packages using LLM,
    and install them into the specified virtual environment.

    Input format: 'repo_abs_path, venv_abs_path'
    """
    try:
        parts = [x.strip() for x in arg.split(',')]
        if len(parts) != 2:
            return "❌ Invalid input. Format must be: 'repo_abs_path, venv_abs_path'"

        repo_path_str, env_path_str = parts

        repo_path = Path(repo_path_str).resolve()
        if not repo_path.exists():
            repo_path = REPOS_DIR / repo_path.name

        env_path = Path(env_path_str).resolve()
        if not str(env_path).startswith(str(VENV_DIR)):
            env_path = VENV_DIR / env_path.name

        if not env_path.exists():
            create_virtualenv.invoke(env_path.name)

        pip_path = ensure_pip_in_env.invoke(str(env_path))
        if not Path(pip_path).exists():
            return f"❌ pip executable not found at: {pip_path}"

        subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        summary_file = repo_path / "repo_summary.txt"
        if not summary_file.exists():
            return "❌ repo_summary.txt not found. Please ensure analyze_repo tool ran successfully."

        with open(summary_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            return "✅ Repo summary is empty, no analysis needed."

        def chunk_text(text: str, max_chars: int = 30000) -> list:
            return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

        content_chunks = chunk_text(content)

        prompt_template = """
            Extract pip-installable package names from the following project content.

            If a library requires another backend library to work (e.g., keras needs tensorflow), automatically include the backend library also.

            Return the output in two comma-separated lists:
            1. Packages that must be installed first (for dependency reasons like numpy before scikit-learn).
            2. Packages that can be installed after.

            Format:
            install_first: pkg1, pkg2, ...
            install_after: pkg3, pkg4, ...

            Ignore standard libraries. Only return clean, lowercase pip names.
            Do not return 'faiss'. Use 'faiss-cpu' if needed and only if installable via pip and similarly to all the libraries ok.

            If a library is known to be installable only via GitHub, ignore it here (leave to developer).

            Content:
            {text}
            """

        all_found = []
        
        chain = ChatPromptTemplate.from_template("{text}") | llm | StrOutputParser()
        for chunk in content_chunks:
            prompt = prompt_template.format(text=chunk)
            raw = chain.invoke({"text": prompt})
            raw = raw.replace('\n', ',').replace('-', '').replace('•', '').replace('*', '')
            all_found += [p.strip().lower() for p in raw.split(',') if re.match(r"^[a-z0-9_.\\-]+$", p.strip())]

        PACKAGE_ALIASES = {
            "sklearn": "scikit-learn",
            "scikitlearn": "scikit-learn",
            "tensorflow-gpu": "tensorflow",
            "tflearn": "tflearn tensorflow",
            "torchvision": "torchvision torch",
            "torch": "torch",
            "keras": "keras tensorflow",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "catboost": "catboost",
            "cv2": "opencv-python",
            "pillow": "pillow",
            "pil": "pillow",
            "bs4": "beautifulsoup4",
            "yaml": "pyyaml",
            "html5lib": "html5lib",
            "lxml": "lxml",
            "nltk": "nltk",
            "spacy": "spacy",
            "gensim": "gensim",
            "transformers": "transformers",
            "flask": "flask",
            "django": "django",
            "fastapi": "fastapi",
            "requests": "requests",
            "urllib3": "urllib3",
            "aiohttp": "aiohttp",
            "crypto": "pycryptodome",
            "cryptography": "cryptography",
            "pymongo": "pymongo",
            "sqlalchemy": "sqlalchemy",
            "psycopg2": "psycopg2",
            "mysqlclient": "mysqlclient",
            "pytest": "pytest",
            "unittest": "unittest2",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "plotly": "plotly",
        }

        STANDARD_LIBRARIES = {
            "os", "sys", "pathlib", "subprocess", "shutil", "logging",
            "threading", "multiprocessing", "asyncio", "time", "datetime",
            "functools", "argparse", "collections", "itertools", "copy",
            "pprint", "inspect", "uuid", "traceback", "math", "statistics",
            "decimal", "fractions", "random", "numbers", "hashlib", "base64",
            "hmac", "secrets", "json", "csv", "xml", "yaml", "html", "configparser",
            "pickle", "marshal", "socket", "http", "urllib", "email", "ftplib",
            "smtplib", "xmlrpc", "html.parser", "sqlite3", "dbm", "shelve",
            "unittest", "doctest", "pytest", "warnings", "queue", "weakref",
            "types", "enum", "contextlib", "typing",
        }

        corrected = []
        for pkg in all_found:
            expanded = PACKAGE_ALIASES.get(pkg, pkg)
            corrected.extend(expanded.split())

        corrected = [pkg for pkg in corrected if pkg not in STANDARD_LIBRARIES]
        unique_packages = sorted(set(corrected))

        if not unique_packages:
            return "✅ No missing dependencies detected."

        messages = []
        failed_pkgs = []

        for pkg in unique_packages:
            try:
                subprocess.run([str(pip_path), "install", pkg], check=True)
                messages.append(f"✅ Installed: {pkg}")
            except subprocess.CalledProcessError:
                retry_cmd = [str(pip_path), "install", "--prefer-binary", "--no-build-isolation", pkg]
                retry = subprocess.run(retry_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if retry.returncode == 0:
                    messages.append(f"✅ Retried with binary: {pkg}")
                else:
                    messages.append(f"❌ Failed to install {pkg}")
                    failed_pkgs.append(pkg)

        for failed in failed_pkgs:
            found = False
            for py_file in repo_path.rglob("*.py"):
                try:
                    with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                        file_content = f.read().lower()
                        if f"import {failed}" in file_content or f"from {failed}" in file_content:
                            found = True
                            break
                except:
                    continue
            if found:
                messages.append(f"⚠️ Package '{failed}' is used in the repo → likely required.")
            else:
                messages.append(f"ℹ️ Package '{failed}' not found in code → might be optional.")

        return "\n".join(messages)

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
