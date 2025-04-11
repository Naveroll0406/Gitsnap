import streamlit as st
from agent import run_agent_on_repo
import os
from pathlib import Path

# Set page configuration
st.set_page_config(page_title="AI Dev Setup Agent", page_icon="💻", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 0;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput > div > div > input {
        padding: 8px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🛠️ Auto Dev Setup Agent")
st.write("Enter a GitHub repository URL below, and the agent will automate the setup process for local development.")

# Input field for repo URL
repo_url = st.text_input("🔗 GitHub Repo URL", placeholder="https://github.com/user/repo", key="repo_url_input")

# Run button
if st.button("🚀 Run Agent"):
    if not repo_url:
        st.warning("Please enter a GitHub repository URL.")
    elif not repo_url.startswith("https://github.com/"):
        st.error("Please enter a valid GitHub URL that starts with https://github.com/")
    else:
        try:
            with st.status("🤖 Running setup steps...", expanded=True) as status:
                steps = [
                    "📦 Cloning the GitHub repository...",
                    "🧪 Detecting dependency files...",
                    "📁 Creating virtual environment...",
                    "🔄 Ensuring pip is installed...",
                    "🔧 Installing listed dependencies...",
                    "🕵️ Scanning .py and .ipynb files for missing libraries (via LLM)...",
                    "📦 Installing missing pip packages...",
                    "💻 Launching VS Code...",
                    "📄 Summarizing the entire repository..."
                ]

                for i, step in enumerate(steps):
                    with st.spinner(step):
                        st.write(step)
                        if i == 0:
                            st.write("⏳ This might take a moment...")

                result = run_agent_on_repo(repo_url)

                status.update(label="✅ Agent setup completed!", state="complete", expanded=True)
                st.success("Everything is set up! 🎉")

                output = result.get("output", str(result)) if isinstance(result, dict) else str(result)

                st.text_area(
                    "📋 Agent Output",
                    value=output,
                    height=400,
                    help="This is the full setup log from the agent.",
                    key="output_area"
                )

                st.download_button(
                    label="📥 Download Log",
                    data=output,
                    file_name=f"agent_output_{Path(repo_url).name}.txt",
                    mime="text/plain",
                    key="download_button"
                )

        except Exception as e:
            status.update(label="❌ Error during setup", state="error")
            st.error(f"An error occurred during setup: {str(e)}")
            st.exception(e)
