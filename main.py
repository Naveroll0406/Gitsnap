import streamlit as st
from agent import run_agent_on_repo
from pathlib import Path

# Page setup
st.set_page_config(page_title="GitHub Repository Setup Agent", page_icon="💻", layout="centered")

# Simple styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 6px;
        padding: 8px 16px;
        border: none;
    }
    .stTextInput > div > div > input {
        padding: 8px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("💻 GitHub Repository Setup Agent")
st.caption("Automatically clone, set up, and summarize any public GitHub repo for local development.")

# Input field
repo_url = st.text_input("Enter GitHub Repo URL", placeholder="https://github.com/user/repo")

# Run agent
if st.button("🚀 Run Setup"):
    if not repo_url or not repo_url.startswith("https://github.com/"):
        st.warning("Please enter a valid GitHub URL.")
    else:
        with st.spinner("Running AI agent setup... ⏳"):
            try:
                result = run_agent_on_repo(repo_url)

                # ✅ Just show success message after setup
                st.success("✅ Setup completed successfully!")

            except Exception as e:
                st.error("❌ Something went wrong during setup.")
                st.exception(e)
