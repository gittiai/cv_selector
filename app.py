import streamlit as st
import tempfile
import os
import pandas as pd
from pathlib import Path
from pipeline_gemini import run_pipeline

groq_key = st.secrets.get("GROQ_API_KEY", "")
if not groq_key:
    groq_key = st.text_input("Groq API Key", type="password")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

st.title("PhD Candidate Screener")

uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if st.button("Run Pipeline") and uploaded_files:
    tmp_dir = tempfile.mkdtemp()
    tmp_paths = []
    for f in uploaded_files:
        tmp_path = os.path.join(tmp_dir, f.name)
        with open(tmp_path, "wb") as out:
            out.write(f.read())
        tmp_paths.append(Path(tmp_path))

    progress_bar = st.progress(0)
    status = st.empty()

    def update_progress(msg, current, total):
        progress_bar.progress((current + 1) / total)
        status.text(msg)

    selected_df, rejected_df = run_pipeline(tmp_paths, progress_callback=update_progress)
    status.text("Done!")

    df = pd.concat([selected_df, rejected_df], ignore_index=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "results.csv", "text/csv")
