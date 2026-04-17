import streamlit as st
import tempfile
import os
import pandas as pd
from pathlib import Path
from pipeline import run_pipeline

st.set_page_config(page_title="PhD Candidate Screener", page_icon="", layout="wide")


st.markdown('<div class="main-header"> PhD Candidate Screener</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔑 Groq API Key")
    groq_key = st.text_input("Enter your Groq API Key", type="password", placeholder="gsk_...")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("✅ API Key set")
    else:
        st.warning("⚠️ Add your free key at console.groq.com")
  
st.markdown("### 📂 Upload PhD Resumes")
uploaded_files = st.file_uploader(
    "Upload one or more PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload PDF files of PhD candidates"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")

    cols = st.columns(len(uploaded_files) if len(uploaded_files) <= 4 else 4)
    for i, f in enumerate(uploaded_files[:4]):
        with cols[i]:
            st.markdown(f"**📄 {f.name}**")
            st.caption(f"{round(f.size/1024, 1)} KB")

run_btn = st.button("🚀 Run Pipeline", type="primary", disabled=not uploaded_files, use_container_width=True)

if run_btn and uploaded_files:
    st.markdown("---")
    st.markdown("### ⏳ Processing...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    tmp_paths = []
    tmp_dir = tempfile.mkdtemp()

    for uploaded_file in uploaded_files:
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        tmp_paths.append(Path(tmp_path))

    def update_progress(msg, current, total):
        progress_bar.progress((current + 1) / total)
        status_text.markdown(f"**{msg}**")

    with st.spinner("Running AI pipeline..."):
        selected_df, rejected_df = run_pipeline(tmp_paths, progress_callback=update_progress)

    progress_bar.progress(1.0)
    status_text.markdown("**✅ Pipeline complete!**")

    st.markdown("---")
    st.markdown("### 📊 Results Summary")

    total = len(selected_df) + len(rejected_df)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Processed", total)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("✅ Selected", len(selected_df))
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("❌ Rejected", len(rejected_df))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2 = st.tabs(["✅ Selected Candidates", "❌ Rejected Candidates"])

    with tab1:
        if not selected_df.empty:
            st.markdown(f"**{len(selected_df)} candidate(s) selected**")

            def style_selected(df):
                return df.style.set_properties(**{
                    'background-color': '#f0fff4',
                    'border': '1px solid #c3e6cb'
                }).set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#28a745'), ('color', 'white'), ('font-weight', 'bold')]}
                ])

            st.dataframe(selected_df, use_container_width=True, hide_index=True)

            csv = selected_df.to_csv(index=False)
        else:
            st.info("No candidates were selected.")

    with tab2:
        if not rejected_df.empty:
            st.markdown(f"**{len(rejected_df)} candidate(s) rejected**")
            st.dataframe(rejected_df, use_container_width=True, hide_index=True)

            csv = rejected_df.to_csv(index=False)
            st.download_button("⬇️ Download Rejected CSV", csv, "rejected_candidates.csv", "text/csv")
        else:
            st.info("No candidates were rejected.")

    combined_df = pd.concat([selected_df, rejected_df], ignore_index=True)
    csv_all = combined_df.to_csv(index=False)

    for tmp_path in tmp_paths:
        try:
            os.remove(tmp_path)
        except:
            pass