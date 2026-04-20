import time
import requests
import pandas as pd
import json
import os
import fitz
import base64
import streamlit as st
from groq import Groq

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def pdf_to_base64(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[0].get_pixmap(dpi=150)
    img_bytes = pix.tobytes("jpeg")
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")

def extract_resume_info(base64_image):
    prompt = """Analyze this resume. Return ONLY a valid JSON object with these exact keys:
    "name": "Candidate's full name",
    "has_iit_nit": true/false (boolean, if they attended an IIT or NIT),
    "college_name": "The exact name of the IIT/NIT, or 'None'"
    """
    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    )
    return json.loads(response.choices[0].message.content)

def is_q1_journal(journal_name):
    if not journal_name or len(journal_name) < 3:
        return False
    url = f"https://www.scimagojr.com/journalsearch.php?q={requests.utils.quote(journal_name)}&tip=title&clean=0"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
    return "Q1" in resp.text

def fetch_q1_papers(author_name):
    clean_name = author_name.lower().replace("dr.", "").replace("prof.", "").strip()
    url = f"https://api.semanticscholar.org/graph/v1/author/search?query={clean_name}&fields=papers.venue&limit=1"
    res = requests.get(url, timeout=5).json()
    if not res.get("data"):
        return 0
    papers = res["data"][0].get("papers", [])
    return sum(1 for p in papers if is_q1_journal(p.get("venue")))

st.title("Resume Screener")

uploaded_files = st.file_uploader("Upload up to 10 resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Run"):
    results = []
    for uploaded_file in uploaded_files[:10]:
        img_b64 = pdf_to_base64(uploaded_file.read())
        info = extract_resume_info(img_b64)
        name = info.get("name", "Unknown")
        college_found = info.get("has_iit_nit", False)
        college_name = info.get("college_name", "None")
        q1_count = fetch_q1_papers(name)
        time.sleep(1)
        results.append({
            "Candidate Name": name,
            "College": college_name,
            "Q1 Papers": q1_count,
            "Decision": "SELECTED" if college_found else "REJECTED",
            "Reason": f"Attended {college_name}." if college_found else "Did not attend an IIT/NIT.",
        })
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
