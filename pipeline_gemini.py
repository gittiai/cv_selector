import time
import requests
import pandas as pd
import json
import os
import fitz
import base64
from groq import Groq

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def pdf_to_base64(pdf_path):
    doc = fitz.open(str(pdf_path))
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
    q1_count = sum(1 for p in papers if is_q1_journal(p.get("venue")))
    
    return q1_count

def run_pipeline(pdf_paths, progress_callback=None):
    results = []

    for i, pdf_path in enumerate(pdf_paths):
        label = pdf_path.name if hasattr(pdf_path, "name") else str(pdf_path)
        if progress_callback:
            progress_callback(f"Processing: {label}", i, len(pdf_paths))

        try:
            img_b64 = pdf_to_base64(pdf_path)

            info = extract_resume_info(img_b64)
            name = info.get("name", "Unknown")
            college_found = info.get("has_iit_nit", False)
            college_name = info.get("college_name", "None")

            q1_count = fetch_q1_papers(name)
            time.sleep(1)

            decision = "SELECTED" if college_found else "REJECTED"
            reason = f"Candidate attended {college_name}." if college_found else "Did not attend an IIT/NIT."

            results.append({
                "Candidate Name": name,
                "College Found": college_name,
                "Q1 Papers Count": q1_count,
                "Decision": decision,
                "Reason": reason,
            })

        except Exception as e:
            results.append({
                "Candidate Name": label,
                "College Found": "Error",
                "Q1 Papers Count": 0,
                "Decision": "REJECTED",
                "Reason": f"Processing error: {str(e)}",
            })

    df = pd.DataFrame(results)
    selected_df = df[df["Decision"] == "SELECTED"].reset_index(drop=True)
    rejected_df = df[df["Decision"] != "SELECTED"].reset_index(drop=True)

    return selected_df, rejected_df
