import time
import requests
import pandas as pd
from rapidfuzz import fuzz
from groq import Groq
import os
import fitz
from PIL import Image
import base64
import io

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"


def pdf_page_to_base64(pdf_path, page_index=0, dpi=200):
    doc = fitz.open(str(pdf_path))
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def groq_vision_call(base64_image, prompt):
    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    return response.choices[0].message.content.strip()


def extract_name(base64_image):
    prompt = (
        "Look at this resume image. "
        "Extract ONLY the full name of the candidate. "
        "Return just the name with no extra text or punctuation."
    )
    return groq_vision_call(base64_image, prompt)


def extract_education(base64_image):
    prompt = (
        "Look at this resume image. "
        "Find and return ONLY the section about educational qualifications "
        "(degrees, universities, institutes, years). "
        "Include every line from that section verbatim. "
        "Do not add commentary or headings — just the raw text from that section."
    )
    return groq_vision_call(base64_image, prompt)


def check_iit_nit(education_text):
    prompt = f"""From the education section below, identify if the candidate has studied 
at any IIT (Indian Institute of Technology) or NIT (National Institute of Technology).

Respond in EXACTLY this format (two lines, nothing else):
FOUND: YES or NO
COLLEGE: the exact college name as written in the text, or NONE

Education Section:
{education_text}"""

    response = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content.strip()

    found = False
    college_name = None
    for line in result.splitlines():
        if line.upper().startswith("FOUND:"):
            found = "YES" in line.upper()
        if line.upper().startswith("COLLEGE:") and "NONE" not in line.upper():
            college_name = line.split(":", 1)[1].strip()

    return found, college_name


def is_q1_journal(journal_name):
    if not journal_name or len(journal_name.strip()) < 3:
        return False
    try:
        url = f"https://www.scimagojr.com/journalsearch.php?q={requests.utils.quote(journal_name)}&tip=title&clean=0"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        return "Q1" in resp.text
    except Exception:
        return False


def clean_name(name):
    prefixes = ["dr", "mr", "mrs", "ms", "prof", "professor", "phd", "ph.d"]
    parts = name.strip().split()
    parts = [p for p in parts if p.lower().strip(".,") not in prefixes]
    return " ".join(parts)


def fetch_q1_papers(author_name):
    try:
        author_name = clean_name(author_name)

        search_url = "https://api.semanticscholar.org/graph/v1/author/search"
        params = {"query": author_name, "fields": "authorId,name,paperCount", "limit": 5}
        resp = requests.get(search_url, params=params, timeout=10)
        data = resp.json()

        if not data.get("data"):
            return 0, []

        best_match = None
        best_score = 0
        for candidate in data["data"]:
            score = fuzz.token_sort_ratio(author_name.lower(), candidate["name"].lower())
            if score > best_score:
                best_score = score
                best_match = candidate

        if not best_match or best_score < 60:
            return 0, []

        author_id = best_match["authorId"]
        papers_url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
        paper_params = {"fields": "title,venue,year,externalIds", "limit": 50}
        papers_resp = requests.get(papers_url, params=paper_params, timeout=10)
        papers_data = papers_resp.json()

        q1_papers = []
        for paper in papers_data.get("data", []):
            venue = paper.get("venue", "")
            if venue and is_q1_journal(venue):
                q1_papers.append({
                    "title": paper.get("title", ""),
                    "venue": venue,
                    "year": paper.get("year", ""),
                })
            time.sleep(0.5)

        return len(q1_papers), q1_papers

    except Exception:
        return 0, []


def analyze_candidate_with_llm(candidate_name, college_found, college_name, q1_count, q1_papers):
    papers_text = (
        "\n".join(f"- {p['title']} ({p['venue']}, {p['year']})" for p in q1_papers[:5])
        if q1_papers
        else "None found"
    )
    decision = "SELECTED" if college_found else "REJECTED"

    prompt = f"""You are evaluating a PhD candidate for a research position.

Candidate: {candidate_name}
College in approved list (IIT/NIT): {"Yes — " + college_name if college_found else "No"}
Q1 Research Papers Count: {q1_count}
Q1 Papers:
{papers_text}

The candidate has already been {decision} based solely on whether their college
appears in the approved list under their educational qualifications.
Write a clear 1-sentence reason reflecting this decision.
Do NOT mention Q1 paper count as a selection/rejection criterion.

Respond in EXACTLY this format:
DECISION: {decision}
REASON: your reason here"""

    response = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    result_text = response.choices[0].message.content.strip()
    reason = "Does not meet criteria."
    for line in result_text.splitlines():
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
            break

    return decision, reason


def run_pipeline(pdf_paths, progress_callback=None):
    selected = []
    rejected = []

    for i, pdf_path in enumerate(pdf_paths):
        label = pdf_path.name if hasattr(pdf_path, "name") else str(pdf_path)
        if progress_callback:
            progress_callback(f"Processing: {label}", i, len(pdf_paths))

        try:
            base64_image = pdf_page_to_base64(pdf_path, page_index=0, dpi=200)
            candidate_name = extract_name(base64_image)
            education_text = extract_education(base64_image)
            college_found, college_name = check_iit_nit(education_text)
            q1_count, q1_papers = fetch_q1_papers(candidate_name)
            time.sleep(1)
            decision, reason = analyze_candidate_with_llm(
                candidate_name, college_found, college_name, q1_count, q1_papers
            )

            record = {
                "Candidate Name": candidate_name,
                "College Found": college_name if college_found else "None",
                "Q1 Papers Count": q1_count,
                "Decision": decision,
                "Reason": reason,
            }
            (selected if decision == "SELECTED" else rejected).append(record)

        except Exception as e:
            rejected.append({
                "Candidate Name": label,
                "College Found": "Error",
                "Q1 Papers Count": 0,
                "Decision": "REJECTED",
                "Reason": f"Processing error: {str(e)}",
            })

    return pd.DataFrame(selected), pd.DataFrame(rejected)
