import json
import time
import easyocr
import requests
import pandas as pd
from pdf2image import convert_from_path
from rapidfuzz import fuzz
from groq import Groq
import os
import numpy as np
import re

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
reader = easyocr.Reader(['en'], gpu=False)  

def load_iit_names(json_path="iit_names.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["iit_names"]


def extract_education_section_from_page1(pdf_path):
    """
    Converts only PAGE 1 to image, runs EasyOCR,
    then extracts only the text between
    'Particulars of Educational Qualification' and the next major section.
    """
    images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
    page1_img = np.array(images[0])

    results = reader.readtext(page1_img, detail=1, paragraph=False)

    results_sorted = sorted(results, key=lambda r: r[0][0][1])  
    lines = [res[1] for res in results_sorted]
    full_text = "\n".join(lines)

    start_markers = [
        "particulars of educational qualification",
        "educational qualification",
        "academic qualification",
    ]
    end_markers = [
        "titles pg",
        "titles ph",
        "experience",
        "research experience",
        "teaching experience",
        "declaration",
    ]

    lower_text = full_text.lower()
    start_idx = -1
    for marker in start_markers:
        idx = lower_text.find(marker)
        if idx != -1:
            start_idx = idx
            break

    if start_idx == -1:
        return full_text

    end_idx = len(full_text)
    for marker in end_markers:
        idx = lower_text.find(marker, start_idx + 10)
        if idx != -1 and idx < end_idx:
            end_idx = idx

    education_block = full_text[start_idx:end_idx]
    return education_block


def extract_full_text_from_pdf(pdf_path):
    """Full text via EasyOCR across all pages (used for name extraction only)."""
    images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
    page1_img = np.array(images[0])
    results = reader.readtext(page1_img, detail=0, paragraph=True)
    return "\n".join(results)


def check_college_in_education_block(education_block, college_names):
    """
    Match college names ONLY within the extracted education section,
    never against the full document (avoids false positives from referee section).
    """
    block_lower = education_block.lower()
    for college in college_names:
        if college.lower() in block_lower:
            return True, college
        if fuzz.partial_ratio(college.lower(), block_lower) > 90:
            return True, college
    return False, None


def extract_candidate_name(resume_text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                "Extract only the full name of the candidate from this resume. "
                "Return just the name, nothing else.\n\n"
                f"{resume_text[:2000]}"
            )
        }]
    )
    return response.choices[0].message.content.strip()


def fetch_q1_papers(author_name):
    try:
        search_url = "https://api.semanticscholar.org/graph/v1/author/search"
        params = {"query": author_name, "fields": "authorId,name,paperCount", "limit": 1}
        resp = requests.get(search_url, params=params, timeout=10)
        data = resp.json()

        if not data.get("data"):
            return 0, []

        author_id = data["data"][0]["authorId"]
        papers_url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
        paper_params = {"fields": "title,venue,year,externalIds", "limit": 50}
        papers_resp = requests.get(papers_url, params=paper_params, timeout=10)
        papers_data = papers_resp.json()

        q1_venues = [
            "Nature", "Science", "Cell", "IEEE Transactions", "ACM",
            "NeurIPS", "ICML", "ICLR", "CVPR", "ECCV", "ICCV", "ACL",
            "EMNLP", "NAACL", "AAAI", "IJCAI", "KDD", "WWW", "SIGMOD",
            "VLDB", "ICDE", "Lancet", "NEJM", "JAMA", "Nature Medicine",
            "Nature Communications", "Advanced Materials", "Angewandte Chemie",
            "Physical Review Letters", "Journal of the American Chemical Society",
            "Bioinformatics", "Nucleic Acids Research", "PNAS"
        ]

        q1_papers = []
        for paper in papers_data.get("data", []):
            venue = paper.get("venue", "")
            if any(fuzz.partial_ratio(q1.lower(), venue.lower()) > 80 for q1 in q1_venues):
                q1_papers.append({
                    "title": paper.get("title", ""),
                    "venue": venue,
                    "year": paper.get("year", "")
                })

        return len(q1_papers), q1_papers

    except Exception:
        return 0, []


def analyze_candidate_with_llm(candidate_name, college_found, college_name, q1_count, q1_papers):
    papers_text = "\n".join(
        [f"- {p['title']} ({p['venue']}, {p['year']})" for p in q1_papers[:5]]
    ) if q1_papers else "None found"

    decision = "SELECTED" if college_found else "REJECTED"

    prompt = f"""You are evaluating a PhD candidate for a research position.

Candidate: {candidate_name}
College in approved list (from education section only): {"Yes - " + college_name if college_found else "No"}
Q1 Research Papers Count: {q1_count}
Q1 Papers:
{papers_text}

The candidate has already been {decision} based solely on whether their college
appears in the approved list under their educational qualifications.
Write a clear 1-sentence reason reflecting this decision.
Do NOT mention Q1 paper count as a selection/rejection criterion.

Respond in this exact format:
DECISION: {decision}
REASON: your reason here"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    result_text = response.choices[0].message.content.strip()
    reason = "Does not meet criteria."
    for line in result_text.split("\n"):
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
            break

    return decision, reason


def run_pipeline(pdf_paths, college_json_path="iit_names.json", progress_callback=None):
    college_names = load_iit_names(college_json_path)
    selected = []
    rejected = []

    for i, pdf_path in enumerate(pdf_paths):
        if progress_callback:
            progress_callback(
                f"Processing: {pdf_path.name if hasattr(pdf_path, 'name') else pdf_path}",
                i,
                len(pdf_paths)
            )

        try:
            education_block = extract_education_section_from_page1(pdf_path)

            page1_text = extract_full_text_from_pdf(pdf_path)
            candidate_name = extract_candidate_name(page1_text)

            college_found, college_name = check_college_in_education_block(
                education_block, college_names
            )

            q1_count, q1_papers = fetch_q1_papers(candidate_name)
            time.sleep(1)

            decision, reason = analyze_candidate_with_llm(
                candidate_name, college_found, college_name, q1_count, q1_papers
            )

            record = {
                "Candidate Name":  candidate_name,
                "College Found":   college_name if college_found else "None",
                "Q1 Papers Count": q1_count,
                "Decision":        decision,
                "Reason":          reason,
            }

            (selected if decision == "SELECTED" else rejected).append(record)

        except Exception as e:
            rejected.append({
                "Candidate Name":  str(pdf_path),
                "College Found":   "Error",
                "Q1 Papers Count": 0,
                "Decision":        "REJECTED",
                "Reason":          f"Processing error: {str(e)}",
            })

    return pd.DataFrame(selected), pd.DataFrame(rejected)
