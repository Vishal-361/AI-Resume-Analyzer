"""
app.py — Streamlit UI for the AI Resume Analyzer.

Run with:  streamlit run app.py

The interface lets a recruiter or applicant upload a resume (PDF or
TXT), paste or upload a job description, and view the match score,
skill gaps, keyword coverage and improvement suggestions produced by
resume_analyzer.py. All analysis logic lives in that module; this file
only handles input and presentation.
"""

from pathlib import Path

import streamlit as st

import resume_analyzer as ra

BASE_DIR = Path(__file__).parent
SAMPLE_RESUME_PATH = BASE_DIR / "samples" / "sample_resume.txt"
SAMPLE_JOB_PATH = BASE_DIR / "samples" / "sample_job_description.txt"


def load_sample(path):
    """Read one of the bundled sample text files."""
    return path.read_text(encoding="utf-8")


def score_verdict(score):
    """Short human-readable verdict for a match score.

    The bands are calibrated for raw TF-IDF cosine similarity: a resume
    and a job description are different kinds of documents, so even a
    good match rarely scores above ~60.
    """
    if score >= 55:
        return "Strong match — this resume speaks the job description's language."
    if score >= 35:
        return "Decent match — a few targeted edits should push it higher."
    if score >= 20:
        return "Partial match — significant gaps to close before applying."
    return "Weak match — the resume and job description barely overlap."


def show_results(result):
    """Render the dict returned by ra.analyze_resume()."""
    score = result["match_score"]
    skills = result["skills"]
    keywords = result["keywords"]

    st.divider()
    st.subheader("Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Match score", f"{score}%")
    col2.metric(
        "Skill coverage",
        f"{skills['coverage']}%",
        help=f"{len(skills['matched'])} of {len(skills['required'])} required skills found",
    )
    col3.metric("Keyword coverage", f"{keywords['coverage']}%")
    col4.metric("Resume length", f"{result['word_count']} words")
    st.progress(min(score / 100, 1.0))
    st.caption(score_verdict(score))

    # --- Skills ---------------------------------------------------------
    st.subheader("Skills")
    matched_col, missing_col = st.columns(2)
    with matched_col:
        st.markdown("**✅ Matched skills**")
        if skills["matched"]:
            st.write(", ".join(skills["matched"]))
        else:
            st.write("None of the required skills were found.")
    with missing_col:
        st.markdown("**❌ Missing skills**")
        if skills["missing"]:
            st.write(", ".join(skills["missing"]))
        else:
            st.write("Nothing missing — full coverage.")
    if skills["extra"]:
        with st.expander(f"Other skills on the resume ({len(skills['extra'])})"):
            st.write(", ".join(skills["extra"]))

    # --- Keywords -------------------------------------------------------
    st.subheader("Keyword overlap")
    st.write(
        "The job description's most frequent keywords — these are what "
        "recruiters and ATS filters scan for."
    )
    present_col, missing_kw_col = st.columns(2)
    with present_col:
        st.markdown("**Mentioned in the resume**")
        st.write(", ".join(keywords["present"]) or "—")
    with missing_kw_col:
        st.markdown("**Not mentioned**")
        st.write(", ".join(keywords["missing"]) or "—")

    # --- Sections -------------------------------------------------------
    st.subheader("Resume checklist")
    for check in result["sections"]:
        if check["passed"]:
            st.markdown(f"✅ {check['label']}")
        else:
            st.markdown(f"❌ {check['label']} — *{check['tip']}*")

    # --- Suggestions ----------------------------------------------------
    st.subheader("Suggestions")
    for suggestion in result["suggestions"]:
        st.markdown(f"- {suggestion}")


def main():
    st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄",
                       layout="wide")
    st.title("📄 AI Resume Analyzer")
    st.write(
        "Compare a resume against a job description: get a match score, "
        "see which required skills are present or missing, and receive "
        "actionable suggestions to close the gaps."
    )

    resume_col, job_col = st.columns(2)

    # --- Resume input ---------------------------------------------------
    with resume_col:
        st.subheader("1. Resume")
        uploaded_resume = st.file_uploader(
            "Upload a resume (PDF or TXT)", type=["pdf", "txt"],
            key="resume_upload",
        )
        use_sample_resume = st.checkbox(
            "Use the sample resume instead", key="sample_resume",
        )

    # --- Job description input ------------------------------------------
    with job_col:
        st.subheader("2. Job description")
        uploaded_job = st.file_uploader(
            "Upload a job description (TXT or PDF)", type=["txt", "pdf"],
            key="job_upload",
        )
        if st.button("Load the sample job description"):
            st.session_state["job_text"] = load_sample(SAMPLE_JOB_PATH)
        job_text_input = st.text_area(
            "…or paste the job description here", height=220,
            key="job_text",
            placeholder="Paste the job description text here…",
        )

    # Resolve the resume text: an upload wins over the sample checkbox.
    resume_text = ""
    if uploaded_resume is not None:
        try:
            resume_text = ra.extract_text(
                uploaded_resume.getvalue(), uploaded_resume.name
            )
        except Exception as exc:  # unreadable PDF etc.
            st.error(f"Could not read the uploaded resume: {exc}")
    elif use_sample_resume:
        resume_text = load_sample(SAMPLE_RESUME_PATH)

    # Resolve the job description text: an upload wins over the text box.
    job_text = job_text_input
    if uploaded_job is not None:
        try:
            job_text = ra.extract_text(
                uploaded_job.getvalue(), uploaded_job.name
            )
        except Exception as exc:
            st.error(f"Could not read the uploaded job description: {exc}")

    # --- Run -------------------------------------------------------------
    if st.button("Analyze", type="primary"):
        if not resume_text.strip():
            st.error("Please upload a resume or tick 'Use the sample resume instead'.")
        elif not job_text or not job_text.strip():
            st.error("Please paste or upload a job description (or load the sample).")
        else:
            with st.spinner("Analyzing…"):
                result = ra.analyze_resume(resume_text, job_text)
            show_results(result)

    with st.expander("How it works"):
        st.markdown(
            "1. **Text extraction** — the resume's text is pulled from the "
            "PDF/TXT file.\n"
            "2. **Match score** — resume and job description are converted "
            "to TF-IDF vectors and compared with cosine similarity.\n"
            "3. **Skill extraction** — both texts are scanned for 72 "
            "known tech skills; the JD's skills are the requirement list, "
            "and the resume is checked against it.\n"
            "4. **Keyword & section checks** — the JD's top keywords and "
            "the sections recruiters expect (contact details, education, "
            "experience, projects, skills) are verified.\n"
            "5. **Suggestions** — every gap found above becomes a concrete "
            "editing suggestion."
        )


if __name__ == "__main__":
    main()
