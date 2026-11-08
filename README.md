# AI Resume Analyzer

An NLP-based tool that compares a resume against a job description and surfaces actionable skill-match insights. Upload a resume, paste a job description, and get a match score, a matched-vs-missing skills breakdown, keyword coverage, a resume checklist, and concrete suggestions to close the gaps — all in an interactive Streamlit interface designed for a smooth recruiter and applicant experience.

## Features

- **Resume upload** — PDF or TXT; the text is extracted automatically (PDFs via `pypdf`).
- **Match score** — TF-IDF + cosine similarity between the resume and the job description, shown as a 0–100 score.
- **Skill-gap analysis** — both texts are scanned against a built-in database of 72 tech skills (languages, frameworks, databases, tools, concepts). Skills the job asks for are reported as *matched* or *missing*, with a coverage percentage.
- **Keyword overlap** — the job description's most important keywords are checked against the resume, the way recruiters and ATS filters scan.
- **Resume checklist** — verifies email, phone number, and the presence of education, experience, projects, skills and summary sections.
- **Actionable suggestions** — every gap found becomes a specific editing suggestion, not just a number.
- **Instant demo** — a sample resume and sample job description are bundled in `samples/`, so the app works end-to-end the moment it starts.

## Tech Stack

- **Python** — core language
- **Streamlit** — interactive web UI
- **scikit-learn** — TF-IDF vectorisation and cosine similarity
- **pypdf** — PDF text extraction
- **NLP techniques** — tokenisation, stopword removal, phrase matching

## Setup & Run

```bash
# 1. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens in your browser (usually at http://localhost:8501). Tick **"Use the sample resume instead"** and click **"Load the sample job description"**, then **Analyze**, to see a full demo without any files of your own.

You can also run the analysis engine from the command line, without the UI:

```bash
python resume_analyzer.py
```

This analyses the bundled samples and prints a report to the terminal.

## Project Structure

```
AI-Resume-Analyzer/
├── app.py                    # Streamlit UI (input + presentation only)
├── resume_analyzer.py        # Core logic: extraction, scoring, skills, suggestions
├── requirements.txt          # streamlit, scikit-learn, pypdf
├── samples/
│   ├── sample_resume.txt         # Demo resume
│   └── sample_job_description.txt  # Demo job description
└── README.md
```

## How It Works

1. **Text extraction** — `extract_text()` picks a parser by file extension: `pypdf` reads every page of a PDF; TXT files are decoded as UTF-8.
2. **Cleaning & tokenisation** — text is lowercased and split into word tokens, with punctuation and extra whitespace normalised.
3. **Match score** — resume and job description are converted into TF-IDF vectors (rare, distinctive words weigh more than common ones). The cosine similarity between the two vectors is the match score.
4. **Skill extraction** — both documents are scanned for the skills database using whole-phrase matching, so "Java" is never credited for "JavaScript" and "SQL" never for "MySQL". Common aliases count too — "NLP" counts as Natural Language Processing, "k8s" as Kubernetes. The job description's skills form the requirement list; the resume is checked against it to produce matched and missing lists.
5. **Keyword overlap** — the job description's 20 most frequent content words (stopwords removed) are checked against the resume, giving a keyword coverage percentage.
6. **Section checks** — regex and keyword checks confirm contact details and the sections recruiters expect.
7. **Suggestions** — missing skills, missing keywords and failed section checks are converted into plain-language, actionable advice.

## Design Note

The project is deliberately split in two: `resume_analyzer.py` contains all the logic and has **no Streamlit dependency**, so it can be imported, unit-tested and run from the command line; `app.py` only collects input and renders results. This separation keeps the analysis code reusable (e.g. in a REST API or a batch screening script) and easy to explain.

## Possible Extensions

- Named-entity recognition to pull out names, companies and dates automatically.
- Experience extraction ("3 years of Python") compared against the JD's requirement.
- Support for `.docx` resumes and batch screening of many resumes against one JD, ranked by score.
