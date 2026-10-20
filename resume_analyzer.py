"""
resume_analyzer.py — core logic for the AI Resume Analyzer.

This module holds everything except the UI. It extracts text from PDF/TXT
files, compares a resume against a job description using TF-IDF + cosine
similarity, extracts skills from a built-in skills database, checks for
the sections recruiters look for, and turns the gaps it finds into
actionable suggestions.

It has no Streamlit dependency, so it can be imported and unit-tested on
its own. Run it directly (`python resume_analyzer.py`) to analyse the
sample resume against the sample job description in ./samples.
"""

import io
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Skills database
#
# Skills are grouped by category so results can be shown per category.
# Matching is case-insensitive and works on whole words/phrases only, so
# "Java" never matches inside "JavaScript" and "SQL" never matches
# inside "MySQL".
# ---------------------------------------------------------------------------

SKILL_CATEGORIES = {
    "Languages": [
        "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript",
        "SQL", "HTML", "CSS", "R", "Go", "Kotlin", "Swift", "PHP", "Ruby",
    ],
    "Frameworks & Libraries": [
        "React", "Angular", "Vue.js", "Node.js", "Express", "Django",
        "Flask", "FastAPI", "Spring Boot", ".NET", "TensorFlow", "PyTorch",
        "scikit-learn", "Pandas", "NumPy", "Streamlit", "Bootstrap",
        "Tailwind CSS",
    ],
    "Databases": [
        "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Redis", "Oracle",
        "Firebase", "Elasticsearch",
    ],
    "Tools & Platforms": [
        "Git", "GitHub", "GitLab", "Docker", "Kubernetes", "AWS", "Azure",
        "Google Cloud", "Linux", "Jenkins", "Postman", "VS Code", "Jira",
        "CI/CD",
    ],
    "Concepts": [
        "Data Structures", "Algorithms", "Object-Oriented Programming",
        "Machine Learning", "Deep Learning", "Natural Language Processing",
        "REST APIs", "Agile", "Unit Testing", "System Design",
        "Operating Systems", "Computer Networks", "DBMS", "Cloud Computing",
        "Microservices", "Version Control",
    ],
}

# Common abbreviations / alternate spellings mapped to the canonical skill
# name (lowercased) they should count as. E.g. a resume that says "NLP"
# is credited with "Natural Language Processing".
SKILL_ALIASES = {
    "javascript": ["js"],
    "typescript": ["ts"],
    "postgresql": ["postgres"],
    "kubernetes": ["k8s"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "natural language processing": ["nlp"],
    "object-oriented programming": ["oop", "object oriented programming"],
    "rest apis": ["rest api", "restful apis", "restful api", "rest"],
    "google cloud": ["gcp"],
    "aws": ["amazon web services"],
    "scikit-learn": ["sklearn"],
    "version control": ["git version control"],
}


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_source):
    """Return the text of a PDF.

    `pdf_source` may be a file path, raw bytes, or a binary file-like
    object (which is what Streamlit's file uploader provides).
    """
    if isinstance(pdf_source, (bytes, bytearray)):
        pdf_source = io.BytesIO(bytes(pdf_source))
    reader = PdfReader(pdf_source)
    # Pages with no extractable text (e.g. scanned images) contribute "".
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text_from_txt(txt_source):
    """Return the text of a .txt file (path, bytes, or file-like object)."""
    if isinstance(txt_source, (str, Path)):
        return Path(txt_source).read_text(encoding="utf-8", errors="ignore")
    if isinstance(txt_source, (bytes, bytearray)):
        return bytes(txt_source).decode("utf-8", errors="ignore")
    # File-like object: an opened file or an uploaded file.
    data = txt_source.read()
    return data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else data


def extract_text(source, filename=""):
    """Extract text from a resume or job description.

    Picks the PDF or TXT parser based on the file extension taken from
    `filename` (or from `source` itself when it is a path).
    """
    name = (filename or str(source)).lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(source)
    return extract_text_from_txt(source)


# ---------------------------------------------------------------------------
# Cleaning and tokenisation
# ---------------------------------------------------------------------------

def clean_text(text):
    """Lowercase the text and strip punctuation / extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    """Split text into a list of lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", clean_text(text))


# ---------------------------------------------------------------------------
# Match score — TF-IDF + cosine similarity
# ---------------------------------------------------------------------------

def compute_match_score(resume_text, job_text):
    """Score how well the resume's wording matches the job description.

    Both documents are converted to TF-IDF vectors (words that appear in
    many documents count for less; distinctive words count for more) and
    the cosine similarity between the two vectors is returned as a
    percentage between 0 and 100.
    """
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    except ValueError:
        # Raised when a document has no usable words at all.
        return 0.0
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(float(similarity) * 100, 1)


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------

def _contains_phrase(text_lower, phrase):
    """True if `phrase` occurs in `text_lower` as a whole word/phrase.

    Uses lookarounds instead of \\b because \\b misbehaves around skills
    containing symbols such as "C++", "C#" or ".NET".
    """
    if phrase == "c":
        # Plain "C" must not match the C inside "C++" or "C#".
        return re.search(r"(?<!\w)c(?!\w|\+|#)", text_lower) is not None
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text_lower) is not None


def extract_skills_by_category(text):
    """Return {category: [skills found]} for the skills database."""
    text_lower = text.lower()
    found = {}
    for category, skills in SKILL_CATEGORIES.items():
        hits = []
        for skill in skills:
            names = [skill.lower()] + SKILL_ALIASES.get(skill.lower(), [])
            if any(_contains_phrase(text_lower, name) for name in names):
                hits.append(skill)
        if hits:
            found[category] = hits
    return found


def extract_skills(text):
    """Return the flat set of skills found in `text`."""
    by_category = extract_skills_by_category(text)
    return {skill for skills in by_category.values() for skill in skills}


def analyze_skills(resume_text, job_text):
    """Compare the skills a job description asks for with the resume's.

    Returns the required skills (from the JD), which of them the resume
    has (matched), which it lacks (missing), and any extra skills the
    resume mentions that the JD did not ask for.
    """
    required_by_category = extract_skills_by_category(job_text)
    required = {s for skills in required_by_category.values() for s in skills}
    resume_skills = extract_skills(resume_text)

    matched = sorted(required & resume_skills)
    missing = sorted(required - resume_skills)
    coverage = round(len(matched) / len(required) * 100, 1) if required else 0.0
    return {
        "required": sorted(required),
        "required_by_category": required_by_category,
        "matched": matched,
        "missing": missing,
        "extra": sorted(resume_skills - required),
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Keyword overlap
# ---------------------------------------------------------------------------

def keyword_overlap(resume_text, job_text, top_n=20):
    """Find the job description's most important keywords in the resume.

    Takes the `top_n` most frequent content words of the JD (stopwords
    such as "the" and "and" removed) and reports which of them appear in
    the resume. Recruiters and ATS software scan for exactly these words.
    """
    jd_tokens = [
        token for token in tokenize(job_text)
        if token not in ENGLISH_STOP_WORDS and len(token) > 2
    ]
    resume_tokens = set(tokenize(resume_text))
    top_keywords = [word for word, _ in Counter(jd_tokens).most_common(top_n)]

    present = [word for word in top_keywords if word in resume_tokens]
    missing = [word for word in top_keywords if word not in resume_tokens]
    coverage = round(len(present) / len(top_keywords) * 100, 1) if top_keywords else 0.0
    return {
        "top_keywords": top_keywords,
        "present": present,
        "missing": missing,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Section checks
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _has_phone_number(text):
    """Heuristic phone check: a digit sequence totalling 10–15 digits."""
    for candidate in re.findall(r"\+?\d[\d\s\-()]{8,}\d", text):
        digits = re.sub(r"\D", "", candidate)
        if 10 <= len(digits) <= 15:
            return True
    return False


def check_sections(resume_text):
    """Check the resume for the sections and details recruiters expect.

    Returns a list of checks, each with a label, whether it passed, and
    a tip shown when it did not.
    """
    text_lower = resume_text.lower()
    checks = [
        (
            "email", "Email address",
            bool(EMAIL_RE.search(resume_text)),
            "Add a professional email address at the top of the resume.",
        ),
        (
            "phone", "Phone number",
            _has_phone_number(resume_text),
            "Add a phone number (with country code) so recruiters can reach you.",
        ),
        (
            "education", "Education details",
            any(word in text_lower for word in (
                "education", "bachelor", "b.tech", "btech", "university",
                "college", "degree", "master",
            )),
            "Mention your degree, institution and graduation year.",
        ),
        (
            "experience", "Experience / internships",
            any(word in text_lower for word in (
                "experience", "internship", "intern", "worked at",
                "employed",
            )),
            "Add an Experience section — internships and freelance work count.",
        ),
        (
            "projects", "Projects",
            "project" in text_lower,
            "Add a Projects section with 2–3 projects and the tech used in each.",
        ),
        (
            "skills_section", "Dedicated skills section",
            "skill" in text_lower,
            "Add a clearly labelled 'Skills' section — it is what ATS software scans first.",
        ),
        (
            "summary", "Summary / objective",
            any(word in text_lower for word in ("summary", "objective", "profile")),
            "Open with a 2–3 line summary tailored to the role you are applying for.",
        ),
    ]
    return [
        {"key": key, "label": label, "passed": passed, "tip": tip}
        for key, label, passed, tip in checks
    ]


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

ACTION_VERBS = [
    "built", "developed", "designed", "implemented", "created", "improved",
    "automated", "optimised", "optimized", "led", "launched", "reduced",
    "increased", "analysed", "analyzed", "deployed",
]


def generate_suggestions(match_score, skills_result, keywords_result,
                         section_checks, resume_text):
    """Turn the gaps found by the other checks into actionable advice."""
    suggestions = []

    missing_skills = skills_result["missing"]
    if missing_skills:
        top = ", ".join(missing_skills[:5])
        extra = f" (and {len(missing_skills) - 5} more)" if len(missing_skills) > 5 else ""
        suggestions.append(
            f"Add these skills from the job description if you genuinely have "
            f"them: {top}{extra}. If you don't, they are your learning list."
        )
    elif skills_result["required"]:
        suggestions.append(
            "Great skill coverage — every skill named in the job description "
            "appears in your resume."
        )

    missing_keywords = keywords_result["missing"]
    if missing_keywords:
        suggestions.append(
            "Work these job-description keywords naturally into your project "
            f"and experience bullets: {', '.join(missing_keywords[:6])}."
        )

    for check in section_checks:
        if not check["passed"]:
            suggestions.append(check["tip"])

    verbs_used = {verb for verb in ACTION_VERBS if verb in resume_text.lower()}
    if len(verbs_used) < 3:
        suggestions.append(
            "Start bullet points with strong action verbs (built, developed, "
            "implemented, automated) and quantify results where you can — "
            "'improved load time by 30%' beats 'worked on performance'."
        )

    word_count = len(tokenize(resume_text))
    if word_count < 200:
        suggestions.append(
            "Your resume is quite short — expand project and experience "
            "bullets with what you built, the tools used, and the outcome."
        )
    elif word_count > 1200:
        suggestions.append(
            "Your resume is long — trim it towards one page by cutting "
            "bullets that don't support this specific role."
        )

    # Bands are calibrated for raw TF-IDF cosine similarity: even a good
    # resume/JD pair rarely exceeds ~60, since the two documents differ.
    if match_score < 20:
        suggestions.append(
            "Overall wording differs a lot from the job description. Mirror "
            "the JD's own terminology for the skills you have — many "
            "companies filter resumes by keyword before a human reads them."
        )
    elif match_score >= 55:
        suggestions.append(
            "Strong overall match. Spend the remaining effort on the "
            "interview: be ready to explain every skill listed on the resume."
        )

    return suggestions


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_resume(resume_text, job_text):
    """Run the complete analysis and return all results in one dict."""
    match_score = compute_match_score(resume_text, job_text)
    skills_result = analyze_skills(resume_text, job_text)
    keywords_result = keyword_overlap(resume_text, job_text)
    section_checks = check_sections(resume_text)
    suggestions = generate_suggestions(
        match_score, skills_result, keywords_result, section_checks, resume_text
    )
    return {
        "match_score": match_score,
        "skills": skills_result,
        "keywords": keywords_result,
        "sections": section_checks,
        "suggestions": suggestions,
        "word_count": len(tokenize(resume_text)),
    }


# ---------------------------------------------------------------------------
# Demo: analyse the bundled sample files when run as a script.
# ---------------------------------------------------------------------------

def _print_report(result):
    skills = result["skills"]
    print(f"Match score:       {result['match_score']} / 100")
    print(f"Resume word count: {result['word_count']}")
    print(f"Skill coverage:    {skills['coverage']}% "
          f"({len(skills['matched'])}/{len(skills['required'])} required skills)")
    print(f"\nMatched skills: {', '.join(skills['matched']) or 'none'}")
    print(f"Missing skills: {', '.join(skills['missing']) or 'none'}")
    print("\nSection checks:")
    for check in result["sections"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {check['label']}")
    print("\nSuggestions:")
    for suggestion in result["suggestions"]:
        print(f"  - {suggestion}")


if __name__ == "__main__":
    samples_dir = Path(__file__).parent / "samples"
    resume = extract_text(samples_dir / "sample_resume.txt")
    job = extract_text(samples_dir / "sample_job_description.txt")
    _print_report(analyze_resume(resume, job))
