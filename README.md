# AI Resume–Job Matching & Placement Assistant

A beginner-friendly end-to-end NLP/GenAI project that:

- uploads a resume PDF,
- extracts and cleans its text,
- semantically compares it with a job description,
- detects matched and missing skills,
- calculates a transparent match-score breakdown,
- generates deterministic resume/interview recommendations,
- optionally uses Gemini for personalized coaching, and
- includes a lightweight RAG-style "Ask Resume" feature.

## Architecture

```text
Resume PDF ──> PyMuPDF text extraction ──> cleaning ───────────────┐
                                                                  │
Job Description ──────────────────────────────────────────────────┤
                                                                  v
                                      Sentence Transformers / TF-IDF
                                                  │
                                                  v
                                           Semantic Match
                                                  │
Resume + JD ──> Skill dictionary ──> skill coverage───────────────┤
                                                                  │
Resume ──> structural checks ──> completeness─────────────────────┤
                                                                  v
                                          Transparent Overall Score
                                                  │
                           ┌──────────────────────┴─────────────────┐
                           v                                        v
                  Rule-based coaching                         Gemini (optional)
                                                                    │
                                                        Resume coaching + RAG Q&A
```

## Important design choice

The displayed score is **not claimed to be an employer ATS score**.

It is a project-specific, explainable heuristic:

- 50% semantic similarity
- 35% detected JD skill coverage
- 15% basic resume completeness

That makes the project easier to defend in an interview than an unexplained "ATS score".

## Tech stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- scikit-learn (TF-IDF fallback + cosine similarity)
- Gemini through the current `google-genai` SDK (optional)
- Lightweight retrieval over resume chunks

## Recommended Python version

Python **3.11 or 3.12** is a comfortable choice for this project.

## Setup on Windows

Open PowerShell in the project folder.

### 1. Create an environment

```powershell
py -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can instead use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run

```powershell
streamlit run app.py
```

You can also double-click `run_windows.bat`, although running the commands
manually is better while learning the project.

## Gemini setup (optional)

The entire matching pipeline works without Gemini.

To enable AI Coach and generated RAG answers:

1. Copy `.env.example` to `.env`.
2. Put your Gemini API key in `.env`:

```env
GEMINI_API_KEY=your_real_key
GEMINI_MODEL=gemini-3.7-flash
```

Never commit `.env` to GitHub.

The model name is configurable so you can switch models without modifying the
source code.

## First Sentence Transformer run

`all-MiniLM-L6-v2` normally needs to be downloaded the first time.
After it is available locally, the model can be reused from cache.

If Sentence Transformers cannot load, the program automatically falls back to
TF-IDF so the demo does not completely fail.

## Project structure

```text
ai-resume-placement-assistant/
├── app.py
├── data/
│   └── skills.json
├── src/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── recommendations.py
│   ├── resume_parser.py
│   ├── scoring.py
│   ├── skill_extractor.py
│   └── text_utils.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── run_windows.bat
└── sample_job_description.txt
```

## How each module works

### `resume_parser.py`
Uses PyMuPDF to extract text from the uploaded PDF. The current project does
not use OCR, so scanned/image-only resumes should be converted to searchable
PDFs first.

### `text_utils.py`
Cleans text, redacts common contact information before LLM calls, splits common
resume sections, creates retrieval chunks, and computes basic completeness.

### `skill_extractor.py`
Loads the curated `data/skills.json` dictionary and detects technologies using
aliases and safe regex boundaries.

### `embeddings.py`
Primary backend: Sentence Transformers (`all-MiniLM-L6-v2`).

Fallback: TF-IDF + cosine similarity.

It also ranks resume chunks for the "Ask Resume" retrieval feature.

### `scoring.py`
Combines the transparent project metrics into one score.

### `recommendations.py`
Creates useful feedback and interview questions even if no LLM API is
available.

### `llm.py`
Uses Gemini for:
- personalized resume feedback,
- gap analysis,
- interview questions,
- grounded answers from retrieved resume chunks.

### `app.py`
Connects everything in the Streamlit UI and stores analysis results in
`st.session_state`.

## Suggested demo flow

1. Start the app.
2. Upload your own resume.
3. Paste `sample_job_description.txt`.
4. Click **Analyze Resume**.
5. Show the score breakdown.
6. Show matched/missing skills.
7. Open **AI Coach** and generate recommendations.
8. Open **Ask Resume** and ask:
   - "What ML projects has this candidate done?"
   - "What skills make this candidate relevant to the role?"
   - "Does the resume mention Docker?"

## What to say in an interview

A concise explanation:

> I built an AI resume–job matching application that combines semantic
> embeddings with explicit skill extraction rather than relying only on
> keyword matching. I used Sentence Transformers for resume/JD similarity,
> added a TF-IDF fallback for reliability, built transparent scoring and
> skill-gap analysis, and integrated Gemini for personalized resume coaching.
> I also implemented lightweight retrieval over resume chunks so questions can
> be answered using relevant resume evidence.

## Next upgrades

Do these only after the current version works:

1. Better section parser.
2. Configurable role-specific skill dictionaries.
3. Resume bullet quality scoring.
4. Persist past analyses using SQLite/PostgreSQL.
5. FastAPI backend + React frontend.
6. Authentication.
7. Evaluate matching quality on a labeled resume/JD dataset.
8. Dockerize and deploy.
9. Add OCR as an optional scanned-PDF path.
10. Replace the heuristic score with a trained/ranked model only if you obtain
    appropriate labeled data.

## Privacy note

Resume content can contain personal data. The app locally redacts common
email/phone patterns before sending resume text to Gemini, but this is not a
complete de-identification system. For sensitive resumes, keep Gemini disabled
or implement a stronger privacy layer.
