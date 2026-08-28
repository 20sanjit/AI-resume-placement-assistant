from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.embeddings import SemanticMatcher
from src.llm import GeminiAssistant
from src.recommendations import (
    rule_based_interview_questions,
    rule_based_recommendations,
)
from src.resume_parser import extract_pdf_text
from src.scoring import overall_match_score, score_label
from src.skill_extractor import SkillExtractor
from src.text_utils import (
    chunk_text,
    clean_text,
    extract_sections,
    redact_contact_info,
    resume_completeness_score,
)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="AI Resume Placement Assistant",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading semantic matching model...")
def get_matcher() -> SemanticMatcher:
    return SemanticMatcher("all-MiniLM-L6-v2")


@st.cache_resource
def get_skill_extractor() -> SkillExtractor:
    return SkillExtractor(BASE_DIR / "data" / "skills.json")


def get_secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def analyze(resume_bytes: bytes, jd: str) -> dict:
    raw_resume = extract_pdf_text(resume_bytes)
    resume_text = clean_text(raw_resume)
    jd_text = clean_text(jd)

    matcher = get_matcher()
    skill_extractor = get_skill_extractor()

    semantic = matcher.similarity(resume_text, jd_text)

    resume_skills = skill_extractor.extract(resume_text)
    jd_skills = skill_extractor.extract(jd_text)
    comparison = skill_extractor.compare(resume_skills, jd_skills)

    sections = extract_sections(resume_text)
    completeness, completeness_missing = resume_completeness_score(
        resume_text, sections
    )

    overall = overall_match_score(
        semantic_score=semantic,
        skill_score=float(comparison["coverage"]),
        completeness_score=completeness,
    )

    return {
        "raw_resume": raw_resume,
        "resume_text": resume_text,
        "jd_text": jd_text,
        "semantic": round(semantic, 1),
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matched": comparison["matched"],
        "missing": comparison["missing"],
        "resume_only": comparison["resume_only"],
        "skill_coverage": round(float(comparison["coverage"]), 1),
        "sections": sections,
        "completeness": round(completeness, 1),
        "completeness_missing": completeness_missing,
        "overall": overall,
        "backend": matcher.backend,
        "backend_error": matcher.error,
    }


st.title("📄 AI Resume–Job Matching & Placement Assistant")
st.caption(
    "Compare a resume with a job description, find skill gaps, get transparent "
    "matching metrics, and optionally use Gemini for personalized coaching."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This project uses semantic text similarity + explicit skill matching. "
        "Its scores are project heuristics, **not official employer ATS scores**."
    )
    st.divider()
    gemini_key = get_secret("GEMINI_API_KEY")
    gemini_model = get_secret("GEMINI_MODEL", "gemini-3.7-flash")
    if gemini_key:
        st.success("Gemini API configured")
    else:
        st.info("Gemini API not configured — core matching still works.")
    st.caption(f"Gemini model: `{gemini_model}`")

left, right = st.columns(2)
with left:
    uploaded_file = st.file_uploader(
        "Upload resume (PDF)",
        type=["pdf"],
        help="Text-based PDFs work best. Image-only/scanned PDFs are not OCR'd.",
    )

with right:
    job_description = st.text_area(
        "Paste job description",
        height=230,
        placeholder="Paste the complete job description here...",
    )

analyze_clicked = st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    if uploaded_file is None:
        st.error("Please upload a resume PDF.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        try:
            with st.spinner("Analyzing resume and job description..."):
                result = analyze(uploaded_file.getvalue(), job_description)
                st.session_state["analysis"] = result
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

result = st.session_state.get("analysis")

if result:
    st.divider()

    st.subheader("Match Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Match", f'{result["overall"]}%')
    c2.metric("Semantic Match", f'{result["semantic"]}%')
    c3.metric("Skill Coverage", f'{result["skill_coverage"]}%')
    c4.metric("Resume Completeness", f'{result["completeness"]}%')

    st.progress(int(round(result["overall"])))
    st.write(f'**{score_label(result["overall"])}**')

    if result["backend"] == "tfidf":
        st.warning(
            "Sentence Transformers could not load, so the app used the TF-IDF fallback. "
            "The project still works, but semantic quality is lower."
        )
        if result["backend_error"]:
            with st.expander("Fallback details"):
                st.code(result["backend_error"])

    with st.expander("How the overall score is calculated"):
        st.markdown(
            """
- **50% Semantic Match** — similarity between the complete resume and JD.
- **35% Skill Coverage** — JD skills detected in the resume.
- **15% Resume Completeness** — basic structural checks.

This is intentionally transparent and is **not** presented as a real employer ATS score.
"""
        )

    overview_tab, skills_tab, coach_tab, rag_tab, text_tab = st.tabs(
        [
            "Overview",
            "Skills",
            "AI Coach",
            "Ask Resume",
            "Extracted Text",
        ]
    )

    with overview_tab:
        st.subheader("Immediate Recommendations")
        tips = rule_based_recommendations(
            result["resume_text"],
            result["missing"],
            result["completeness_missing"],
        )
        if tips:
            for tip in tips:
                st.write(f"• {tip}")
        else:
            st.success("No obvious structural issues detected by the basic checks.")

        st.subheader("Practice Interview Questions")
        questions = rule_based_interview_questions(
            result["matched"],
            result["missing"],
        )
        for index, question in enumerate(questions, start=1):
            st.write(f"**{index}.** {question}")

    with skills_tab:
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("### ✅ Matched Skills")
            if result["matched"]:
                for skill in result["matched"]:
                    st.write(f"• {skill}")
            else:
                st.write("No matched skills were detected from the current skill dictionary.")

        with s2:
            st.markdown("### ⚠️ JD Skills Not Detected in Resume")
            if result["missing"]:
                for skill in result["missing"]:
                    st.write(f"• {skill}")
            else:
                st.write("No missing JD skills were detected.")

        with st.expander("All skills detected in the resume"):
            st.write(", ".join(result["resume_skills"]) or "None detected")

        with st.expander("All skills detected in the job description"):
            st.write(", ".join(result["jd_skills"]) or "None detected")

        st.caption(
            '"Not detected" means the text matcher did not find the skill. '
            "It does not prove that the candidate lacks that skill."
        )

    with coach_tab:
        st.subheader("Gemini Resume Coach")
        st.caption(
            "For privacy, common email/phone patterns are redacted before resume "
            "text is sent to Gemini. The job description and remaining resume text "
            "are still sent to the Gemini API."
        )

        if not gemini_key:
            st.info(
                "Add `GEMINI_API_KEY` to a local `.env` file or Streamlit secrets "
                "to enable this feature."
            )
        else:
            if st.button("Generate Personalized AI Feedback"):
                try:
                    with st.spinner("Generating personalized feedback..."):
                        assistant = GeminiAssistant(
                            api_key=gemini_key,
                            model=gemini_model,
                        )
                        feedback = assistant.analyze_resume(
                            resume_text=redact_contact_info(result["resume_text"]),
                            job_description=result["jd_text"],
                            matched_skills=result["matched"],
                            missing_skills=result["missing"],
                        )
                        st.session_state["ai_feedback"] = feedback
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")

            if st.session_state.get("ai_feedback"):
                st.markdown(st.session_state["ai_feedback"])

    with rag_tab:
        st.subheader("Ask Questions About the Resume")
        st.write(
            "This uses lightweight retrieval: the resume is chunked, the most relevant "
            "chunks are retrieved, and Gemini answers from those chunks only."
        )

        question = st.text_input(
            "Question",
            placeholder="Example: What machine learning projects has this candidate done?",
        )

        if st.button("Ask Resume"):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                matcher = get_matcher()
                chunks = chunk_text(result["resume_text"])
                ranked = matcher.rank_chunks(question, chunks, top_k=3)
                retrieved = [chunk for chunk, _score in ranked]

                st.session_state["retrieved_chunks"] = ranked

                if gemini_key:
                    try:
                        assistant = GeminiAssistant(
                            api_key=gemini_key,
                            model=gemini_model,
                        )
                        answer = assistant.answer_from_resume(
                            question=question,
                            retrieved_chunks=[
                                redact_contact_info(chunk) for chunk in retrieved
                            ],
                        )
                        st.session_state["rag_answer"] = answer
                    except Exception as exc:
                        st.error(f"Gemini request failed: {exc}")
                else:
                    st.session_state["rag_answer"] = (
                        "Gemini is not configured. The relevant resume chunks are shown below."
                    )

        if st.session_state.get("rag_answer"):
            st.markdown("### Answer")
            st.write(st.session_state["rag_answer"])

        if st.session_state.get("retrieved_chunks"):
            with st.expander("Retrieved resume evidence"):
                for i, (chunk, score) in enumerate(
                    st.session_state["retrieved_chunks"], start=1
                ):
                    st.markdown(f"**Chunk {i} · similarity {score:.3f}**")
                    st.write(chunk)

    with text_tab:
        st.subheader("Extracted Resume Text")
        st.text_area(
            "Resume text",
            value=result["resume_text"],
            height=400,
            disabled=True,
        )
        st.subheader("Parsed Sections")
        if result["sections"]:
            for name, content in result["sections"].items():
                with st.expander(name.title()):
                    st.text(content)
        else:
            st.write("No common sections were detected.")
