from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeminiAssistant:
    api_key: str
    model: str = "gemini-3.7-flash"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("A Gemini API key is required.")

        from google import genai

        self.client = genai.Client(api_key=self.api_key)

    def analyze_resume(
        self,
        resume_text: str,
        job_description: str,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> str:
        prompt = f"""
You are a resume coach helping a student prepare for a job application.

Rules:
- Be specific and concise.
- Never tell the candidate to claim a skill or experience they do not actually have.
- Separate "resume evidence is missing" from "candidate definitely lacks the skill".
- Do not claim that the score is an employer's ATS score.
- Focus on changes that improve relevance, clarity, evidence, and interview readiness.

JOB DESCRIPTION:
{job_description[:9000]}

RESUME (contact details may be redacted):
{resume_text[:12000]}

MATCHED SKILLS:
{", ".join(matched_skills) if matched_skills else "None detected"}

JD SKILLS NOT DETECTED IN RESUME:
{", ".join(missing_skills) if missing_skills else "None detected"}

Return markdown with exactly these sections:
## Top Resume Improvements
5 numbered, actionable suggestions.

## Strongest Evidence
3 bullets identifying the most relevant evidence already present.

## Potential Gaps
Up to 5 bullets. Phrase uncertainty carefully.

## Personalized Interview Questions
8 questions based on the resume and job description.
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text or "No response was returned by Gemini."

    def answer_from_resume(
        self,
        question: str,
        retrieved_chunks: list[str],
    ) -> str:
        context = "\n\n---\n\n".join(retrieved_chunks)
        prompt = f"""
Answer the question using only the resume context below.
If the answer is not supported by the context, say that it is not stated in the resume.
Do not invent experience, dates, metrics, skills, or achievements.

RESUME CONTEXT:
{context}

QUESTION:
{question}
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text or "No response was returned by Gemini."
