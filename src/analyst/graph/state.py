from typing import Annotated
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    question: str # original user question, never mutated

    plan: str
    needs_qualitative: bool
    needs_quantitative: bool

    retrieved_chunks: list[dict]
    narrative_answer: str
    retrieval_query: str
    retry_count: int

    quantitative_answer: str # exact figures from structured tables

    critic_feedback: str # what the Critic found, if anything
    critic_approved: bool

    final_answer: str

    human_feedback: str
    human_approved: bool