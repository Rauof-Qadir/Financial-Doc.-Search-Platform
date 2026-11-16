
import re
from typing import Any

from src.generation.gemini_client import GeminiGenerator
from src.generation.rag_pipeline import FinancialRAGPipeline


CITATION_PATTERN = re.compile(
    r"Page:\s*(\d+)\s*\|\s*Chunk:\s*([A-Za-z0-9_.-]+)"
)

UNVERIFIED_ANSWER = (
    "I generated a response, but its citations could not be verified "
    "against the retrieved document evidence. Please review the source "
    "evidence before relying on this answer."
)

def validate_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:

    matches = CITATION_PATTERN.findall(answer)

    if not matches:
        return {
            "valid": False,
            "citations_found": 0,
            "invalid_citations": ["No valid citations found."],
        }

    source_map = {
        source["chunk_id"]: source["page_number"]
        for source in sources
    }

    invalid_citations = []

    for page_text, chunk_id in matches:
        page_number = int(page_text)

        if chunk_id not in source_map:
            invalid_citations.append(
                f"Unknown chunk ID: {chunk_id}"
            )
        elif source_map[chunk_id] != page_number:
            invalid_citations.append(
                f"Page mismatch for {chunk_id}: "
                f"claimed {page_number}, "
                f"actual {source_map[chunk_id]}"
            )

    return {
        "valid": len(invalid_citations) == 0,
        "citations_found": len(matches),
        "invalid_citations": invalid_citations,
    }


class FinancialQuestionAnsweringPipeline:

    def __init__(
        self,
        metadata: list[dict[str, Any]],
        top_k: int = 5,
    ):
        self.rag = FinancialRAGPipeline(
            metadata=metadata,
            top_k=top_k,
        )
        self.generator = GeminiGenerator()

    @staticmethod
    def build_prompt(question: str, context: str) -> str:
        return f"""
You are a financial document question-answering assistant.

Answer the user's question using ONLY the evidence provided below.

STRICT RULES:
1. Do not use outside knowledge.
2. Do not invent financial figures, dates, or facts.
3. If evidence is insufficient, clearly say so.
4. Give a direct and concise answer.
5. Cite sources using this exact format:
   [Page: PAGE_NUMBER | Chunk: CHUNK_ID]
6. Use only page numbers and chunk IDs present in the evidence.
7. Put each citation in its own separate bracket.
8. Do not combine multiple citations inside one bracket.
9. Preserve the units and currency used in the document.

USER QUESTION:
{question}

DOCUMENT EVIDENCE:
{context}

YOUR ANSWER:
""".strip()

    def answer(self, question: str) -> dict[str, Any]:

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        result = self.rag.run(query=question)

        if not result["evidence_found"]:
            return {
                "query": question,
                "answer": (
                    "I could not find supporting evidence for this "
                    "question in the indexed document."
                ),
                "evidence_found": False,
                "sources": [],
                "citation_validation": {
                    "valid": False,
                    "citations_found": 0,
                    "invalid_citations": [],
                },
                "model": self.generator.model,
            }

        prompt = self.build_prompt(
            question=question,
            context=result["context"],
        )

        generated_answer = self.generator.generate(prompt)

        sources = [
            {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
            }
            for chunk in result["retrieved_chunks"]
        ]

        citation_validation = validate_citations(
            answer=generated_answer,
            sources=sources,
        )

        final_answer = generated_answer

        if not citation_validation["valid"]:
            final_answer = UNVERIFIED_ANSWER

        return {
            "query": question,
            "answer": final_answer,
            "evidence_found": True,
            "sources": sources,
            "citation_validation": citation_validation,
            "model": self.generator.model,
        }
