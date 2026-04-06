"""
Claude LLM Client - RAG answer generation using Claude claude-opus-4-6.
Uses adaptive thinking + streaming for best quality and timeout safety.
"""

import anthropic


SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions using the provided context.

Instructions:
- Answer only based on the given context.
- If the context does not contain enough information, say so clearly.
- Cite the source document when possible.
- Be concise and accurate.
"""


class ClaudeRAGClient:
    """
    Wraps the Anthropic Claude claude-opus-4-6 API for RAG-based question answering.
    Uses adaptive thinking and streaming for reliability on long outputs.
    """

    def __init__(self, model: str = "claude-opus-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model

    def build_context_block(self, retrieved_chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk["metadata"].get("source", "unknown")
            parts.append(f"[{i}] Source: {source}\n{chunk['content']}")
        return "\n\n---\n\n".join(parts)

    def answer(self, question: str, retrieved_chunks: list[dict]) -> str:
        """Generate a streamed answer grounded in retrieved context."""
        context = self.build_context_block(retrieved_chunks)
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        full_response = []
        with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response.append(text)

        print()  # newline after streaming
        return "".join(full_response)

    def answer_batch(self, qa_pairs: list[dict]) -> list[dict]:
        """
        Non-streaming batch answer for multiple questions.
        Each item in qa_pairs: {"question": str, "chunks": list[dict]}
        """
        results = []
        for item in qa_pairs:
            context = self.build_context_block(item["chunks"])
            user_message = f"Context:\n{context}\n\nQuestion: {item['question']}"

            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            answer_text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            results.append({"question": item["question"], "answer": answer_text})
        return results
