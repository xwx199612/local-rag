from app.vectorstore import VectorStore
from app.llm import ask_llm
from app.utils import split_text


def detect_answer_language(question):
    has_chinese = any("\u4e00" <= char <= "\u9fff" for char in question)
    has_english = any(("a" <= char.lower() <= "z") for char in question)

    if has_chinese and has_english:
        return "Traditional Chinese"
    if has_english:
        return "English"
    return "Traditional Chinese"


class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()

    def index(self, text):
        chunks = split_text(text)
        self.store.build(chunks)

    def query(self, question):
        answer, _ = self.query_with_sources(question)
        return answer

    def query_with_sources(self, question):
        if self.store.index is None:
            raise RuntimeError("Please upload and index at least one PDF before asking a question.")

        sources = self.store.search(question)
        context = "\n\n".join(
            self._format_source(source, source_num)
            for source_num, source in enumerate(sources, start=1)
        )
        answer_language = detect_answer_language(question)

        prompt = f"""You are a practical and careful document question-answering assistant.
Answer the user question based on the provided document context.

Answer language:
- You must answer in {answer_language}.
- If the question mixes Chinese and English, answer in Traditional Chinese.
- Do not choose the answer language based on the document language.

Rules:
1. Use the document context as the main basis for your answer. You may summarize, organize, compare, and make reasonable simple inferences from it.
2. Do not add unsupported facts, numbers, people, dates, causal claims, or external knowledge.
3. If the context provides partial clues, answer the supported part first, then briefly state what the document does not provide.
4. Only say there is not enough information when the context is almost completely unrelated or gives no useful clues.
5. If there is not enough information and the answer language is English, answer: "I cannot find enough information in the provided document to answer this question."
6. If there is not enough information and the answer language is Traditional Chinese, answer: "根據目前文件內容，我找不到足夠資訊回答這個問題。"
7. Keep the answer concise and clear. Use bullet points when helpful.
8. You may cite "Source 1", "Source 2", etc. when useful, but do not invent sources that are not listed below.

Document context:
{context}

User question:
{question}

Answer in {answer_language}:"""
        return ask_llm(prompt), sources

    def _format_source(self, source, source_num):
        page = source.get("page") if isinstance(source, dict) else None
        source_name = source.get("source") if isinstance(source, dict) else None
        text = source.get("text") if isinstance(source, dict) else source

        metadata = []
        if source_name:
            metadata.append(f"File: {source_name}")
        if page:
            metadata.append(f"Page: {page}")

        metadata_text = f" ({', '.join(metadata)})" if metadata else ""
        return f"[Source {source_num}{metadata_text}]\n{text}"
