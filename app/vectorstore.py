import faiss
from app.embedding import embed

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = None

    def build(self, chunks):
        if not chunks:
            raise ValueError("No text chunks were created from the uploaded PDF files.")

        self.chunks = chunks
        texts = [chunk["text"] if isinstance(chunk, dict) else chunk for chunk in chunks]
        vectors = embed(texts)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    def search(self, query, k=3):
        if self.index is None or not self.chunks:
            raise RuntimeError("Please upload and index at least one PDF before asking a question.")

        k = min(k, len(self.chunks))
        q = embed([query])
        _, idx = self.index.search(q, k)
        return [self.chunks[i] for i in idx[0]]
