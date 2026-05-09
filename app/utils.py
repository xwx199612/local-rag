from pypdf import PdfReader


def load_pdf_pages(file, source=None):
    reader = PdfReader(file)
    return [
        {
            "source": source,
            "page": page_num,
            "text": page.extract_text() or "",
        }
        for page_num, page in enumerate(reader.pages, start=1)
    ]


def load_pdf(file):
    return "\n".join(page["text"] for page in load_pdf_pages(file))


def split_text(text, chunk_size=500):
    if isinstance(text, list):
        chunks = []
        for page in text:
            page_chunks = split_text(page["text"], chunk_size=chunk_size)
            for chunk in page_chunks:
                chunk["source"] = page.get("source")
                chunk["page"] = page["page"]
            chunks.extend(page_chunks)
        return chunks

    words = text.split()
    return [
        {
            "source": None,
            "page": None,
            "text": " ".join(words[i:i + chunk_size]),
        }
        for i in range(0, len(words), chunk_size)
    ]
