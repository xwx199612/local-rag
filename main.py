import hashlib

import streamlit as st

from app.rag import RAGPipeline
from app.utils import load_pdf_pages

st.title("Offline RAG AI Assistant")

st.write("Step 1: start")

# ===== RAG singleton =====
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()

rag = st.session_state.rag

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.write(f"Step 3: {len(uploaded_files)} PDF(s) uploaded")

    hash_source = b"".join(
        uploaded.name.encode("utf-8") + uploaded.getvalue()
        for uploaded in uploaded_files
    )
    file_hash = hashlib.md5(hash_source).hexdigest()

    try:
        all_pages = []
        total_length = 0

        for uploaded in uploaded_files:
            pages = load_pdf_pages(uploaded, source=uploaded.name)
            all_pages.extend(pages)
            total_length += sum(len(page["text"]) for page in pages)
            st.write(f"{uploaded.name}: {len(pages)} page(s)")

        st.write(f"Total pages: {len(all_pages)}")
        st.write(f"Total text length: {total_length}")
    except Exception as e:
        st.error(f"PDF load failed: {e}")
        st.stop()

    if "last_pdf" not in st.session_state or st.session_state.last_pdf != file_hash:
        try:
            rag.index(all_pages)
            st.session_state.last_pdf = file_hash
            st.success("Indexed OK")
        except Exception as e:
            st.error(f"Index failed: {e}")
            st.stop()

q = st.text_input("Ask a question")

if q:
    try:
        ans, sources = rag.query_with_sources(q)
        st.write(ans)

        with st.expander("Sources used"):
            for source_num, source in enumerate(sources, start=1):
                source_name = source.get("source") or "Unknown file"
                page = source.get("page") or "Unknown page"
                text = source.get("text", "")
                st.markdown(f"**Source {source_num} - {source_name}, Page {page}**")
                st.write(text)
    except Exception as e:
        st.error(f"Query failed: {e}")
