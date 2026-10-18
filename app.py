import re
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="DocuMind AI", page_icon="📚", layout="wide")
st.title("📚 DocuMind AI")
st.caption("Ask questions about your PDF documents. Answers are grounded in the uploaded pages.")

with st.sidebar:
    st.header("About")
    st.write(
        "DocuMind AI extracts text from PDFs, finds relevant passages using TF-IDF "
        "similarity, and shows page references so you can verify the results."
    )
    st.info("This demo runs locally and does not require an API key.")
    st.write("**Supported input:** text-based PDF files")


def extract_pages(pdf_file):
    reader = PdfReader(BytesIO(pdf_file.getvalue()))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_number, "text": text})
    return pages


def split_sentences(text):
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text)
        if len(part.strip()) > 25
    ]


def retrieve_passages(question, pages, top_k=4):
    passages = [
        {"page": item["page"], "text": sentence}
        for item in pages
        for sentence in split_sentences(item["text"])
    ]
    if not passages:
        return []

    corpus = [question] + [item["text"] for item in passages]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return []

    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    ranked = sorted(zip(scores, passages), key=lambda item: item[0], reverse=True)
    return [
        {"score": float(score), **passage}
        for score, passage in ranked[:top_k]
        if score > 0
    ]


uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    try:
        pages = extract_pages(uploaded)
    except Exception as exc:
        st.error(f"Could not read this PDF: {exc}")
        st.stop()

    if not pages:
        st.warning("No selectable text was found. This may be a scanned PDF; try a text-based PDF.")
        st.stop()

    total_words = sum(len(item["text"].split()) for item in pages)
    first, second, third = st.columns(3)
    first.metric("Pages with text", len(pages))
    second.metric("Approx. words", f"{total_words:,}")
    third.metric("Document", uploaded.name)

    with st.expander("Preview extracted text"):
        for item in pages[:3]:
            st.markdown(f"**Page {item['page']}**")
            text = item["text"]
            st.write(text[:1200] + ("…" if len(text) > 1200 else ""))

    question = st.text_input(
        "Ask a question about this document",
        placeholder="Example: What are the main findings?"
    )
    if st.button("Find answer", type="primary", disabled=not question.strip()):
        with st.spinner("Searching the document..."):
            results = retrieve_passages(question, pages)

        st.subheader("Relevant passages")
        if not results:
            st.warning("No strong text match found. Try different keywords or ask a more specific question.")
        else:
            st.caption("These are relevant passages, not guaranteed generated answers. Check the cited page in the original PDF.")
            for index, result in enumerate(results, start=1):
                with st.container(border=True):
                    st.markdown(f"**Passage {index} · Page {result['page']}**")
                    st.write(result["text"])
                    st.progress(min(result["score"], 1.0), text=f"Text similarity: {result['score']:.0%}")
else:
    st.markdown(
        """
        ### How to use it
        1. Upload a text-based PDF.
        2. Ask a question using words from the document.
        3. Review relevant passages and page numbers.

        **Note:** This starter uses TF-IDF retrieval rather than a large language model.
        It is API-free and can later be extended with semantic embeddings or a RAG model.
        """
    )

st.divider()
st.caption("Built with Streamlit, PyPDF, and scikit-learn.")
