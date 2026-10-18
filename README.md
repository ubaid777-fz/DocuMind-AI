# DocuMind AI — Chat with PDFs

A local PDF question assistant built with Python, Streamlit, PyPDF, and scikit-learn. It extracts text from PDFs, ranks passages using TF-IDF and cosine similarity, and displays relevant passages with page references.

## Features
- Upload text-based PDF files
- Preview extracted text and document statistics
- Search relevant passages using TF-IDF similarity
- Display source page numbers and similarity scores
- No API key required

## Run on Windows
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Use Python 3.10 or newer. Scanned PDFs without selectable text are not supported because OCR is not included. This is retrieval-based search, not a generative chatbot; verify passages against the source PDF.

## Project structure
```text
DocuMind-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── sample_documents/
    └── README.md
```

## Future improvements
Semantic search, multi-PDF support, optional RAG integration, OCR, chat history, tests, and deployment.
