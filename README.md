# Document Q&A with Citations (RAG)

Upload a PDF, ask questions, get answers with page-level citations.

**Stack:** Python, Streamlit, pypdf, sentence-transformers (all-MiniLM-L6-v2), ChromaDB, Groq API.

## Setup in VS Code (Windows)

1. Unzip the folder and open it in VS Code (File > Open Folder).
2. Open a terminal (Ctrl + `) and run:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. In VS Code press Ctrl+Shift+P > "Python: Select Interpreter" > choose the one inside `venv`.
4. Copy `.env.example` to a new file named `.env` and paste your Groq key:

```
GROQ_API_KEY=your_key_here
```

Get a free key at console.groq.com > API Keys.

5. Run the app:

```
streamlit run app.py
```

It opens at http://localhost:8501. The first run downloads the embedding model (about 90 MB).

## Mac/Linux

Use `source venv/bin/activate` instead of `venv\Scripts\activate`.

## How it works

1. `load_pdf` reads text page by page (pypdf).
2. `split_text` cuts each page into overlapping chunks.
3. `RagEngine.index` embeds chunks and stores them in ChromaDB with page numbers.
4. `RagEngine.retrieve` embeds the question and fetches the top 4 similar chunks.
5. `RagEngine.answer` sends the chunks plus the question to a Groq LLM, told to answer only from the context.
6. `app.py` shows the answer and the source passages in Streamlit.

## Troubleshooting

- **GROQ_API_KEY is missing:** check that `.env` exists in the project root and has no quotes or spaces.
- **Model not found error:** Groq retires models occasionally. Add `GROQ_MODEL=<current model name>` to `.env` (see console.groq.com/docs/models).
- **No readable text:** the PDF is likely scanned images; use a text-based PDF.
