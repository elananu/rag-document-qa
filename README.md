# Document Q&A with Citations (RAG)

Upload a PDF, ask questions, and get answers with page-level citations.

**Live Demo:** https://rag-document-app0.streamlit.app/

## Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- Groq API

## Run Locally in VS Code (Windows)

### 1. Open the project

Unzip the project folder and open it in VS Code.

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Get your API key from https://console.groq.com/keys.

**Security:** Never upload your `.env` file or API key to GitHub.

### 5. Run the application

```powershell
streamlit run app.py
```

Open http://localhost:8501 in your browser.

The first run downloads the sentence-embedding model, which may take some time.

## Deploy on Streamlit Community Cloud

1. Push the project files to a GitHub repository.
2. Visit https://share.streamlit.io/.
3. Sign in with GitHub and select **Create app**.
4. Choose your repository, branch (`main`), and main file (`app.py`).
5. Open the app's settings and add your secrets:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
GROQ_MODEL = "openai/gpt-oss-20b"
```

6. Save the secrets and deploy the application.

Live app: https://rag-document-app0.streamlit.app/

## How It Works

1. **PDF loading:** `load_pdf` extracts text page by page using PyPDF.
2. **Text chunking:** `split_text` creates overlapping text chunks.
3. **Embedding and indexing:** `RagEngine.index` converts chunks into embeddings and stores them in ChromaDB with page numbers.
4. **Retrieval:** `RagEngine.retrieve` finds the four most relevant chunks for a question.
5. **Answer generation:** `RagEngine.answer` sends the retrieved context and question to the Groq language model.
6. **Citations:** Streamlit displays the answer and relevant source passages.

## Troubleshooting

- **Missing API key:** Check that `.env` exists locally or that `GROQ_API_KEY` is configured in Streamlit Cloud Secrets.
- **Model not found:** Check the currently available models in https://console.groq.com/docs/models and update `GROQ_MODEL`.
- **SyntaxError:** Check quotes and Python syntax in `rag.py`.
- **No readable text:** The PDF may contain scanned images rather than extractable text. OCR may be required.
- **Deployment error:** Open the app dashboard and inspect the logs.
