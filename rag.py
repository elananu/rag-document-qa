"""Core RAG pipeline: PDF -> chunks -> embeddings -> ChromaDB -> Groq LLM."""

import os
import uuid

import chromadb
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM = "openai/gpt-oss-20b"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 4             # chunks retrieved per question


# ---------- Stage 2: load PDF and split into chunks ----------

def load_pdf(file) -> list[dict]:
    """Return [{'page': 1, 'text': '...'}, ...] for every page that has text."""
    reader = PdfReader(file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": number, "text": text})
    return pages


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring to cut at whitespace."""
    text = " ".join(text.split())  # normalise whitespace
    if len(text) <= size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(pages: list[dict]) -> list[dict]:
    """Chunk every page and keep the page number with each chunk."""
    chunks = []
    for page in pages:
        for piece in split_text(page["text"]):
            chunks.append({"page": page["page"], "text": piece})
    return chunks


# ---------- Stages 3-5: embed, store, retrieve, answer ----------

class RagEngine:
    def __init__(self, embedder: SentenceTransformer | None = None):
        self.embedder = embedder or SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.EphemeralClient()
        self.collection = None

    def index(self, chunks: list[dict]) -> int:
        """Embed all chunks and store them in a fresh ChromaDB collection."""
        if not chunks:
            raise ValueError("No text chunks to index.")
        self.collection = self.client.create_collection(name=f"doc_{uuid.uuid4().hex}")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
        self.collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"page": c["page"]} for c in chunks],
        )
        return len(chunks)

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        """Return the k chunks most similar to the question."""
        if self.collection is None:
            raise RuntimeError("Index a document first.")
        query_embedding = self.embedder.encode([question], show_progress_bar=False).tolist()
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count()),
        )
        return [
            {"page": meta["page"], "text": doc}
            for doc, meta in zip(result["documents"][0], result["metadatas"][0])
        ]

    def answer(self, question: str) -> dict:
        """Retrieve context, ask the LLM, return the answer and its sources."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

        sources = self.retrieve(question)
        context = "\n\n".join(f"[Page {s['page']}]\n{s['text']}" for s in sources)

        system_prompt = (
            "You answer questions using ONLY the context provided from a document. "
            "Cite the page number like [Page 3] after each claim. "
            "If the context does not contain the answer, reply exactly: "
            "\"I couldn't find that in the document.\" Do not use outside knowledge."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", DEFAULT_LLM),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return {"answer": response.choices[0].message.content, "sources": sources}
