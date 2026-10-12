"""Streamlit UI for the Document Q&A (RAG) app. Run: streamlit run app.py"""

import hashlib

import streamlit as st
from sentence_transformers import SentenceTransformer
from rag import EMBED_MODEL, RagEngine, build_chunks, load_pdf
st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


st.title("📄 Document Q&A with Citations")
st.caption("Upload a PDF, ask questions, and see which page each answer came from.")

with st.sidebar:
    st.header("1. Upload")
    uploaded = st.file_uploader("PDF file", type="pdf")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Index the document only when a new file is uploaded
if uploaded is not None:
    data = uploaded.getvalue()
    file_id = hashlib.md5(data).hexdigest()
    if st.session_state.get("file_id") != file_id:
        with st.spinner("Reading and indexing your PDF..."):
            pages = load_pdf(uploaded)
            if not pages:
                st.error("No readable text found. This PDF may be scanned images.")
                st.stop()
            chunks = build_chunks(pages)
            engine = RagEngine(embedder=get_embedder())
            count = engine.index(chunks)
        st.session_state.engine = engine
        st.session_state.file_id = file_id
        st.session_state.file_name = uploaded.name
        st.session_state.messages = []
        st.session_state.stats = (len(pages), count)

if "engine" not in st.session_state:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

pages_n, chunks_n = st.session_state.stats
st.sidebar.success(f"{st.session_state.file_name}\n\n{pages_n} pages, {chunks_n} chunks indexed")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**Page {s['page']}**")
                    st.write(s["text"])

# New question
question = st.chat_input("Ask a question about your document")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = st.session_state.engine.answer(question)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.stop()
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for s in result["sources"]:
                st.markdown(f"**Page {s['page']}**")
                st.write(s["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
