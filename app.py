"""Local RAG Interactive Textbook Q&A — Streamlit application."""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import streamlit as st
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


APP_TITLE = "Textbook Tutor"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GENERATION_MODEL = "google/flan-t5-base"
CHUNK_SIZE = 850
CHUNK_OVERLAP = 140


@dataclass
class Passage:
    text: str
    source: str
    page: int | None

    @property
    def label(self) -> str:
        return f"{self.source}{f' · page {self.page}' if self.page else ''}"


def read_uploaded_file(upload) -> list[Passage]:
    """Extract text from a PDF, TXT, or Markdown upload."""
    suffix = Path(upload.name).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(upload)
        return [
            Passage((page.extract_text() or "").strip(), upload.name, i + 1)
            for i, page in enumerate(reader.pages)
            if (page.extract_text() or "").strip()
        ]
    raw = upload.getvalue().decode("utf-8", errors="replace")
    return [Passage(raw, upload.name, None)] if raw.strip() else []


def split_passage(passage: Passage) -> list[Passage]:
    text = re.sub(r"\s+", " ", passage.text).strip()
    if len(text) <= CHUNK_SIZE:
        return [passage]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1
        chunks.append(Passage(text[start:end].strip(), passage.source, passage.page))
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_generator(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)


@st.cache_resource(show_spinner=False)
def vector_client() -> chromadb.PersistentClient:
    """Open the local Chroma database stored beside this app."""
    database_path = Path(__file__).parent / "data" / "chroma"
    return chromadb.PersistentClient(path=str(database_path))


def textbook_collection():
    return vector_client().get_or_create_collection(
        name="textbook_library",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,  # We explicitly create embeddings with local Hugging Face.
    )


def store_passages(passages: list[Passage]) -> None:
    """Embed textbook chunks locally and persist their text/vectors in Chroma."""
    database = vector_client()
    try:
        database.delete_collection("textbook_library")
    except Exception:
        pass
    embeddings = load_embedder().encode(
        [passage.text for passage in passages],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    textbook_collection().add(
        ids=[f"chunk-{index}" for index in range(len(passages))],
        documents=[passage.text for passage in passages],
        embeddings=embeddings,
        metadatas=[{"source": passage.source, "page": passage.page or 0} for passage in passages],
    )


def retrieve(question: str, k: int) -> list[tuple[Passage, float]]:
    """Embed the question locally, then run cosine search in the vector store."""
    book = textbook_collection()
    query_embedding = load_embedder().encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    ).tolist()
    results = book.query(
        query_embeddings=query_embedding,
        n_results=min(k, book.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        (
            Passage(text=document, source=metadata["source"], page=metadata["page"] or None),
            max(0.0, min(1.0, 1 - float(distance))),
        )
        for document, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def answer_question(question: str, contexts: list[tuple[Passage, float]], model_name: str) -> str:
    evidence = "\n\n".join(f"[{p.label}]\n{p.text}" for p, _ in contexts)
    prompt = f"""You are a careful textbook tutor. Answer the question only using the supplied textbook excerpts.
If the excerpts do not contain the answer, say exactly that you could not find it in the uploaded textbook.
Explain clearly for a student and do not invent facts.

Textbook excerpts:
{evidence}

Question: {question}
Answer:"""
    # FLAN-T5 has a modest context window; preserve question plus the most useful material.
    prompt = prompt[:5500]
    result = load_generator(model_name)(prompt, max_new_tokens=220, do_sample=False)
    return result[0]["generated_text"].strip()


def show_answer(answer: str) -> None:
    """Render a theme-independent, readable generated answer."""
    safe_answer = html.escape(answer).replace("\n", "<br>")
    st.markdown(f"<div class='answer-text'>{safe_answer}</div>", unsafe_allow_html=True)


def show_source(source: Passage, score: float) -> None:
    """Render source labels and excerpts with explicit readable colors."""
    label = html.escape(source.label)
    excerpt = html.escape(source.text)
    st.markdown(
        f"<div class='source-card'><div class='source-title'>{label} · relevance {score:.0%}</div>"
        f"<div class='source-text'>{excerpt}</div></div>",
        unsafe_allow_html=True,
    )


def clear_library() -> None:
    try:
        vector_client().delete_collection("textbook_library")
    except Exception:
        pass
    for key in ("library_id", "messages"):
        st.session_state.pop(key, None)


st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
st.markdown(
    """<style>
    .stApp { background: linear-gradient(135deg, #f7fbff 0%, #ffffff 50%, #eefaff 100%); }
    [data-testid="stSidebar"] { background: #063b65; }
    [data-testid="stSidebar"] * { color: #f4fbff !important; }
    .hero { background: linear-gradient(100deg, #07a9ca, #0564a4); padding: 2rem 2.2rem;
      border-radius: 18px; color: white; margin-bottom: 1.5rem; }
    .hero h1 { margin: 0; font-size: 2.35rem; }
    .hero p { margin: .45rem 0 0; font-size: 1.1rem; opacity: .95; }
    .source-card { padding: .75rem 1rem; border-left: 4px solid #09a8ca; background: #eef8fc;
      border-radius: 0 8px 8px 0; margin-bottom: .55rem; color: #102a43 !important; }
    .source-card * { color: #102a43 !important; }
    .source-title { color: #063b65 !important; font-weight: 700; margin-bottom: .3rem; }
    .source-text { color: #102a43 !important; line-height: 1.5; }
    .sources-heading { background: #1d2029 !important; color: #ffffff !important; padding: .85rem 1.1rem;
      border-radius: 10px 10px 0 0; font-size: 1.05rem; font-weight: 700; margin-top: .9rem; }
    [data-testid="stExpander"] details { border: 0 !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary:hover {
      background: #1d2029 !important; color: #ffffff !important; border-radius: 10px !important;
      padding: .85rem 1.1rem !important;
    }
    [data-testid="stExpander"] summary * { color: #ffffff !important; }
    .answer-text { background: #ffffff !important; color: #102a43 !important; padding: 1rem 1.1rem;
      border: 1px solid #cfe1eb; border-left: 5px solid #08a9ca; border-radius: 10px;
      font-size: 1rem; line-height: 1.65; }
    .answer-text * { color: #102a43 !important; }
    /* Keep conversation text readable even when Streamlit or the browser uses a dark theme. */
    [data-testid="stMain"] [data-testid="stChatMessage"] {
      background: #ffffff !important; border: 1px solid #d9e8f0; border-radius: 12px;
      padding: .35rem .75rem; margin-bottom: .7rem;
    }
    [data-testid="stMain"] [data-testid="stChatMessage"] *,
    [data-testid="stMain"] [data-testid="stChatMessage"] p,
    [data-testid="stMain"] [data-testid="stChatMessage"] li {
      color: #102a43 !important;
    }
    [data-testid="stMain"] [data-testid="stChatMessage"] code {
      background: #e8f3f8 !important; color: #063b65 !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Your textbook library")
    uploads = st.file_uploader(
        "Upload textbook files", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    st.caption("PDF, TXT, and Markdown are embedded and stored locally in Chroma.")
    top_k = st.slider("Passages to retrieve", 2, 6, 4)
    model_name = st.selectbox("Local answer model", [DEFAULT_GENERATION_MODEL])
    if st.button("Clear library and chat", use_container_width=True):
        clear_library()
        st.rerun()

st.markdown("""<div class="hero"><h1>📚 Interactive Textbook Q&amp;A</h1>
<p>Ask a question. Search your local vector store. Learn from answers grounded in your book.</p></div>""", unsafe_allow_html=True)

if uploads:
    library_id = hashlib.sha256(
        "".join(f"{f.name}:{len(f.getvalue())}" for f in uploads).encode()
    ).hexdigest()
    if st.session_state.get("library_id") != library_id:
        with st.spinner("Reading and indexing your textbook locally…"):
            raw_passages = [p for upload in uploads for p in read_uploaded_file(upload)]
            chunks = [chunk for passage in raw_passages for chunk in split_passage(passage)]
            if not chunks:
                st.error("No readable text was found in these files.")
                st.stop()
            store_passages(chunks)
            st.session_state.library_id = library_id
            st.session_state.messages = []
        st.success(f"Ready: stored {len(chunks)} textbook chunks in the local vector database.")
else:
    st.info("Start by uploading one or more textbook files in the sidebar.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_answer(message["content"])
        else:
            st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View textbook sources", expanded=True):
                for source, score in message["sources"]:
                    show_source(source, score)

if prompt := st.chat_input("Ask about your textbook…", disabled="library_id" not in st.session_state):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Finding the best passages and composing an answer…"):
            selected = retrieve(prompt, top_k)
            response = answer_question(prompt, selected, model_name)
        show_answer(response)
        with st.expander("View textbook sources", expanded=True):
            for source, score in selected:
                show_source(source, score)
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": selected})
