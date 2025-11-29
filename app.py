"""
Knowledge-Base Agent (single-file Streamlit app)

This file is intentionally self-contained so it can be dropped into a repository
and run locally. The app implements:

- Admin-only ingestion and management of documents (upload / replace / delete)
- Persistence of raw uploaded files to `uploaded_docs/`
- Optional preload folder `preload_docs/` for files dropped on the server
- Content-based deduplication using SHA-256 (to avoid ingesting identical content)
- A simple vector store saved as `vector_store/vectors.npy` and `vector_store/meta.json`
- A user-facing chat UI that searches the ingested documents and returns
  short answers with source references

Notes for readers:
- The app uses `sentence_transformers` for embeddings when available. If not, a
  zero-vector fallback is used so the rest of the app remains functional.
- A local LLM (`google/flan-t5-base`) is optionally used for answer generation
  if `transformers` are installed and CUDA/CPU is available.
- The code intentionally keeps all functionality in a single file for ease of
  demonstration (competition/demo style). Modularizing into multiple files is
  recommended for production.
"""

# ---------------------- Standard / Third-party imports ----------------------
import os
import json
import uuid
import re
import difflib
import tempfile
import time
import hashlib
from io import BytesIO
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np

# Document parsing libraries
import pdfplumber
import docx
import shutil

# Optional ML model imports (lazy-loaded below)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
except Exception:
    pipeline = None
import torch

# Streamlit app
import streamlit as st

# ----------------------------- Streamlit config -----------------------------
# IMPORTANT: set_page_config must be the first Streamlit call
st.set_page_config(page_title="Knowledge Base", layout="wide", initial_sidebar_state="expanded")

# ------------------------------- Configuration ------------------------------
PERSIST_DIR = "vector_store"
PRELOAD_DIR = "preload_docs"
UPLOADED_DIR = "uploaded_docs"

# Ensure persistence directories exist
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(PRELOAD_DIR, exist_ok=True)
os.makedirs(UPLOADED_DIR, exist_ok=True)

VEC_FILE = os.path.join(PERSIST_DIR, "vectors.npy")
META_FILE = os.path.join(PERSIST_DIR, "meta.json")

# Embedding & generator defaults (changeable)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_LLM_MODEL = "google/flan-t5-base"
LLM_MAX_TOKENS = 256

# Sample text used to provide a default policy example when needed
SAMPLE_POLICY = """Company HR Policy Document

1. Leave Policy:
- Employees are allowed 20 days of annual leave per year.
- Sick leave can be taken as needed with medical proof.
- Emergency leave of up to 3 days may be granted.

2. Working Hours:
- Official working hours are from 9:00 AM to 6:00 PM.
- Employees are expected to log in on time.
- Remote working is allowed with manager approval.

3. Dress Code:
- Casual attire is allowed from Monday to Friday.
- Formal attire may be required for client meetings.

4. Company Benefits:
- Health insurance is provided to all full-time employees.
- Free lunch is available in the cafeteria.
- Transportation allowance is offered.

5. Conduct:
- Employees must maintain professional behavior at all times.
- Harassment of any form is strictly prohibited.
"""

# ------------------------------ Lazy model loader ---------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Load optional embedding and generator models.

    This function is cached by Streamlit to avoid re-loading models on each
    script rerun. If models are unavailable, returns (None, None).
    """
    embedder = None
    generator = None
    # Sentence-transformers (embedding) - optional
    if SentenceTransformer is not None:
        try:
            embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception:
            embedder = None
    # Local LLM generator using Hugging Face transformers - optional
    if pipeline is not None:
        try:
            device = 0 if torch.cuda.is_available() else -1
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_MODEL)
            generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
        except Exception:
            generator = None
    return embedder, generator


embedder, generator = load_models()

# ------------------------------- Utilities ----------------------------------

def compute_file_hash_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest for given bytes. Used for deduplication."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


# ---------------------------- Text extraction -------------------------------

def extract_text_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber. Returns combined text."""
    out = []
    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            for p in pdf.pages:
                out.append(p.extract_text() or "")
    except Exception:
        return ""
    return "\n".join(out)


def extract_text_from_docx(data: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(data)
            tmp.flush()
            path = tmp.name
        doc = docx.Document(path)
        os.remove(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""


def extract_text_file(uploaded) -> Tuple[str, str, bytes]:
    """Given a streamlit UploadedFile or a file path, return (text, filename, raw_bytes)."""
    raw = None
    name = "uploaded"
    # If uploaded is a filepath string -> read raw bytes
    if isinstance(uploaded, str):
        path = uploaded
        name = os.path.basename(path)
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except Exception:
            return "", name, b""
    else:
        try:
            uploaded.seek(0)
        except Exception:
            pass
        raw = uploaded.read()
        name = getattr(uploaded, "name", "uploaded")
        name = name if isinstance(name, str) else "uploaded"

    # Dispatch based on filename extension
    if name.lower().endswith(".pdf"):
        return extract_text_from_pdf(raw), name, raw
    if name.lower().endswith(".docx"):
        return extract_text_from_docx(raw), name, raw
    try:
        return raw.decode("utf-8"), name, raw
    except Exception:
        return "", name, raw


# ------------------------------- Chunking ----------------------------------

def chunk_text(text: str, size=1000, overlap=200) -> List[str]:
    """Split long text into overlapping chunks.

    Default chunk size 1000 characters with 200-character overlap.
    """
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + size
        chunks.append(text[start:end])
        if end >= L:
            break
        start = end - overlap
    return chunks


# ------------------------------ Vector store --------------------------------

def embed_texts(texts: List[str]) -> np.ndarray:
    """Return embeddings for a list of texts using the loaded embedder.

    If no embedder is available this will raise a RuntimeError and callers
    should handle it by falling back to zeros.
    """
    if embedder is None:
        raise RuntimeError("Embedder not available")
    return embedder.encode(texts, convert_to_numpy=True)


def load_store() -> Tuple[np.ndarray, List[Dict]]:
    """Load persisted vectors and metadata from disk. Returns (vecs, meta).

    If store does not exist it returns (empty array, empty list).
    """
    if not os.path.exists(META_FILE) or not os.path.exists(VEC_FILE):
        return np.array([]), []
    try:
        vecs = np.load(VEC_FILE, allow_pickle=False)
    except Exception:
        vecs = np.array([])
    try:
        meta = json.load(open(META_FILE, "r", encoding="utf-8"))
    except Exception:
        meta = []
    return vecs, meta


def save_store(vecs: np.ndarray, meta: List[Dict]):
    """Persist vectors and metadata to disk."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    np.save(VEC_FILE, vecs)
    json.dump(meta, open(META_FILE, "w", encoding="utf-8"), indent=2)


def add_to_store(chunks: List[str], metas: List[Dict], embeddings: np.ndarray):
    """Append new chunks, metadata and embeddings to the persisted store."""
    vecs, meta = load_store()
    if vecs.size == 0:
        new_vecs = embeddings
    else:
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, 0)
        if vecs.ndim == 1:
            vecs = np.expand_dims(vecs, 0)
        try:
            new_vecs = np.vstack([vecs, embeddings])
        except Exception:
            new_vecs = embeddings
    new_meta = meta + metas
    save_store(new_vecs, new_meta)


def remove_source_from_store(source_name: str) -> int:
    """Remove all chunks that came from a specific source filename.

    Returns how many chunks were removed.
    """
    vecs, meta = load_store()
    if not meta or vecs.size == 0:
        return 0
    keep_indices = [i for i, m in enumerate(meta) if m.get("source") != source_name]
    removed_count = len(meta) - len(keep_indices)
    if removed_count <= 0:
        return 0
    new_meta = [meta[i] for i in keep_indices]
    try:
        if vecs.ndim == 1:
            # fallback for unexpected shapes
            if not keep_indices:
                new_vecs = np.array([])
            else:
                new_vecs = np.zeros((len(new_meta), vecs.shape[0]))
        else:
            new_vecs = vecs[keep_indices, :]
    except Exception:
        new_vecs = np.array([])
    save_store(new_vecs, new_meta)
    return removed_count


# ------------------------------- Retrieval ----------------------------------

def cosine(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_top_k(query: str, k: int = 4):
    """Return top-k candidate chunks for a query.

    Attempts semantic retrieval if embeddings are available, otherwise falls
    back to a simple keyword overlap scoring.
    """
    vecs, meta = load_store()
    if vecs.size == 0 or len(meta) == 0:
        return []

    # Try semantic retrieval using the embedder
    try:
        if embedder is not None:
            q_emb = embed_texts([query])[0]
            if vecs.ndim == 2 and q_emb.shape[0] == vecs.shape[1]:
                sims = [(i, cosine(q_emb, vecs[i])) for i in range(len(vecs))]
                sims.sort(key=lambda x: x[1], reverse=True)
                seen = set()
                results = []
                for idx, score in sims:
                    m = meta[idx]
                    docid = m.get("id")
                    if docid in seen:
                        continue
                    seen.add(docid)
                    results.append({"meta": m, "score": score})
                    if len(results) >= k:
                        break
                return results
    except Exception:
        pass

    # Fallback: basic keyword scoring
    q_words = [w for w in re.findall(r"[A-Za-z]{3,}", query.lower())]
    scored = []
    for i, m in enumerate(meta):
        txt = m.get("text", "").lower()
        score = sum(1 for w in q_words if w in txt)
        if q_words and " ".join(q_words) in txt:
            score += 2
        scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in scored:
        if score <= 0:
            continue
        m = meta[idx]
        results.append({"meta": m, "score": float(score)})
        if len(results) >= k:
            break
    return results


# ------------------------- Smart sentence extraction ------------------------

def extract_best_sentence(text: str, query: str) -> str:
    """Heuristically pick the sentence that best answers a query from a chunk."""
    if not text:
        return ""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return ""

    q_low = query.lower()
    q_words = re.findall(r"[A-Za-z]{3,}", q_low)

    scored_sents = []
    for s in sents:
        sl = s.lower()
        match_count = sum(1 for w in q_words if w in sl) if q_words else 0
        score = match_count * 10
        if q_low and q_low in sl:
            score += 6
        if re.search(r'\b\d+\b', s):
            score += 3
        # small domain-specific boosts
        if "leave" in sl and "annual" in sl:
            score += 2
        if "remote" in sl and ("remote" in q_low or "work" in q_low or "wfh" in q_low):
            score += 2
        scored_sents.append((s, score))

    scored_sents.sort(key=lambda x: x[1], reverse=True)
    if scored_sents and scored_sents[0][1] > 0:
        return scored_sents[0][0]

    # fallback: return first sentence that matches strong phrases
    strong_phrases = [
        "remote working", "remote work", "remote", "work from home", "wfh",
        "annual leave", "annual leaves", "leave policy", "leave", "sick leave", "emergency leave", "working hours"
    ]
    for s in sents:
        sl = s.lower()
        for p in strong_phrases:
            if p in sl:
                return s

    return sents[0]


# ---------------------------- Answer generation -----------------------------

def generate_answer_and_sources(query: str, retrieved: List[Dict]) -> Tuple[str, List[Dict]]:
    """Given a query and retrieved chunks, try to generate a concise answer.

    1) If a local generator is available, build a prompt and ask the model.
    2) Otherwise, heuristically extract short sentences from the top chunks.
    """
    blocks = []
    for i, r in enumerate(retrieved):
        m = r["meta"]
        chunknum = m.get("chunk", m.get("chunk_index", "?"))
        txt = m.get("text", "")[:1500]
        blocks.append(f"[Source {i + 1}] ({m.get('source', 'unknown')} - chunk {chunknum}): {txt}")
    context = "\n\n".join(blocks) if blocks else ""

    prompt = f"""Use the context to answer the question. Cite sources like [Source 1].
If answer not found in context say: "I don't know based on the provided documents."\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer (concise):\n"""

    # Try generator if available
    if generator is not None:
        try:
            out = generator(prompt, max_length=LLM_MAX_TOKENS, do_sample=False)
            ans = out[0].get("generated_text", "").strip()
            if ans and len(ans) > 8:
                return ans, retrieved
        except Exception:
            pass

    # Fallback: extract best sentence(s) from retrieved chunks
    extracted = []
    for r in retrieved:
        txt = r["meta"].get("text", "")
        best = extract_best_sentence(txt, query)
        if best:
            extracted.append(best)
    seen = set()
    dedup = []
    for s in extracted:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    if dedup:
        short = dedup[0].strip()
        return short, retrieved
    return "No answer found in the documents.", retrieved


# --------------------------- Autocorrect helpers ----------------------------

def build_vocab():
    """Build a vocabulary from all stored texts (used for autocorrect)."""
    _, meta = load_store()
    vocab = set()
    for m in meta:
        t = m.get("text", "").lower()
        words = re.findall(r"[a-zA-Z]{3,}", t)
        vocab.update(words)
    return sorted(vocab)


def autocorrect_query(q, vocab, cutoff=0.75):
    """Attempt to autocorrect query words using the built vocabulary."""
    words = q.split()
    out = []
    for w in words:
        lw = re.sub(r"[^a-zA-Z]", "", w).lower()
        if not lw:
            out.append(w)
            continue
        if lw in vocab:
            out.append(w)
            continue
        match = difflib.get_close_matches(lw, vocab, n=1, cutoff=cutoff)
        if match:
            out.append(match[0])
        else:
            out.append(w)
    return " ".join(out)


# ------------------- Ingestion / persistence / deduplication ----------------

def ingest_chunks_from_text(text: str, source_name: str, source_hash: str = None):
    """Chunk text, compute embeddings and add them to the vector store."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    try:
        embeddings = embed_texts(chunks) if embedder is not None else np.zeros((len(chunks), 1))
    except Exception:
        embeddings = np.zeros((len(chunks), 1))
    metas = []
    for j, c in enumerate(chunks):
        meta = {"id": str(uuid.uuid4()), "source": source_name, "chunk": j, "text": c}
        if source_hash:
            meta["source_hash"] = source_hash
        metas.append(meta)
    add_to_store(chunks, metas, embeddings)
    return len(chunks)


def ingest_file_path(path: str):
    """Ingest a persisted file (used for preload/persisted files). Returns number of chunks ingested."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except Exception:
        return 0
    name = os.path.basename(path)
    text = ""
    if name.lower().endswith(".pdf"):
        text = extract_text_from_pdf(raw)
    elif name.lower().endswith(".docx"):
        text = extract_text_from_docx(raw)
    else:
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = ""
    if not text.strip():
        return 0
    s_hash = compute_file_hash_bytes(raw)
    # dedupe by previously stored source_hash values
    _, meta = load_store()
    existing_hashes = set(m.get("source_hash") for m in meta if m.get("source_hash"))
    if s_hash in existing_hashes:
        return 0
    cnt = ingest_chunks_from_text(text, name, source_hash=s_hash)
    return cnt


def ingest_uploaded_fileobj(uploaded, overwrite_name: str = None):
    """Ingest an uploaded stream (Streamlit UploadedFile). Persists raw file and ingests chunks.

    Returns (count, source_hash, status_str) where status_str is one of
    {"duplicate", "no_text", "ingested"}.
    """
    try:
        uploaded.seek(0)
    except Exception:
        pass
    raw = uploaded.read()
    fname = getattr(uploaded, "name", f"upload_{int(time.time())}")
    if overwrite_name:
        fname = overwrite_name
    s_hash = compute_file_hash_bytes(raw)
    _, meta = load_store()
    existing_hashes = set(m.get("source_hash") for m in meta if m.get("source_hash"))
    if s_hash in existing_hashes:
        # already ingested identical content
        return 0, s_hash, "duplicate"
    # persist raw file
    safe_path = os.path.join(UPLOADED_DIR, fname)
    try:
        with open(safe_path, "wb") as fh:
            fh.write(raw)
    except Exception:
        pass
    # extract and ingest
    text = ""
    if fname.lower().endswith(".pdf"):
        text = extract_text_from_pdf(raw)
    elif fname.lower().endswith(".docx"):
        text = extract_text_from_docx(raw)
    else:
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = ""
    if not text.strip():
        return 0, s_hash, "no_text"
    cnt = ingest_chunks_from_text(text, fname, source_hash=s_hash)
    return cnt, s_hash, "ingested"


def ingest_sample_policy():
    """Ingest the SAMPLE_POLICY into the store (useful for demos)."""
    chunks = chunk_text(SAMPLE_POLICY)
    try:
        embeddings = embed_texts(chunks) if embedder is not None else np.zeros((len(chunks), 1))
    except Exception:
        embeddings = np.zeros((len(chunks), 1))
    metas = [{"id": str(uuid.uuid4()), "source": "policy.txt", "chunk": i, "text": c, "source_hash": None} for i, c in
             enumerate(chunks)]
    add_to_store(chunks, metas, embeddings)


def ingest_preload_folder(path=PRELOAD_DIR):
    """Ingest files that are already present in the preload folder on disk."""
    files = []
    for fn in sorted(os.listdir(path)):
        if fn.lower().endswith((".pdf", ".docx", ".txt")):
            files.append(os.path.join(path, fn))
    total = 0
    for p in files:
        total += ingest_file_path(p)
    return total


def list_persisted_files():
    """Return a list of persisted files from preload and uploaded directories."""
    files = []
    for d in [PRELOAD_DIR, UPLOADED_DIR]:
        if os.path.exists(d):
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".pdf", ".docx", ".txt")):
                    files.append({"path": os.path.join(d, fn), "name": fn, "dir": d})
    return files


def ingest_missing_persisted_files():
    """Ingest persisted files that are present on disk but not yet in the vector store."""
    _, meta = load_store()
    existing_sources = set(m.get("source") for m in meta)
    files = list_persisted_files()
    total = 0
    for f in files:
        if f["name"] not in existing_sources:
            try:
                total += ingest_file_path(f["path"])
            except Exception:
                pass
    return total


# ----------------------------- UI styling (CSS) -----------------------------
PAGE_CSS = """
<style>
:root{ /* color tokens */
  --bg-top:#051226;        /* top darker */
  --bg-bottom:#071827;     /* bottom darker */
  --panel:#0b1220;         /* card / panel */
  --muted:#9fb1c9;         /* muted text */
  --accent:#2563eb;        /* primary accent */
  --accent2:#60a5fa;
  --panel-contrast: rgba(255,255,255,0.02);
}

/* App background to match header and content */
.stApp, html, body, main, .block-container {
  background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%) !important;
  color: #e6eef8 !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; box-shadow: none !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#061223 0%, #071827 100%) !important; color: #fff !important; border-right: 1px solid rgba(255,255,255,0.02); }
.sidebar .sidebar-content { padding-top: 16px; }

h1,h2,h3 { color:#f8fafc; }
.small-muted { color:var(--muted); font-size:14px; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:18px; border-radius:12px; border:1px solid rgba(255,255,255,0.03); }
.dropzone { background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01)); border:2px dashed rgba(255,255,255,0.03); padding:28px; border-radius:14px; text-align:center; color:var(--muted); }
.stButton>button { background: linear-gradient(90deg,var(--accent),var(--accent2)) !important; color:white; border: none; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
.stButton > button[data-baseweb="button"][aria-label="Delete"] { background: linear-gradient(90deg, #dc2626, #ef4444) !important; color: white !important; border: none !important; box-shadow: 0 4px 12px rgba(220,38,38,0.4); }
.footer-badge { position: fixed; right: 12px; bottom: 10px; color:var(--muted); font-size:12px; }
.chat-window { background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01)); padding:18px; border-radius:12px; height:520px; overflow:auto; }
.msg-row { display:flex; margin-bottom:12px; align-items:flex-end; }
.msg-row.user { justify-content:flex-end; }
.msg-row.assistant { justify-content:flex-start; }
.avatar { width:36px; height:36px; border-radius:50%; background:#1f2937; display:inline-flex; align-items:center; justify-content:center; color:#fff; font-weight:600; margin:0 8px; font-size:14px; }
.bubble { max-width:72%; padding:10px 12px; border-radius:18px; line-height:1.3; font-size:14px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
.bubble.user { background: linear-gradient(90deg,var(--accent),var(--accent2)); color:white; border-bottom-right-radius:4px; }
.bubble.assistant { background: rgba(255,255,255,0.04); color:#e6eef8; border-bottom-left-radius:4px; }
.meta { font-size:11px; color:var(--muted); margin-top:6px; }
.input-area { display:flex; gap:8px; align-items:center; padding:10px 0; }
.input-box { flex:1; }
.send-btn { padding:10px 14px; border-radius:10px; }
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ------------------------ Session state initialization ----------------------
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'last_suggestion' not in st.session_state:
    st.session_state['last_suggestion'] = None
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'role' not in st.session_state:
    st.session_state['role'] = None


def perform_logout():
    """Clear auth state and rerun the app."""
    st.session_state['authenticated'] = False
    st.session_state['role'] = None
    time.sleep(0.15)
    st.rerun()


# -------------------- Auto-preload & ingest persisted files ----------------
def auto_ingest_preload_if_empty():
    """If the vector store is empty, auto ingest files from PRELOAD_DIR."""
    vecs, meta = load_store()
    if (vecs.size == 0 or len(meta) == 0):
        fns = [fn for fn in os.listdir(PRELOAD_DIR) if fn.lower().endswith((".pdf", ".docx", ".txt"))]
        if not fns:
            return 0
        total = ingest_preload_folder(PRELOAD_DIR)
        return total
    return 0


_preload_count = 0
try:
    _preload_count = auto_ingest_preload_if_empty()
except Exception:
    _preload_count = 0

_missing_ingested = 0
try:
    _missing_ingested = ingest_missing_persisted_files()
except Exception:
    _missing_ingested = 0

# ------------------------------- Login page --------------------------------
if not st.session_state.get('authenticated', False):
    st.markdown("<div style='max-width:700px;margin:18px auto;'><h1 style='margin:0'>Rooman QueryBot</h1><div class='small-muted'> </div></div>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='card'><h3 style='margin-top:2px;'>Login as User</h3>", unsafe_allow_html=True)
        with st.form(key='user_login_form'):
            user_username = st.text_input("Username", value="", placeholder="username", key="user_username")
            user_pwd = st.text_input("Password", type="password", key="user_pwd")
            login_user = st.form_submit_button("Login as User")
            if login_user:
                # Allow any password for user role (as requested by the developer).
                # Require at least a non-empty username or non-empty password to avoid accidental empty-press logins.
                if (user_username and user_username.strip()) or (user_pwd and user_pwd.strip()):
                    st.session_state['authenticated'] = True
                    st.session_state['role'] = 'user'
                    st.success("Logged in as User.")
                    time.sleep(0.15)
                    st.rerun()
                else:
                    st.error("Please enter a username or password to login.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h3 style='margin-top:2px;'>Login as Admin</h3></div>", unsafe_allow_html=True)
        with st.form(key='admin_login_form'):
            admin_username = st.text_input("Admin username", value="", placeholder="admin", key="admin_username")
            admin_pwd = st.text_input("Admin password", type="password", key="admin_pwd")
            login_admin = st.form_submit_button("Login as Admin")
            if login_admin:
                ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "roomanadmin")
                if admin_pwd == ADMIN_PASSWORD:
                    st.session_state['authenticated'] = True
                    st.session_state['role'] = 'admin'
                    st.success("Logged in as Admin.")
                    time.sleep(0.15)
                    st.rerun()
                else:
                    st.error("Invalid admin credentials.")
        st.markdown("</div>", unsafe_allow_html=True)

    # show notices about what got ingested automatically
    if _preload_count > 0:
        st.info(f"{_preload_count} chunk(s) auto-ingested from the preload folder ({PRELOAD_DIR}).")
    if _missing_ingested > 0:
        st.info(f"{_missing_ingested} chunk(s) ingested from persisted files (uploaded/preload).")

    st.stop()

# ----------------------------- Sidebar / Navigation ------------------------
with st.sidebar:
    st.markdown("<div style='padding:10px 6px 14px 8px'><h2 style='margin:0;color:#fff'>Knowledge Base</h2><div class='small-muted'>AI Assistant</div></div>", unsafe_allow_html=True)
    # Admins see Upload & Admin pages, users see Chat only
    if st.session_state.get('role') == 'admin':
        page = st.radio("", ["Upload Docs", "Admin"], index=0)
    else:
        page = st.radio("", ["Chat"], index=0)
    st.markdown("---")
    if st.button("Logout"):
        perform_logout()

# Guard: prevent non-admins from accessing upload page by URL
if 'page' in locals() and page == "Upload Docs" and st.session_state.get('role') != 'admin':
    st.warning("Upload is restricted to Admin users only.")
    st.stop()

# ------------------------------- Upload Page --------------------------------
if page == "Upload Docs":
    st.markdown("<h1>Upload Documents (Admin)</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Add documents to your knowledge base. Uploaded files are saved</div>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose files (PDF / DOCX / TXT)", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
    if st.button("Ingest uploaded files"):
        if not uploaded:
            st.warning("No files selected.")
        else:
            total_chunks = 0
            skipped = 0
            n_files = len(uploaded)
            progress = st.progress(0)
            status = st.empty()
            for i, f in enumerate(uploaded):
                status.info(f"Saving & reading {getattr(f, 'name', 'file')} ({i + 1}/{n_files})...")
                cnt, s_hash, status_str = ingest_uploaded_fileobj(f)
                if status_str == "duplicate":
                    skipped += 1
                else:
                    total_chunks += cnt
                progress.progress(int(((i + 1) / n_files) * 100))
                time.sleep(0.15)
            msg = f"Ingested files — {total_chunks} chunks saved."
            if skipped:
                msg += f" Skipped {skipped} file(s) because identical content already exists."
            status.success(msg)
            progress.empty()

    st.markdown("<br/>", unsafe_allow_html=True)

# ------------------------------- Chat Page ----------------------------------
elif page == "Chat":
    st.markdown("<div style='display:flex;align-items:center;justify-content:space-between'><div><h1 style='margin:0'>AI Assistant</h1><div class='small-muted'>Ask me a question</div></div></div>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    def process_query(q: str):
        """Handle user query: retrieve candidates, generate answer, append to history."""
        q = (q or "").strip()
        if not q:
            st.warning("Please type a question.")
            return
        st.session_state['chat_history'].append({"sender": "user", "text": q, "ts": datetime.utcnow().isoformat()})
        search_q = q
        st.session_state['last_suggestion'] = None
        retrieved = retrieve_top_k(search_q, k=4)
        if not retrieved:
            st.session_state['chat_history'].append(
                {"sender": "assistant", "text": "No documents found. Please ask an admin to ingest documents first.", "ts": datetime.utcnow().isoformat()})
            return
        answer, used = generate_answer_and_sources(search_q, retrieved)
        st.session_state['chat_history'].append({"sender": "assistant", "text": answer, "ts": datetime.utcnow().isoformat()})

    # Quick sample question tiles
    cols = st.columns(4, gap="large")
    sample_questions = [
        "Is emergency leave available?",
        "Is work from home allowed?",
        "What are company benefits?",
        "What are the working hours?"
    ]
    for i, q in enumerate(sample_questions):
        with cols[i]:
            if st.button(q, key=f"tile_{i}"):
                process_query(q)
            st.write("")

    st.markdown("<hr/>", unsafe_allow_html=True)

    chat_col, sidebar_col = st.columns([3, 1])
    with chat_col:
        st.markdown("<div class='card'><h3 style='margin-top:2px;'>Conversation</h3>", unsafe_allow_html=True)
        chat_win = st.container()
        with chat_win:
            html = ["<div class='chat-window' id='chat-window'>"]
            for msg in st.session_state['chat_history']:
                sender = msg.get('sender')
                text = msg.get('text', '')
                ts = msg.get('ts')
                try:
                    t = datetime.fromisoformat(ts)
                    timestr = t.strftime('%I:%M %p')
                except Exception:
                    timestr = ''
                if sender == 'user':
                    html.append(f"<div class='msg-row user'><div class='bubble user'>{text.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')}</div><div class='meta'>{timestr}</div></div>")
                else:
                    avatar = 'AI'
                    html.append(f"<div class='msg-row assistant'><div class='avatar'>{avatar}</div><div><div class='bubble assistant'>{text.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')}</div><div class='meta'>{timestr}</div></div></div>")
            html.append("</div>")
            st.markdown(''.join(html), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.form(key='chat_form', clear_on_submit=True):

            query = st.text_input("", value="", placeholder="Type a message...", key='chat_input')
            submitted = st.form_submit_button("Send")
            if submitted:
                process_query(query)

    # ---------------- Sidebar: show available documents (admins only) -----
    with sidebar_col:
        if st.session_state.get('last_suggestion'):
            st.markdown(f"<div style='margin-top:12px' class='card'><strong>Last suggestion</strong><div class='small-muted'>{st.session_state['last_suggestion']}</div></div>", unsafe_allow_html=True)

        if st.session_state.get('role') == 'admin':
            st.markdown("<div style='margin-top:12px' class='card'><strong>Available Documents</strong><div class='small-muted'>Files uploaded / preloaded on the server</div></div>", unsafe_allow_html=True)
            files = list_persisted_files()
            if not files:
                st.markdown("<div class='small-muted' style='padding:8px'>No files available. Ask Admin to upload or drop files into the preload folder.</div>", unsafe_allow_html=True)
            else:
                for f in files:
                    fname = f["name"]
                    fpath = f["path"]
                    st.markdown(f"<div style='margin-top:6px'>{fname}</div>", unsafe_allow_html=True)
                    try:
                        with open(fpath, "rb") as fh:
                            data = fh.read()
                        st.download_button(f"Download {fname}", data=data, file_name=fname, mime="application/octet-stream")
                    except Exception:
                        st.markdown("<div class='small-muted'>Unable to read file for download.</div>", unsafe_allow_html=True)
        else:
            pass

    # Small JS to auto-scroll the chat window to the bottom
    scroll_js = """
    <script>
    const el = document.getElementById('chat-window');
    if(el){ el.scrollTop = el.scrollHeight; }
    </script>
    """
    st.components.v1.html(scroll_js, height=0)

# ------------------------------- Admin dashboard ---------------------------
elif page == "Admin":
    st.markdown("<h1>Admin Dashboard</h1>", unsafe_allow_html=True)

    vecs, meta = load_store()
    n_docs = len({m.get("source") for m in meta})
    n_chunks = len(meta)
    col1, col2 = st.columns(2)
    col1.markdown(f"<div class='kpi'><h3>{n_docs}</h3><div class='small-muted'>Documents</div></div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Documents", unsafe_allow_html=True)
    if n_chunks == 0:
        st.info("No documents yet. Use Upload Docs (admin) or drop files into the preload folder before starting the app.")
    else:
        srcs = {}
        for m in meta:
            s = m.get("source", "unknown")
            srcs.setdefault(s, 0)
            srcs[s] += 1
        for s, cnt in srcs.items():
            st.write(f"- {s}")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Document Manager", unsafe_allow_html=True)
    persisted = list_persisted_files()
    if not persisted:
        st.markdown("<div class='small-muted'>No files ,please upload</div>", unsafe_allow_html=True)
    else:
        for i, f in enumerate(persisted):
            fname = f["name"]
            fpath = f["path"]
            fdir = os.path.basename(f["dir"])
            # show file header
            st.markdown(f"**{fname}**  ·  <span class='small-muted'>{fdir}</span>", unsafe_allow_html=True)

            # three columns: uploader (wide) | download | delete
            col_upl, col_dl, col_del = st.columns([6, 1, 1])

            with col_upl:
                # compact replace uploader (label collapsed so only browse button shows)
                rep = st.file_uploader(label=f"Replace {fname} (upload new file)", type=['pdf', 'docx', 'txt'], key=f"rep_upl_{i}", label_visibility="collapsed")
                if rep is not None:
                    try:
                        rep.seek(0)
                    except Exception:
                        pass
                    raw = rep.read()
                    # overwrite the persisted raw file
                    try:
                        with open(fpath, "wb") as fh:
                            fh.write(raw)
                    except Exception as e:
                        st.error(f"Failed to write replacement file: {e}")
                    # remove old chunks and ingest the replacement
                    removed = remove_source_from_store(fname)
                    tmp = BytesIO(raw)
                    tmp.name = fname
                    cnt, s_hash, status_str = ingest_uploaded_fileobj(tmp, overwrite_name=fname)
                    if status_str == "duplicate":
                        st.info("Replacement content already exists in the store (duplicate).")
                    elif status_str == "no_text":
                        st.warning("Replacement file has no extractable text and was not ingested.")
                    else:
                        st.success(f"Replaced {fname}: removed {removed} old chunk(s), ingested {cnt} new chunk(s).")
                    st.rerun()

            with col_dl:
                st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
                try:
                    with open(fpath, "rb") as fh:
                        data = fh.read()
                    st.download_button(label="Download", data=data, file_name=fname, mime="application/octet-stream", key=f"dl_{i}")
                except Exception:
                    st.markdown("<div class='small-muted'>Unable to read file</div>", unsafe_allow_html=True)

            with col_del:
                st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
                if st.button("Delete", key=f"del_{i}"):
                    removed = remove_source_from_store(fname)
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
                    st.success(f"Deleted {fname} — removed {removed} chunk(s) from vector store and deleted raw file.")
                    st.rerun()

