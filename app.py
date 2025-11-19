import streamlit as st
import pdfplumber
import numpy as np
import requests
import io

# --------------------------
# CONFIG
# --------------------------

st.set_page_config(page_title="PDF Q&A (Bytez FREE)", page_icon="📄", layout="wide")
st.title("📄 PDF Q&A Chatbot (FREE, Powered by Bytez API)")

BYTEZ_API_KEY = st.secrets.get("OPENAI_API_KEY")   # using your saved key
BASE_URL = "https://api.bytez.com/v1"              # Bytez API base


# --------------------------
# PDF READING
# --------------------------

def read_pdf(file):
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            content = page.extract_text() or ""
            text.append(content)
    return "\n\n".join(text)


def split_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# --------------------------
# BYTEZ EMBEDDINGS
# --------------------------

def get_embeddings(text_list):
    headers = {
        "Authorization": f"Bearer {BYTEZ_API_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(
        f"{BASE_URL}/embeddings",
        headers=headers,
        json={
            "model": "embed-english-v3.0",
            "input": text_list
        }
    )

    data = resp.json()
    vectors = [item["embedding"] for item in data["data"]]
    return np.array(vectors)


def embed_query(query):
    headers = {
        "Authorization": f"Bearer {BYTEZ_API_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(
        f"{BASE_URL}/embeddings",
        headers=headers,
        json={
            "model": "embed-english-v3.0",
            "input": [query]
        }
    )

    return np.array(resp.json()["data"][0]["embedding"])


# --------------------------
# COSINE SIMILARITY
# --------------------------

def cosine_similarities(query_vec, doc_vecs):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
    return d @ q


# --------------------------
# BYTEZ CHAT COMPLETION
# --------------------------

def ask_bytez(question, context):
    headers = {
        "Authorization": f"Bearer {BYTEZ_API_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4o-mini",   # free Bytez-compatible model
            "messages": [
                {
                    "role": "system",
                    "content": "Use ONLY the document context. If answer is not found, say 'I don't know based on this PDF.'"
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
        }
    )

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# --------------------------
# STREAMLIT STATE
# --------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------
# UI: PDF Upload
# --------------------------

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    if st.button("Process PDF"):
        with st.spinner("Extracting text..."):
            pdf_bytes = uploaded_file.read()
            text = read_pdf(io.BytesIO(pdf_bytes))

        with st.spinner("Splitting into chunks..."):
            chunks = split_text(text)

        with st.spinner("Creating embeddings (Bytez)..."):
            vectors = get_embeddings(chunks)

        st.session_state.chunks = chunks
        st.session_state.vectors = vectors

        st.success(f"PDF processed! {len(chunks)} chunks created.")


# --------------------------
# UI: Ask Questions
# --------------------------

question = st.text_input("Ask a question about the PDF:")

if st.button("Ask"):
    if not st.session_state.chunks:
        st.error("Upload and process a PDF first.")
    else:
        with st.spinner("Searching document..."):
            q_vec = embed_query(question)
            sims = cosine_similarities(q_vec, st.session_state.vectors)

            top_indices = np.argsort(sims)[-4:][::-1]
            selected = [st.session_state.chunks[i] for i in top_indices]

            context = "\n\n---\n\n".join(selected)

        with st.spinner("Generating answer (Bytez)..."):
            answer = ask_bytez(question, context)

        st.session_state.history.append(("You", question))
        st.session_state.history.append(("Bot", answer))


# --------------------------
# CHAT HISTORY
# --------------------------

for speaker, msg in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
