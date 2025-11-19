import io
import numpy as np
import pdfplumber
import streamlit as st
from openai import OpenAI
import os

# -------------- CONFIG --------------

st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF Q&A Chatbot")
st.caption("Upload a PDF and ask questions about its contents.")

# -------------- OPENAI CLIENT --------------

def get_client() -> OpenAI | None:
    """
    Returns an OpenAI client using API key from secrets or env.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# -------------- PDF HANDLING --------------

def read_pdf(file_like) -> str:
    """
    Extracts text from a PDF file-like object using pdfplumber.
    """
    text = []
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text.append(page_text)
    return "\n\n".join(text)

def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    """
    Splits text into overlapping chunks of roughly `chunk_size` characters.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# -------------- EMBEDDINGS & SEARCH --------------

def embed_chunks(client: OpenAI, chunks: list[str]) -> np.ndarray:
    """
    Gets embeddings for all chunks as a 2D numpy array.
    """
    if not chunks:
        return np.empty((0, 0))

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks,
    )
    vectors = np.array([item.embedding for item in response.data])
    return vectors

def embed_query(client: OpenAI, query: str) -> np.ndarray:
    """
    Gets embedding for a single query string.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    return np.array(response.data[0].embedding)

def cosine_similarities(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarities between a query vector and each document vector.
    """
    if doc_vecs.size == 0:
        return np.array([])

    # Normalize vectors
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)

    sims = d @ q
    return sims

# -------------- QA LOGIC --------------

def build_context(chunks: list[str], similarities: np.ndarray, top_k: int = 4) -> str:
    """
    Picks the top_k most similar chunks and concatenates them into a context string.
    """
    if len(chunks) == 0 or similarities.size == 0:
        return ""

    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # highest first

    selected_chunks = [chunks[i] for i in top_indices]
    context = "\n\n---\n\n".join(selected_chunks)
    return context

def answer_question_from_pdf(
    client: OpenAI,
    question: str,
    chunks: list[str],
    chunk_vectors: np.ndarray,
    model: str = "gpt-4.1-mini",
) -> str:
    """
    Retrieves top chunks related to the question and asks a chat model to answer
    based ONLY on those chunks.
    """
    if not chunks or chunk_vectors.size == 0:
        return "I don't have any PDF content loaded yet. Please upload and process a PDF first."

    # 1) Embed the question
    query_vec = embed_query(client, question)

    # 2) Similarity search
    sims = cosine_similarities(query_vec, chunk_vectors)
    if sims.size == 0:
        return "I couldn't compare your question to the document content. Try reprocessing the PDF."

    # 3) Build context from most similar chunks
    context = build_context(chunks, sims, top_k=4)
    if not context.strip():
        return "I couldn't build a context from the PDF. The document might be empty or unreadable."

    # 4) Ask the model, constrained to the context
    system_prompt = (
        "You are a helpful assistant that answers questions ONLY using the provided document context. "
        "If the answer is not contained in the context, reply with \"I don't know based on this document.\""
    )

    user_prompt = f"""
Context from the PDF:
---------------------
{context}

---------------------
Question: {question}

Answer based ONLY on the context above. If the answer is not clearly stated, say you don't know based on this document.
"""

    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return chat_response.choices[0].message.content.strip()

# -------------- SESSION STATE --------------

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "pdf_vectors" not in st.session_state:
    st.session_state.pdf_vectors = np.empty((0, 0))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# -------------- MAIN APP --------------

client = get_client()

if client is None:
    st.error(
        "No OpenAI API key found.\n\n"
        "Set `OPENAI_API_KEY` in Streamlit **Secrets** or as an environment variable."
    )
    st.stop()

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("1. Upload and process your PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.write(f"**Selected file:** `{uploaded_file.name}`")

        if st.button("📥 Process PDF"):
            with st.spinner("Reading and indexing PDF..."):
                pdf_bytes = uploaded_file.read()
                if not pdf_bytes:
                    st.error("The uploaded file seems empty. Please try another PDF.")
                else:
                    pdf_buffer = io.BytesIO(pdf_bytes)
                    text = read_pdf(pdf_buffer)

                    if not text.strip():
                        st.error("I couldn't extract any text from this PDF. It may be scanned images or encrypted.")
                    else:
                        chunks = split_text(text, chunk_size=1200, overlap=200)
                        vectors = embed_chunks(client, chunks)

                        st.session_state.pdf_chunks = chunks
                        st.session_state.pdf_vectors = vectors
                        st.session_state.pdf_name = uploaded_file.name

                        st.success(
                            f"PDF processed successfully! ✅\n\n"
                            f"- Chunks created: **{len(chunks)}**"
                        )

    if st.session_state.pdf_name:
        st.info(
            f"Active PDF: `{st.session_state.pdf_name}` · "
            f"Chunks: **{len(st.session_state.pdf_chunks)}**"
        )

with col_right:
    st.subheader("2. Ask questions about the PDF")

    user_question = st.text_input(
        "Your question",
        placeholder="Example: What is the main conclusion of this document?",
    )

    ask_clicked = st.button("🤖 Ask", use_container_width=True)

    if ask_clicked and user_question:
        if not st.session_state.pdf_chunks:
            st.error("Please upload and process a PDF first.")
        else:
            with st.spinner("Thinking..."):
                answer = answer_question_from_pdf(
                    client=client,
                    question=user_question,
                    chunks=st.session_state.pdf_chunks,
                    chunk_vectors=st.session_state.pdf_vectors,
                    model="gpt-4.1-mini",
                )

            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("Assistant", answer))

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Chat history")

        for speaker, message in st.session_state.chat_history:
            if speaker == "You":
                st.markdown(f"**🧑 You:** {message}")
            else:
                st.markdown(f"**🤖 Assistant:** {message}")
    else:
        st.caption("Ask a question after processing a PDF to see the conversation here.")
