import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq

# ---- CONFIG ----
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---- HELPERS ----
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def get_best_chunks(question, chunks, chunk_embeddings, top_k=3):
    q_embedding = embedder.encode([question])
    scores = np.dot(chunk_embeddings, q_embedding.T).flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def ask_groq(question, context):
    prompt = f"""You are a helpful assistant. Answer the question using only the context below.

Context:
{context}

Question: {question}
Answer:"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ---- UI ----
st.title("📄 RAG Document Chatbot")
st.write("Upload a PDF and ask questions about it!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading and processing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(raw_text)
        chunk_embeddings = embedder.encode(chunks)
    st.success(f"PDF processed! ({len(chunks)} chunks created)")

    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Thinking..."):
            best_chunks = get_best_chunks(question, chunks, chunk_embeddings)
            context = "\n\n".join(best_chunks)
            answer = ask_groq(question, context)
        st.markdown("### 💬 Answer")
        st.write(answer)
