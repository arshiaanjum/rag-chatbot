# 📄 AI-Powered Document Q&A Chatbot (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents and ask questions in natural language.

## 🚀 Features
- **PDF Ingestion & Text Extraction**: Extracts text chunks from uploaded PDF files.
- **Semantic Search**: Uses Sentence-Transformers to compute document embeddings and retrieve relevant context via cosine similarity.
- **Fast LLM Generation**: Powered by the Groq API (Llama 3.1) for accurate, context-aware answers.
- **Interactive UI**: Built with Streamlit for a clean, user-friendly interface.

## 🛠️ Tech Stack
- **Language**: Python
- **Embedding Model**: `sentence-transformers`
- **LLM**: Groq (Llama 3.1)
- **Framework & Libraries**: Streamlit, PyPDF, NumPy

## 💻 Quickstart
1. Clone the repository:
   ```bash
   git clone https://github.com/arshiaanjum/rag-chatbot.git
   cd rag-chatbot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
