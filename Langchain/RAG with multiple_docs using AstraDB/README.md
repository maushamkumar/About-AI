# 📚 Multi-Document RAG Chatbot with AstraDB & LangChain

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot capable of processing multiple document formats including `.txt`, `.pdf`, and `.pptx`. It leverages **LangChain** for document handling and query orchestration, and **AstraDB** as a vector store to enable efficient and scalable semantic search over custom documents.

---

## 🚀 Features

- 📄 **Multi-format Document Support**  
  Loads `.txt`, `.pdf`, and `.pptx` files from a local folder.

- ✂️ **Smart Document Chunking**  
  Uses `RecursiveCharacterTextSplitter` from LangChain for optimal chunking.

- 🧠 **Embedding & Vector Storage**  
  Converts document chunks into embeddings and stores them in **AstraDB**.

- 💬 **LLM-Powered Chat Interface** *(Assumed)*  
  Uses a Large Language Model (e.g., **LLaMA**, **Groq**, **OpenAI**) to generate context-aware responses.

---

## 🧱 Tech Stack

- **Python**
- **LangChain**
- **AstraDB** (Vector store for document embeddings)
- **LLMs** (e.g., Groq, OpenAI, LLaMA)
- **Document Loaders**:
  - `TextLoader` for `.txt`
  - `PyPDFLoader` for `.pdf`
  - `UnstructuredPowerPointLoader` for `.pptx`

---

## 📁 Project Structure

```bash
├── app.py                     # Main chatbot interface
├── utils/
│   ├── loader.py              # Loads and chunks documents
│   ├── embedder.py            # Embedding and vector store logic
├── docs/                      # Folder with input documents (.txt, .pdf, .pptx)
├── requirements.txt           # Project dependencies
├── .env                       # API keys and config
├── README.md                  # Project overview
