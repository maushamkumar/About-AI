import streamlit as st 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import LangChainTracer
import tempfile
import os
import asyncio

# Fix for asyncio and torch issues
import nest_asyncio
nest_asyncio.apply()

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # For Mac users with M1/M2 chips

# LangChain Tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_PROJECT'] = "legal-rag-app"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Streamlit UI setup
st.set_page_config(page_title="📄 Legal RAG Assistant", layout="wide")
st.title("⚖️ Legal RAG Assistant")
st.markdown("Upload a legal PDF (like the Constitution, contracts, laws, etc.) and ask questions about it.")

# Sidebar for Groq API key 
st.sidebar.title("🔐 Settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# Check for Groq API key
if not groq_api_key:
    st.sidebar.warning("Please enter your Groq API key to use the app.")
    st.stop()
    
# Set the key in the environment 
os.environ["GROQ_API_KEY"] = groq_api_key

@st.cache_resource
def load_embedding_model():
    """Cache the embedding model to avoid reloading"""
    try:
        return HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},  # Force CPU to avoid GPU issues
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

def process_pdf(uploaded_file):
    """Process the uploaded PDF and create vector store"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Load and split document
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = splitter.split_documents(pages)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        # Create embeddings and vector store
        embedding_model = load_embedding_model()
        if embedding_model is None:
            return None, None
            
        vectordb = FAISS.from_documents(docs, embedding_model)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        
        return retriever, len(docs)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None

def create_qa_chain(retriever):
    """Create the QA chain"""
    try:
        # Legal Assistant Prompt 
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a helpful legal assistant specialized in interpreting legal documents. "
             "Use only the provided context (from the uploaded PDF) to answer the user's query. "
             "If the context does not contain enough information or the question is too complex, "
             "politely recommend that the user seek professional legal advice. "
             "Do not make up answers. Respond clearly and concisely. "
             "Always cite relevant sections or pages when possible."),
            ("user", "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:")
        ])
        
        llm = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",  # Changed to a more stable model
            temperature=0,
            max_tokens=1000
        )
        
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        return retrieval_chain
        
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# Main app logic
uploaded_file = st.file_uploader("Upload a legal PDF document", type=["pdf"])

if uploaded_file:
    # Initialize session state
    if "retrieval_chain" not in st.session_state:
        st.session_state.retrieval_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    
    # Process document if not already processed
    if not st.session_state.document_processed:
        with st.spinner("Processing your legal document..."):
            retriever, num_chunks = process_pdf(uploaded_file)
            
            if retriever is not None:
                st.session_state.retrieval_chain = create_qa_chain(retriever)
                st.session_state.document_processed = True
                st.success(f"Document processed successfully! Created {num_chunks} text chunks.")
            else:
                st.error("Failed to process the document. Please try again.")
                st.stop()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message['user'])
        with st.chat_message("assistant"):
            st.markdown(message['assistant'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document:"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            if st.session_state.retrieval_chain is not None:
                try:
                    with st.spinner("Thinking..."):
                        response = st.session_state.retrieval_chain.invoke({
                            "input": prompt
                        })
                        answer = response.get('answer', 'No answer generated.')
                        
                    st.markdown(answer)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "user": prompt,
                        "assistant": answer
                    })
                    
                except Exception as e:
                    error_msg = "⚠️ Sorry, an error occurred while processing your request. Please try again."
                    st.markdown(error_msg)
                    st.error(f"Error details: {str(e)}")
            else:
                st.markdown("⚠️ The document processing chain is not available. Please refresh and try again.")

    # Add a clear chat button
    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.info("Please upload a legal PDF document to get started.")
    st.markdown("""
    ### How to use:
    1. Enter your Groq API key in the sidebar
    2. Upload a legal PDF document (Constitution, contracts, laws, etc.)
    3. Wait for the document to be processed
    4. Ask questions about the document content
    
    ### Tips:
    - Be specific in your questions
    - Ask about particular sections or clauses
    - The assistant will only use information from your uploaded document
    """)