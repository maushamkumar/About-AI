# import streamlit as st
# from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
# from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import os
# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_groq import ChatGroq
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from pathlib import Path
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain
# import tempfile
# import nest_asyncio

# # Fix for asyncio issues
# nest_asyncio.apply()

# # Environment setup
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# # Load environment variables
# load_dotenv()

# # Streamlit page configuration
# st.set_page_config(page_title="Chat with Multiple Documents", layout="wide")
# st.title("Chat with Multiple Documents")

# # Sidebar for Groq API key
# st.sidebar.title("🔐 Settings")
# groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# # Check for Groq API key
# if not groq_api_key:
#     st.sidebar.warning("Please enter your Groq API key to use the app.")
#     st.stop()

# @st.cache_resource
# def load_embedding_model():
#     """Cache the embedding model to avoid reloading"""
#     try:
#         return HuggingFaceEmbeddings(
#             model_name='sentence-transformers/all-MiniLM-L6-v2',
#             model_kwargs={'device': 'cpu'},  # Force CPU to avoid GPU issues
#             encode_kwargs={'normalize_embeddings': True}
#         )
#     except Exception as e:
#         st.error(f"Error loading embedding model: {str(e)}")
#         return None

# def load_documents_from_directory(dir_path: str):
#     """Load documents from directory containing txt, pdf, and pptx files"""
#     try:
#         # Define the directory containing documents
#         txt_loader = DirectoryLoader(
#             path=dir_path,
#             glob="**/*.txt",
#             loader_cls=TextLoader
#         )
        
#         pdf_loader = DirectoryLoader(
#             path=dir_path,
#             glob="**/*.pdf",
#             loader_cls=PyPDFLoader
#         )
        
#         pptx_loader = DirectoryLoader(
#             path=dir_path,
#             glob="**/*.pptx",
#             loader_cls=UnstructuredPowerPointLoader
#         )
        
#         # Load documents from each loader
#         txt_docs = txt_loader.load()
#         pdf_docs = pdf_loader.load()
#         pptx_docs = pptx_loader.load()
        
#         # Combine all documents into a single list
#         all_docs = txt_docs + pdf_docs + pptx_docs
        
#         return all_docs
#     except Exception as e:
#         st.error(f"Error loading documents: {str(e)}")
#         return []

# def process_uploaded_files(uploaded_files):
#     """Process uploaded files and return documents"""
#     all_docs = []
    
#     for uploaded_file in uploaded_files:
#         try:
#             # Create temporary file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
#                 temp_file.write(uploaded_file.read())
#                 temp_file_path = temp_file.name
            
#             # Load based on file type
#             file_extension = uploaded_file.name.split('.')[-1].lower()
            
#             if file_extension == 'pdf':
#                 loader = PyPDFLoader(temp_file_path)
#                 docs = loader.load()
#             elif file_extension == 'txt':
#                 loader = TextLoader(temp_file_path)
#                 docs = loader.load()
#             elif file_extension in ['ppt', 'pptx']:
#                 loader = UnstructuredPowerPointLoader(temp_file_path)
#                 docs = loader.load()
#             else:
#                 st.warning(f"Unsupported file type: {file_extension}")
#                 continue
            
#             # Add source information
#             for doc in docs:
#                 doc.metadata['source_file'] = uploaded_file.name
            
#             all_docs.extend(docs)
            
#             # Clean up temp file
#             os.unlink(temp_file_path)
            
#         except Exception as e:
#             st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
#     return all_docs

# def doc_splitter(docs):
#     """Split documents into smaller chunks"""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,    # Increased chunk size for better context
#         chunk_overlap=200   # Increased overlap
#     )
    
#     split_docs = splitter.split_documents(docs)
    
#     return split_docs

# def create_vector_store(docs):
#     """Create vector store from documents"""
#     try:
#         embedding_model = load_embedding_model()
#         if embedding_model is None:
#             return None
        
#         if not docs:
#             st.error("No documents to process")
#             return None
            
#         db = FAISS.from_documents(documents=docs, embedding=embedding_model)
#         retriever = db.as_retriever(search_kwargs={'k': 5})
        
#         return retriever
#     except Exception as e:
#         st.error(f"Error creating vector store: {str(e)}")
#         return None

# def create_qa_chain(retriever):
#     """Create the QA chain"""
#     try:
#         # Create the prompt template
#         prompt = ChatPromptTemplate.from_template(
#             '''
#             You are an AI researcher who is an expert in RAG systems.
#             Answer any question asked by the user based on the provided context.
#             Construct answers in the form of bullet points when appropriate.
#             Craft your response only from the provided context.
#             If you cannot find any related information from the context, simply say "No relevant context provided."
#             Do not hallucinate.
#             Always mention the source file when possible.
            
#             <context>
#             {context}
#             </context>
            
#             QUESTION: {input}
#             '''
#         )
        
#         llm = ChatGroq(
#             model="llama3-8b-8192",
#             temperature=0,
#             api_key=groq_api_key
#         )
        
#         combine_docs_chain = create_stuff_documents_chain(llm, prompt)
#         retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
#         return retrieval_chain
        
#     except Exception as e:
#         st.error(f"Error creating QA chain: {str(e)}")
#         return None

# # Main app interface
# st.sidebar.markdown("---")
# st.sidebar.title("📁 Document Source")
# source_option = st.sidebar.radio(
#     "Choose document source:",
#     ["Upload Files", "Use Directory"]
# )

# # Initialize session state
# if "retrieval_chain" not in st.session_state:
#     st.session_state.retrieval_chain = None
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "documents_processed" not in st.session_state:
#     st.session_state.documents_processed = False

# # Handle different source options
# if source_option == "Upload Files":
#     uploaded_files = st.file_uploader(
#         "Upload your documents", 
#         type=["pdf", "txt", "ppt", "pptx"],
#         accept_multiple_files=True
#     )
    
#     if uploaded_files:
#         if not st.session_state.documents_processed:
#             with st.spinner("Processing uploaded documents..."):
#                 # Process uploaded files
#                 docs = process_uploaded_files(uploaded_files)
                
#                 if docs:
#                     # Split documents
#                     split_docs = doc_splitter(docs)
                    
#                     # Create vector store
#                     retriever = create_vector_store(split_docs)
                    
#                     if retriever:
#                         # Create QA chain
#                         st.session_state.retrieval_chain = create_qa_chain(retriever)
#                         st.session_state.documents_processed = True
                        
#                         st.success(f"Processed {len(uploaded_files)} files into {len(split_docs)} chunks!")
                        
#                         # Show file details
#                         st.info("Processed files: " + ", ".join([f.name for f in uploaded_files]))
#                     else:
#                         st.error("Failed to create vector store")
#                 else:
#                     st.error("No documents were successfully processed")

# elif source_option == "Use Directory":
#     directory_path = st.text_input("Enter directory path:", value="data/")
    
#     if st.button("Process Directory"):
#         if os.path.exists(directory_path):
#             with st.spinner("Processing documents from directory..."):
#                 # Load documents from directory
#                 docs = load_documents_from_directory(directory_path)
                
#                 if docs:
#                     # Split documents
#                     split_docs = doc_splitter(docs)
                    
#                     # Create vector store
#                     retriever = create_vector_store(split_docs)
                    
#                     if retriever:
#                         # Create QA chain
#                         st.session_state.retrieval_chain = create_qa_chain(retriever)
#                         st.session_state.documents_processed = True
                        
#                         st.success(f"Processed {len(docs)} documents into {len(split_docs)} chunks!")
#                     else:
#                         st.error("Failed to create vector store")
#                 else:
#                     st.warning("No documents found in the directory")
#         else:
#             st.error(f"Directory '{directory_path}' not found")

# # Chat interface
# if st.session_state.documents_processed and st.session_state.retrieval_chain:
#     st.markdown("---")
#     st.subheader("💬 Chat with your documents")
    
#     # Display chat history
#     for message in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.markdown(message['user'])
#         with st.chat_message("assistant"):
#             st.markdown(message['assistant'])
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question about your documents:"):
#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate and display assistant response
#         with st.chat_message("assistant"):
#             try:
#                 with st.spinner("Thinking..."):
#                     response = st.session_state.retrieval_chain.invoke({
#                         "input": prompt
#                     })
#                     answer = response.get('answer', 'No answer generated.')
                    
#                 st.markdown(answer)
                
#                 # Add to chat history
#                 st.session_state.chat_history.append({
#                     "user": prompt,
#                     "assistant": answer
#                 })
                
#             except Exception as e:
#                 error_msg = "⚠️ Sorry, an error occurred while processing your request. Please try again."
#                 st.markdown(error_msg)
#                 st.error(f"Error details: {str(e)}")
    
#     # Clear chat button
#     if st.session_state.chat_history:
#         if st.button("Clear Chat History"):
#             st.session_state.chat_history = []
#             st.rerun()
    
#     # Reset documents button
#     if st.button("Process New Documents"):
#         st.session_state.documents_processed = False
#         st.session_state.retrieval_chain = None
#         st.session_state.chat_history = []
#         st.rerun()

# else:
#     st.info("Please process some documents to start chatting!")
#     st.markdown("""
#     ### How to use:
#     1. Enter your Groq API key in the sidebar
#     2. Choose your document source:
#        - **Upload Files**: Upload PDF, TXT, or PowerPoint files
#        - **Use Directory**: Process all files from a local directory
#     3. Wait for documents to be processed
#     4. Ask questions about your documents
    
#     ### Supported file types:
#     - PDF (.pdf)
#     - Text files (.txt)
#     - PowerPoint (.ppt, .pptx)
#     """)


import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import tempfile
import nest_asyncio

# Fix for asyncio issues
nest_asyncio.apply()

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Chat with Multiple Documents", layout="wide")
st.title("Chat with Multiple Documents")

# Sidebar for Groq API key
st.sidebar.title("🔐 Settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# Check for Groq API key
if not groq_api_key:
    st.sidebar.warning("Please enter your Groq API key to use the app.")
    st.stop()

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

def load_documents_from_directory(dir_path: str):
    """Load documents from directory containing txt, pdf, and pptx files"""
    try:
        # Define the directory containing documents
        txt_loader = DirectoryLoader(
            path=dir_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        
        pdf_loader = DirectoryLoader(
            path=dir_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        
        pptx_loader = DirectoryLoader(
            path=dir_path,
            glob="**/*.pptx",
            loader_cls=UnstructuredPowerPointLoader
        )
        
        # Load documents from each loader
        txt_docs = txt_loader.load()
        pdf_docs = pdf_loader.load()
        pptx_docs = pptx_loader.load()
        
        # Combine all documents into a single list
        all_docs = txt_docs + pdf_docs + pptx_docs
        
        return all_docs
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []

def process_uploaded_files(uploaded_files):
    """Process uploaded files and return documents"""
    all_docs = []
    
    for uploaded_file in uploaded_files:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            # Load based on file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
            elif file_extension == 'txt':
                loader = TextLoader(temp_file_path)
                docs = loader.load()
            elif file_extension in ['ppt', 'pptx']:
                loader = UnstructuredPowerPointLoader(temp_file_path)
                docs = loader.load()
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                continue
            
            # Add source information
            for doc in docs:
                doc.metadata['source_file'] = uploaded_file.name
            
            all_docs.extend(docs)
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return all_docs

def doc_splitter(docs):
    """Split documents into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Increased chunk size for better context
        chunk_overlap=200   # Increased overlap
    )
    
    split_docs = splitter.split_documents(docs)
    
    return split_docs

def create_vector_store(docs):
    """Create vector store from documents"""
    try:
        embedding_model = load_embedding_model()
        if embedding_model is None:
            return None
        
        if not docs:
            st.error("No documents to process")
            return None
            
        db = FAISS.from_documents(documents=docs, embedding=embedding_model)
        retriever = db.as_retriever(search_kwargs={'k': 5})
        
        return retriever
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def create_qa_chain(retriever):
    """Create the QA chain"""
    try:
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template(
            '''
            You are an AI researcher who is an expert in RAG systems.
            Answer any question asked by the user based on the provided context.
            Construct answers in the form of bullet points when appropriate.
            Craft your response only from the provided context.
            If you cannot find any related information from the context, simply say "No relevant context provided."
            Do not hallucinate.
            Always mention the source file when possible.
            
            <context>
            {context}
            </context>
            
            QUESTION: {input}
            '''
        )
        
        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            api_key=groq_api_key
        )
        
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        return retrieval_chain
        
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# Main app interface
st.sidebar.markdown("---")
st.sidebar.title("📁 Document Source")
source_option = st.sidebar.radio(
    "Choose document source:",
    ["Upload Files", "Use Directory", "Skip Upload (Demo Mode)"]
)

# Initialize session state
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Handle different source options
if source_option == "Upload Files":
    uploaded_files = st.file_uploader(
        "Upload your documents", 
        type=["pdf", "txt", "ppt", "pptx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if not st.session_state.documents_processed:
            with st.spinner("Processing uploaded documents..."):
                # Process uploaded files
                docs = process_uploaded_files(uploaded_files)
                
                if docs:
                    # Split documents
                    split_docs = doc_splitter(docs)
                    
                    # Create vector store
                    retriever = create_vector_store(split_docs)
                    
                    if retriever:
                        # Create QA chain
                        st.session_state.retrieval_chain = create_qa_chain(retriever)
                        st.session_state.documents_processed = True
                        
                        st.success(f"Processed {len(uploaded_files)} files into {len(split_docs)} chunks!")
                        
                        # Show file details
                        st.info("Processed files: " + ", ".join([f.name for f in uploaded_files]))
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.error("No documents were successfully processed")

elif source_option == "Use Directory":
    directory_path = st.text_input("Enter directory path:", value="data/")
    
    if st.button("Process Directory"):
        if os.path.exists(directory_path):
            with st.spinner("Processing documents from directory..."):
                # Load documents from directory
                docs = load_documents_from_directory(directory_path)
                
                if docs:
                    # Split documents
                    split_docs = doc_splitter(docs)
                    
                    # Create vector store
                    retriever = create_vector_store(split_docs)
                    
                    if retriever:
                        # Create QA chain
                        st.session_state.retrieval_chain = create_qa_chain(retriever)
                        st.session_state.documents_processed = True
                        
                        st.success(f"Processed {len(docs)} documents into {len(split_docs)} chunks!")
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.warning("No documents found in the directory")
        else:
            st.error(f"Directory '{directory_path}' not found")

elif source_option == "Skip Upload (Demo Mode)":
    st.info("🎯 **Demo Mode Active** - You can test the chat interface without uploading documents!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Start Demo Mode", type="primary"):
            # Create sample documents for demo
            sample_docs = [
                Document(
                    page_content="""
                    Artificial Intelligence (AI) is transforming various industries by automating processes, 
                    improving decision-making, and enabling new capabilities. Machine learning, a subset of AI, 
                    uses algorithms to learn patterns from data and make predictions or decisions without 
                    explicit programming for each task.
                    """,
                    metadata={"source": "AI_Overview.txt", "source_file": "Demo Document 1"}
                ),
                Document(
                    page_content="""
                    Natural Language Processing (NLP) enables computers to understand, interpret, and 
                    generate human language. Applications include chatbots, translation services, 
                    sentiment analysis, and document summarization. Modern NLP uses transformer 
                    architectures like BERT and GPT for state-of-the-art performance.
                    """,
                    metadata={"source": "NLP_Guide.txt", "source_file": "Demo Document 2"}
                ),
                Document(
                    page_content="""
                    RAG (Retrieval-Augmented Generation) systems combine the power of information retrieval 
                    with text generation. They first retrieve relevant documents from a knowledge base, 
                    then use that context to generate accurate and informed responses. This approach 
                    reduces hallucination and improves factual accuracy in AI responses.
                    """,
                    metadata={"source": "RAG_Systems.txt", "source_file": "Demo Document 3"}
                ),
                Document(
                    page_content="""
                    Vector databases store high-dimensional vectors representing data embeddings. 
                    They enable semantic search, where queries find conceptually similar content 
                    rather than exact keyword matches. Popular vector databases include FAISS, 
                    Pinecone, and Weaviate, which support similarity search and clustering operations.
                    """,
                    metadata={"source": "Vector_DB.txt", "source_file": "Demo Document 4"}
                )
            ]
            
            with st.spinner("Setting up demo environment..."):
                # Split documents
                split_docs = doc_splitter(sample_docs)
                
                # Create vector store
                retriever = create_vector_store(split_docs)
                
                if retriever:
                    # Create QA chain
                    st.session_state.retrieval_chain = create_qa_chain(retriever)
                    st.session_state.documents_processed = True
                    
                    st.success(f"✅ Demo mode ready! Created {len(split_docs)} sample chunks about AI and RAG systems.")
                    st.balloons()
                else:
                    st.error("Failed to set up demo mode")
    
    with col2:
        st.markdown("""
        ### 📝 Sample Questions to Try:
        - What is artificial intelligence?
        - How does RAG work?
        - Explain natural language processing
        - What are vector databases?
        - How do machine learning algorithms work?
        """)
    
    if st.session_state.documents_processed:
        st.success("🎉 Demo mode is active! You can now chat with the sample documents below.")

# Chat interface
if st.session_state.documents_processed and st.session_state.retrieval_chain:
    st.markdown("---")
    st.subheader("💬 Chat with your documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message['user'])
        with st.chat_message("assistant"):
            st.markdown(message['assistant'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents:"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
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
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Reset documents button
    if st.button("Process New Documents"):
        st.session_state.documents_processed = False
        st.session_state.retrieval_chain = None
        st.session_state.chat_history = []
        st.rerun()

else:
    st.info("Please process some documents to start chatting!")
    st.markdown("""
    ### How to use:
    1. Enter your Groq API key in the sidebar
    2. Choose your document source:
       - **Upload Files**: Upload PDF, TXT, or PowerPoint files
       - **Use Directory**: Process all files from a local directory
       - **Skip Upload (Demo Mode)**: Try the app with sample documents
    3. Wait for documents to be processed
    4. Ask questions about your documents
    
    ### Supported file types:
    - PDF (.pdf)
    - Text files (.txt)
    - PowerPoint (.ppt, .pptx)
    
    ### Demo Mode:
    Perfect for testing the app without your own documents! Includes sample content about AI, NLP, RAG systems, and vector databases.
    """)