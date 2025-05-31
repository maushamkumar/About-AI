# import streamlit as st 
# from langchain_community.document_loaders import DirectoryLoader,TextLoader, PyPDFLoader
# from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import os 
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_core.documents import Document
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_groq import ChatGroq
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from pathlib import Path
# from langchain.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain


# st.set_page_config(page_title="Chat with Multiple Documents", layout="wide")
# st.title("Chat with Multiple Documents")    
# # Sidebar for Groq API key 
# st.sidebar.title("🔐 Settings")
# groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# # Check for Groq API key
# if not groq_api_key:
#     st.sidebar.warning("Please enter your Groq API key to use the app.")
#     st.stop()
    
    
# def Text_Loader(dir:Path):
#     # Define the directory containing documents 

#     txt_loader = DirectoryLoader(
#         path=dir,
#         glob="**/*.txt",
#         loader_cls=TextLoader
#     )


#     pdf_loader = DirectoryLoader(
#         path=dir,
#         glob="**/*.pdf",
#         loader_cls=PyPDFLoader  
#     )


#     pptx_loader = DirectoryLoader( 
#         path=dir,
#         glob="**/*.pptx",
#         loader_cls=UnstructuredPowerPointLoader
#     )


#     # Load documents from each loader 

#     txt_docs = txt_loader.load()
#     pdf_docs = pdf_loader.load()
#     pptx_docs =pptx_loader.load()


#     # Combine all documents into a single list
#     all_docs = txt_docs + pdf_docs + pptx_docs
    
#     return all_docs

# def doc_splitter(docs):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=400,        # each chunk will be ~1000 characters
#         chunk_overlap=80       # 200 characters will overlap between chunks
#     )

#     docs = splitter.split_documents(docs)
    
#     return docs

# directory_path = 'data/'
# docs = Text_Loader(directory_path)
# docs = doc_splitter(docs)
# embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# db = FAISS.from_documents(documents=docs, embedding=embedding_model)
# retriver = db.as_retriever(search_kwargs={'k': 5})

# prompt = ChatPromptTemplate.from_template(
#     '''
#     Your a an AI researcher who is an expert in RAG systems.
#     Answer any question asked by the user.
#     construct answers in the form of bullet points
#     Craft your response only from the provided context only.
#     If you cannot find any related information from the context, simply say no context provied.
#     Do not hallucinate.
    
#     <context>
#     {context}
#     </context>
    
#     QUESTION:{question}
#     '''
# )
# llm = ChatGroq(
#     model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#     temperature=0,
#     api_key="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"
# )
# document_chain = create_stuff_documents_chain(llm, prompt) 

# user_query = st.text_input("Ask a question about the documents:")
# if user_query:
#     with st.spinner("Processing your query..."):
#         retrived_docs = retriver.invoke(user_query)
#         response = document_chain.invoke({
#             "context": retrived_docs,
#             "question": user_query
#         })
#         st.write(response)


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

@st.cache_data
def Text_Loader(dir_path: str):
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

@st.cache_data
def doc_splitter(_docs):
    """Split documents into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,    # each chunk will be ~400 characters
        chunk_overlap=80   # 80 characters will overlap between chunks
    )
    
    docs = splitter.split_documents(_docs)
    
    return docs

# Initialize the system
@st.cache_resource
def initialize_system():
    """Initialize the RAG system with document loading and embedding"""
    directory_path = 'data/'
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        st.error(f"Directory '{directory_path}' not found. Please create it and add your documents.")
        return None, None
    
    # Load and process documents
    docs = Text_Loader(directory_path)
    
    if not docs:
        st.warning("No documents found in the data directory.")
        return None, None
    
    docs = doc_splitter(docs)
    
    # Create embeddings and vector store
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(documents=docs, embedding=embedding_model)
    retriever = db.as_retriever(search_kwargs={'k': 5})
    
    return retriever, len(docs)

# Initialize the system
retriever, num_chunks = initialize_system()

if retriever is None:
    st.stop()

st.success(f"✅ System initialized with {num_chunks} document chunks")

# Create the prompt template
prompt = ChatPromptTemplate.from_template(
    '''
    You are an AI researcher who is an expert in RAG systems.
    Answer any question asked by the user.
    Construct answers in the form of bullet points.
    Craft your response only from the provided context.
    If you cannot find any related information from the context, simply say "No relevant context provided."
    Do not hallucinate.
    
    <context>
    {context}
    </context>
    
    QUESTION: {question}
    '''
)

# Initialize the LLM with the provided API key
try:
    llm = ChatGroq(
        model="llama3-8b-8192",  # Using a more commonly available model
        temperature=0,
        api_key=groq_api_key
    )
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # User input
    user_query = st.text_input("Ask a question about the documents:")
    
    if user_query:
        with st.spinner("Processing your query..."):
            try:
                # Retrieve relevant documents
                retrieved_docs = retriever.invoke(user_query)
                
                # Generate response
                response = document_chain.invoke({
                    "context": retrieved_docs,
                    "question": user_query
                })
                
                # Display response
                st.write("### Response:")
                st.write(response)
                
                # Show retrieved documents (optional)
                with st.expander("View Retrieved Context"):
                    for i, doc in enumerate(retrieved_docs):
                        st.write(f"**Document {i+1}:**")
                        st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        st.write("---")
                        
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                
except Exception as e:
    st.error(f"Error initializing the language model: {str(e)}")
    st.info("Please check your Groq API key and try again.")