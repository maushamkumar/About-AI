import os 
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Langsmith tracking 
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2']  = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Simple Q/A Chatbot with llama-4'


prompt = ChatPromptTemplate.from_messages({
    "system": "You are a helpful assistant. Please respond to the user queries",
    "human": "{input}"
})


def generate_response(question, api_key, model_name, temperature, max_tokens):
    llm = ChatGroq(
        model= model_name, 
        groq_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    answer = chain.invoke({"input": question})
    return answer

# Title of the App 
st.title("Simple Q/A Chatbot with llama-4")

# sidebar for settings and API key 
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

# Drop down to select various GROQ supported models 
selected_model = st.sidebar.selectbox(
    'Select a model',
    [
       "deepseek-r1-distill-llama-70b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "gemma2-9b-it",
        "qwen-qwq-32b" 
    ]
)

# Temperature and Max Tokens 
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.6 )
max_tokens = st.sidebar.slider('Max Tokens', min_value=0, max_value=800, value=300)

# Domain selection (optional, you can also let users type this )
domain = st.sidebar.text_input("Enter the domain of your question (e.g Python, AI, Math)", value="general")

# Main interface for user input 
st.write('Go ahead and ask any question')
user_input = st.text_input("You: " )

if user_input:
    if not api_key:
        st.write("Please enter your Groq API Key in the sidebar.")
    else:
        response = generate_response(user_input, api_key, selected_model, temperature, max_tokens)
        st.write("Bot: ", response)
        
else:
    st.write("Please enter a question to get a response.")