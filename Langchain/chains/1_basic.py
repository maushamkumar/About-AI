from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
# Load environment variables from .env
load_dotenv()

# Create a Groq llama3-8b-8192 model 
model = ChatGroq(model="llama3-8b-8192")

# Define prompt templates (no need for separate Runnable chains)

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."), 
    ("human", "Tell me {joke_count} jokes.")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

# Create the comvined chain using LangChian Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model

# Run the chain 
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output 
print(result)