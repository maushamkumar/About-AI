from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Create a Groq llama3-8b-8192 model 
model = ChatGroq(model="llama3-8b-8192")

# Invoke the model with a message 
result = model.invoke("What is 81 divided by 9")
print("Full result:")
print(result)
print("Content only")
print(result.content)