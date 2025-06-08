from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Create a Groq llama3-8b-8192 model 
model = ChatGroq(model="llama3-8b-8192")

# SystemMessage:
#   Message for priming AI behaviour, usually passed in as the first of a sequenc of input messages. 
# HummanMesages:
#   Message from a human to the AI model. 
messages = [
    SystemMessage(content="Solve the following math problems"), 
    HumanMessage(content="What is 81 divided by 9?")
]

# Inovke the model with message 
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")

# AIMessage:
#   Message from an AI. 
messages = [
    SystemMessage(content="Solve the following math problems"), 
    HumanMessage(content="What is 81 divided by 9?"), 
    AIMessage(content="81 divided by 9 is 9"), 
    HumanMessage(content="What is 10 times 5?")
    
]
# Invoke the model with messages 
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")
