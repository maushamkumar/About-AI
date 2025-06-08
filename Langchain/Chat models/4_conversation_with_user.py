from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

# Create a Groq llama3-8b-8192 model 
model = ChatGroq(model="llama3-8b-8192")

# Use list tto store messages 
chat_history = []

# set an initial message 
system_message = SystemMessage(content="you are a helpful AI assistant.")
chat_history.append(system_message)

# Chat loop 
while True:
    query = input("You: ")
    if query.lower() == 'exit':
        break
    # Add user message
    chat_history.append(HumanMessage(content=query)) 
    
    # Get AI response using history 
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response)) # Add AI message 
    
    print(f"AI: {response}")
    
print("------Message History ---------")
print(chat_history)
    