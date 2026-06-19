from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()

llm_model = "llama-3.1-8b-instant"
llm = ChatGroq(
    model=llm_model,
    temperature=0,
)

class ChatState(TypedDict): 
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState): 
    messages = state["messages"]
    response = llm.invoke(messages)
    
    return {
        'messages': [response]
    }
    
    
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False) # If this Ture you'll get error because you are going to use multiple thread. 
# Checkpointer 
checkpointer = SqliteSaver(conn=conn)

# Define node 
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


chatbot = graph.compile(checkpointer)

def retrieve_all_threads(): 
    all_threads = set()
    for checkpoint in checkpointer.list(None): 
        all_threads.add(checkpoint.config['configurable']['thread_id'])
        
        return list(all_threads)

# CONFIG = {"configurable": {"thread_id":'thread_id-1'}}
# response = chatbot.invoke(
#                 {"messages": [HumanMessage(content='kcuh toh bolo')]}, 
#                 config=CONFIG

#             )