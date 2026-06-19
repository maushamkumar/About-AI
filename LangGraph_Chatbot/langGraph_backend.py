from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

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
    
# Checkpointer 
checkpointer = InMemorySaver()

# Define node 
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


chatbot = graph.compile(checkpointer)

# stream = chatbot.stream(
#     {"messages": [HumanMessage(content='what is the recipe to make pasta')]}, 
#     config={"configurable": {"thread_id": "1"}},
#     stream_mode = 'messages'
# ) # We'll get output as generator. 


# print(type(stream))
# Type: Generator. with the help of generate we have print the output token by token 
# Whenever you want to print generator you have to looping because after all it's an iterator. 

# In stream we get two output. 
# 1. message_chunk and metadata. 

# for message_chunk, metadata in chatbot.stream(
#     {"messages": [HumanMessage(content='what is the recipe to make pasta')]}, 
#     config={"configurable": {"thread_id": "1"}},
#     stream_mode = 'messages'
# ):
#     if message_chunk.content: 
#         print(message_chunk.content, end=" ", flush=True)

# CONFIG = {"configurable": {"thread_id":'thread_id-1'}}
# response = chatbot.invoke(
#                 {"messages": [HumanMessage(content='kcuh toh bolo')]}, 
#                 config=CONFIG

#             )

# print(chatbot.get_state(config=CONFIG).values['messages'])