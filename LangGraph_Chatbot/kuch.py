from langchain_core.messages import HumanMessage
from langGraph_backend import chatbot

state = {"messages": [HumanMessage(content="How are you?")]}

# include config with a thread_id
result = chatbot.invoke(state, config={"configurable": {"thread_id": "1"}})

print(result["messages"][-1].content)
