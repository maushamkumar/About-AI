import streamlit as st 
from backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

# Utility functions 
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id
    
def reset_chat():
    st.session_state['thread_id'] = generate_thread_id()
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    
def add_thread(thread_id): 
    if thread_id not in st.session_state['chat_thread']: 
        st.session_state['chat_thread'].append(thread_id)


# Create a function. 
# If you'll provide thread_id this function will return whole converstation 
def load_conversation(thread_id):
    return chatbot.get_state(config={"configurable": {"thread_id":thread_id}}).values['messages']


# st.session_state -> Dict. 
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
    

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    
if 'chat_thread' not in st.session_state:
    st.session_state['chat_thread'] = retrieve_all_threads()
    
add_thread(st.session_state['thread_id'])


    
# Sidebar UI
st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button('New Chat'): 
    reset_chat()
st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_thread']: 
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        
        temp_messages = [] 
        
        for msg in messages: 
            if isinstance(msg, HumanMessage): 
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})
        
        st.session_state['message_history'] = temp_messages

# Loading the conversation history
for message in st.session_state['message_history']: 
    with st.chat_message(message['role']):
        st.text(message['content'])
    
    
CONFIG={
    "configurable": {"thread_id":st.session_state['thread_id']},
    "metadata": {
        "thread_id": st.session_state['thread_id']
    }, 
    "run_name": "chat_turn"
        
        
        
        }


# {'role': 'user', 'content': 'Hi'}
# {'role': 'assistant', 'content': 'Hi'}
    
user_input = st.chat_input('Type Here')
if user_input: 
    
    # First add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
    
    with st.chat_message('assistant'): 
        
        answer = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=CONFIG,
                stream_mode = 'messages'
            )
        )
        
    st.session_state['message_history'].append({'role': 'assistant', 'content': answer})