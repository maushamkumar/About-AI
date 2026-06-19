import streamlit as st 
from langGraph_backend import chatbot
from langchain_core.messages import HumanMessage

    
# st.session_state -> Dict. 
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
    



# Loading the conversation history
for message in st.session_state['message_history']: 
    with st.chat_message(message['role']):
        st.text(message['content'])
    

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
                config={"configurable": {"thread_id": "1"}},
                stream_mode = 'messages'
            )
        )
        
    st.session_state['message_history'].append({'role': 'assistant', 'content': answer})