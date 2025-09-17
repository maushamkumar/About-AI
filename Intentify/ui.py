import streamlit as st
import requests 

# FastAPI endpoint 
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Intentify", page_icon="🤖")
st.title("Intentify - Ticket Intent Classifier")

ticket_text = st.text_area("Enter your support ticket text here:", height=100)

if st.button("Predict Intent"):
    if ticket_text.strip():
        response = requests.post(API_URL, json={"ticket": ticket_text})
        if response.status_code == 200:
            data = response.json()
            st.success(f"Predicted Intent: **{data['predicted_intent']}**")
            # st.info(f"Confidence: {data['confidence']}")
            
        else:
            st.error("Error calling API. Is FastAPI running?")
    else:
        st.warning("Please enter a ticket first.")
    