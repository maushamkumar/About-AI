import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Intentify", page_icon="🤖")
st.title("Intentify - Ticket Intent Classifier")

# Load Hugging Face zero-shot classifier
@st.cache_resource  # caches the model so it doesn't reload on every run
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()

# Define intent labels
labels = ["Password Reset", "Login Issue", "Leave Balance"]

# Input from user
ticket_text = st.text_area("Enter your support ticket text here:", height=100)

if st.button("Predict Intent"):
    if ticket_text.strip():
        # Directly call classifier (no FastAPI)
        result = classifier(ticket_text, candidate_labels=labels)
        predicted_label = result['labels'][0]
        confidence = result['scores'][0]

        st.success(f"Predicted Intent: **{predicted_label}**")
        st.info(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Please enter a ticket first.")
