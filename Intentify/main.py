from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# Initialize FastAPI 
app = FastAPI()

# Define requrest schema
class TicketRequest(BaseModel):
    ticket: str
    

# Load Hugging Face model 
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Labels. (intents)
labels = ["Password Reset", "Login Issue", "Leave Balance"]

@app.post("/predict")
def predict_intent(request: TicketRequest):
    result = classifier(request.ticket, candidate_labels=labels)
    predicted_label = result['labels'][0]
    confidence = result['scores'][0]
    return {
        "ticket": request.ticket,
        "predicted_intent": predicted_label,
        "confidence": confidence
    }
    
if __name__ == "__main__":
    # Run FastAPI on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)