# Use lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /main

# Copy project files
COPY . /main

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports: 348000 for FastAPI, 8501 for Streamlit
EXPOSE 8000 8501

# Run both FastAPI and Streamlit
CMD uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run ui.py --server.address=0.0.0.0 --server.port=8501
