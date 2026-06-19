# """
# FastAPI server for furniture store chatbot
# """
# import os
# import ssl
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from pymongo import MongoClient
# from agent import call_agent
# import time

# # Load environment variables
# load_dotenv()

# # Create FastAPI application
# app = FastAPI(title="LangGraph Agent Server")

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # MongoDB client with SSL/TLS configuration
# # Option 1: Using certifi (recommended for production)
# try:
#     import certifi
#     client = MongoClient(
#         os.getenv("MONGODB_ATLAS_URI"),
#         tlsCAFile=certifi.where(),
#         serverSelectionTimeoutMS=10000,
#         connectTimeoutMS=20000,
#         socketTimeoutMS=20000
#     )
# except ImportError:
#     # Option 2: Allow invalid certificates (development only!)
#     print("Warning: certifi not found, using tlsAllowInvalidCertificates")
#     client = MongoClient(
#         os.getenv("MONGODB_ATLAS_URI"),
#         tls=True,
#         tlsAllowInvalidCertificates=True,
#         serverSelectionTimeoutMS=10000,
#         connectTimeoutMS=20000,
#         socketTimeoutMS=20000
#     )


# # Request/Response models
# class ChatRequest(BaseModel):
#     message: str


# class ChatResponse(BaseModel):
#     threadId: str = None
#     response: str


# class ContinueChatRequest(BaseModel):
#     message: str


# @app.on_event("startup")
# async def startup_event():
#     """Initialize database connection on startup"""
#     try:
#         # Verify MongoDB connection
#         client.admin.command('ping')
#         print("You successfully connected to MongoDB!")
#     except Exception as e:
#         print(f"Error connecting to MongoDB: {e}")
#         raise


# @app.on_event("shutdown")
# async def shutdown_event():
#     """Close database connection on shutdown"""
#     client.close()


# @app.get("/")
# async def root():
#     """Health check endpoint"""
#     return {"message": "LangGraph Agent Server"}


# @app.post("/chat", response_model=ChatResponse)
# async def start_chat(request: ChatRequest):
#     """Start a new conversation"""
#     initial_message = request.message
#     thread_id = str(int(time.time() * 1000))  # Generate unique thread ID
    
#     print(f"Starting new chat with message: {initial_message}")
    
#     try:
#         response = call_agent(client, initial_message, thread_id)
#         return ChatResponse(threadId=thread_id, response=response)
#     except Exception as error:
#         print(f"Error starting conversation: {error}")
#         raise HTTPException(status_code=500, detail="Internal server error")


# @app.post("/chat/{thread_id}", response_model=ChatResponse)
# async def continue_chat(thread_id: str, request: ContinueChatRequest):
#     """Continue an existing conversation"""
#     message = request.message
    
#     print(f"Continuing chat {thread_id} with message: {message}")
    
#     try:
#         response = call_agent(client, message, thread_id)
#         return ChatResponse(response=response)
#     except Exception as error:
#         print(f"Error in chat: {error}")
#         raise HTTPException(status_code=500, detail="Internal server error")


# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)



"""
FastAPI server for furniture store chatbot
"""
import os
import ssl
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from agent import call_agent
import time

# Load environment variables
load_dotenv()

# Create FastAPI application
app = FastAPI(title="LangGraph Agent Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client with SSL/TLS configuration
# Option 1: Using certifi (recommended for production)
try:
    import certifi
    client = MongoClient(
        os.getenv("MONGODB_ATLAS_URI"),
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000
    )
except ImportError:
    # Option 2: Allow invalid certificates (development only!)
    print("Warning: certifi not found, using tlsAllowInvalidCertificates")
    client = MongoClient(
        os.getenv("MONGODB_ATLAS_URI"),
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000
    )


# Request/Response models
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    threadId: str = None
    response: str


class ContinueChatRequest(BaseModel):
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    try:
        # Verify MongoDB connection
        client.admin.command('ping')
        print("You successfully connected to MongoDB!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    client.close()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "LangGraph Agent Server"}


@app.post("/chat", response_model=ChatResponse)
async def start_chat(request: ChatRequest):
    """Start a new conversation"""
    initial_message = request.message
    thread_id = str(int(time.time() * 1000))  # Generate unique thread ID
    
    print(f"Starting new chat with message: {initial_message}")
    
    try:
        response = call_agent(client, initial_message, thread_id)
        return ChatResponse(threadId=thread_id, response=response)
    except Exception as error:
        print(f"Error starting conversation: {error}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/chat/{thread_id}", response_model=ChatResponse)
async def continue_chat(thread_id: str, request: ContinueChatRequest):
    """Continue an existing conversation"""
    message = request.message
    
    print(f"Continuing chat {thread_id} with message: {message}")
    
    try:
        response = call_agent(client, message, thread_id)
        return ChatResponse(response=response)
    except Exception as error:
        print(f"Error in chat: {error}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)