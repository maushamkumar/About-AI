# """
# AI agent for furniture store chatbot using LangGraph and MongoDB
# """
# import os
# import time
# import json
# from typing import Annotated, Literal, TypedDict, Optional
# from datetime import datetime
# from pymongo import MongoClient
# from pymongo.collection import Collection
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.tools import tool
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode
# from langgraph.checkpoint.mongodb import MongoDBSaver
# from langchain_mongodb import MongoDBAtlasVectorSearch
# from pydantic import BaseModel, Field
# from langchain_groq import ChatGroq
# from langchain_community.embeddings import OllamaEmbeddings

# # Global variable to store the MongoDB collection
# _mongodb_collection: Optional[Collection] = None


# def set_mongodb_collection(collection: Collection):
#     """Set the global MongoDB collection"""
#     global _mongodb_collection
#     _mongodb_collection = collection


# def get_mongodb_collection() -> Collection:
#     """Get the global MongoDB collection"""
#     if _mongodb_collection is None:
#         raise ValueError("MongoDB collection not initialized. Call set_mongodb_collection first.")
#     return _mongodb_collection


# def retry_with_backoff(func, max_retries=3):
#     """Utility function to handle API rate limits with exponential backoff"""
#     for attempt in range(1, max_retries + 1):
#         try:
#             return func()
#         except Exception as error:
#             # Check if it's a rate limit error
#             if hasattr(error, 'status_code') and error.status_code == 429 and attempt < max_retries:
#                 delay = min(2 ** attempt, 30)
#                 print(f"Rate limit hit. Retrying in {delay} seconds...")
#                 time.sleep(delay)
#                 continue
#             raise error
#     raise Exception("Max retries exceeded")


# # Define the agent state
# class AgentState(TypedDict):
#     """State for the agent workflow"""
#     messages: Annotated[list[BaseMessage], "The messages in the conversation"]


# # Define the tool input schema
# class ItemLookupInput(BaseModel):
#     """Input for item lookup tool"""
#     query: str = Field(description="The search query")
#     n: int = Field(default=10, description="Number of results to return")


# @tool(args_schema=ItemLookupInput)
# def item_lookup(query: str, n: int = 10) -> str:
#     """Gathers furniture item details from the Inventory database"""
#     try:
#         print(f"Item lookup tool called with query: {query}")
        
#         # Get MongoDB collection from global variable
#         collection = get_mongodb_collection()
        
#         # Check if database has any data
#         total_count = collection.count_documents({})
#         print(f"Total documents in collection: {total_count}")
        
#         if total_count == 0:
#             print("Collection is empty")
#             return json.dumps({
#                 "error": "No items found in inventory",
#                 "message": "The inventory database appears to be empty",
#                 "count": 0
#             })
        
#         # Get sample documents for debugging
#         sample_docs = list(collection.find({}).limit(3))
#         print(f"Sample documents: {len(sample_docs)} items")
        
#         # Configuration for MongoDB Atlas Vector Search
#         embeddings = embeddings = OllamaEmbeddings(model='mxbai-embed-large')
        
#         # Create vector store instance
#         vector_store = MongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name="vector_index",
#             text_key="embedding_text",
#             embedding_key="embedding"
#         )
        
#         print("Performing vector search...")
#         # Perform semantic search
#         results = vector_store.similarity_search_with_score(query, k=n)
#         print(f"Vector search returned {len(results)} results")
        
#         # If vector search returns no results, fall back to text search
#         if len(results) == 0:
#             print("Vector search returned no results, trying text search...")
#             text_results = list(collection.find({
#                 "$or": [
#                     {"item_name": {"$regex": query, "$options": "i"}},
#                     {"item_description": {"$regex": query, "$options": "i"}},
#                     {"categories": {"$regex": query, "$options": "i"}},
#                     {"embedding_text": {"$regex": query, "$options": "i"}}
#                 ]
#             }).limit(n))
            
#             print(f"Text search returned {len(text_results)} results")
#             return json.dumps({
#                 "results": text_results,
#                 "searchType": "text",
#                 "query": query,
#                 "count": len(text_results)
#             }, default=str)
        
#         # Format vector search results
#         formatted_results = [
#             {
#                 "document": doc.metadata,
#                 "score": score,
#                 "content": doc.page_content
#             }
#             for doc, score in results
#         ]
        
#         return json.dumps({
#             "results": formatted_results,
#             "searchType": "vector",
#             "query": query,
#             "count": len(formatted_results)
#         }, default=str)
        
#     except Exception as error:
#         print(f"Error in item lookup: {error}")
#         import traceback
#         traceback.print_exc()
        
#         return json.dumps({
#             "error": "Failed to search inventory",
#             "details": str(error),
#             "query": query
#         })


# def call_agent(client: MongoClient, query: str, thread_id: str) -> str:
#     """Main function to create and run the AI agent"""
#     try:
#         # Database configuration
#         db_name = "inventory_database"
#         db = client[db_name]
#         collection = db["items"]
        
#         # Set the global MongoDB collection
#         set_mongodb_collection(collection)
        
#         # List of available tools
#         tools = [item_lookup]
        
#         # Initialize the AI model
#         model = ChatGroq(
#             model="llama-3.1-8b-instant",
#             temperature=0,
#             max_retries=0,
#             groq_api_key=os.getenv("GROQ_API_KEY")
#         ).bind_tools(tools)
        
#         # Decision function: determines next step
#         def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
#             """Determine if we should continue to tools or end"""
#             messages = state["messages"]
#             last_message = messages[-1]
            
#             # If the AI wants to use tools, go to tools node
#             if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#                 return "tools"
#             return "__end__"
        
#         # Function that calls the AI model
#         def call_model(state: AgentState) -> dict:
#             """Call the model with retry logic"""
#             def _call():
#                 # Create prompt template
#                 prompt = ChatPromptTemplate.from_messages([
#                     (
#                         "system",
#                         """You are a helpful E-commerce Chatbot Agent for a furniture store. 

# IMPORTANT: You have access to an item_lookup tool that searches the furniture inventory database. ALWAYS use this tool when customers ask about furniture items, even if the tool returns errors or empty results.

# When using the item_lookup tool:
# - If it returns results, provide helpful details about the furniture items
# - If it returns an error or no results, acknowledge this and offer to help in other ways
# - If the database appears to be empty, let the customer know that inventory might be being updated

# Current time: {time}"""
#                     ),
#                     MessagesPlaceholder(variable_name="messages")
#                 ])
                
#                 # Format the prompt
#                 formatted_prompt = prompt.format_messages(
#                     time=datetime.now().isoformat(),
#                     messages=state["messages"]
#                 )
                
#                 # Call the model
#                 result = model.invoke(formatted_prompt)
#                 return {"messages": [result]}
            
#             return retry_with_backoff(_call)
        
#         # Build the workflow graph
#         workflow = StateGraph(AgentState)
        
#         # Add nodes
#         workflow.add_node("agent", call_model)
#         workflow.add_node("tools", ToolNode(tools))
        
#         # Add edges
#         workflow.set_entry_point("agent")
#         workflow.add_conditional_edges(
#             "agent",
#             should_continue,
#             {
#                 "tools": "tools",
#                 "__end__": END
#             }
#         )
#         workflow.add_edge("tools", "agent")
        
#         # Initialize checkpointer
#         checkpointer = MongoDBSaver(client=client, db_name=db_name)
        
#         # Compile the workflow
#         app = workflow.compile(checkpointer=checkpointer)
        
#         # Execute the workflow
#         config = {
#             "configurable": {"thread_id": thread_id},
#             "recursion_limit": 15
#         }
        
#         final_state = app.invoke(
#             {"messages": [HumanMessage(content=query)]},
#             config=config
#         )
        
#         # Extract the final response
#         response = final_state["messages"][-1].content
#         print(f"Agent response: {response}")
        
#         return response
        
#     except Exception as error:
#         print(f"Error in call_agent: {error}")
#         import traceback
#         traceback.print_exc()
        
#         if hasattr(error, 'status_code'):
#             if error.status_code == 429:
#                 raise Exception("Service temporarily unavailable due to rate limits. Please try again in a minute.")
#             elif error.status_code == 401:
#                 raise Exception("Authentication failed. Please check your API configuration.")
        
#         raise Exception(f"Agent failed: {str(error)}")




"""
AI agent for furniture store chatbot using LangGraph, MongoDB, Groq, and Ollama
"""
import os
import time
import json
from typing import Annotated, Literal, TypedDict, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_mongodb import MongoDBAtlasVectorSearch
from pydantic import BaseModel, Field

# Global variable to store the MongoDB collection
_mongodb_collection: Optional[Collection] = None


def set_mongodb_collection(collection: Collection):
    """Set the global MongoDB collection"""
    global _mongodb_collection
    _mongodb_collection = collection


def get_mongodb_collection() -> Collection:
    """Get the global MongoDB collection"""
    if _mongodb_collection is None:
        raise ValueError("MongoDB collection not initialized. Call set_mongodb_collection first.")
    return _mongodb_collection


def retry_with_backoff(func, max_retries=3):
    """Utility function to handle API rate limits with exponential backoff"""
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as error:
            # Check if it's a rate limit error
            if hasattr(error, 'status_code') and error.status_code == 429 and attempt < max_retries:
                delay = min(2 ** attempt, 30)
                print(f"Rate limit hit. Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            raise error
    raise Exception("Max retries exceeded")


# Define the agent state
class AgentState(TypedDict):
    """State for the agent workflow"""
    messages: Annotated[list[BaseMessage], "The messages in the conversation"]


# Define the tool input schema
class ItemLookupInput(BaseModel):
    """Input for item lookup tool"""
    query: str = Field(description="The search query")
    n: int = Field(default=10, description="Number of results to return")


@tool(args_schema=ItemLookupInput)
def item_lookup(query: str, n: int = 10) -> str:
    """Gathers furniture item details from the Inventory database"""
    try:
        print(f"Item lookup tool called with query: {query}")
        
        # Get MongoDB collection from global variable
        collection = get_mongodb_collection()
        
        # Check if database has any data
        total_count = collection.count_documents({})
        print(f"Total documents in collection: {total_count}")
        
        if total_count == 0:
            print("Collection is empty")
            return json.dumps({
                "error": "No items found in inventory",
                "message": "The inventory database appears to be empty",
                "count": 0
            })
        
        # Get sample documents for debugging
        sample_docs = list(collection.find({}).limit(3))
        print(f"Sample documents: {len(sample_docs)} items")
        
        # Configuration for MongoDB Atlas Vector Search
        embeddings = embeddings = OllamaEmbeddings(model='mxbai-embed-large')
        
        # Create vector store instance
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",
            text_key="embedding_text",
            embedding_key="embedding"
        )
        
        print("Performing vector search...")
        # Perform semantic search
        results = vector_store.similarity_search_with_score(query, k=n)
        print(f"Vector search returned {len(results)} results")
        
        # If vector search returns no results, fall back to text search
        if len(results) == 0:
            print("Vector search returned no results, trying text search...")
            text_results = list(collection.find({
                "$or": [
                    {"item_name": {"$regex": query, "$options": "i"}},
                    {"item_description": {"$regex": query, "$options": "i"}},
                    {"categories": {"$regex": query, "$options": "i"}},
                    {"embedding_text": {"$regex": query, "$options": "i"}}
                ]
            }).limit(n))
            
            print(f"Text search returned {len(text_results)} results")
            return json.dumps({
                "results": text_results,
                "searchType": "text",
                "query": query,
                "count": len(text_results)
            }, default=str)
        
        # Format vector search results
        formatted_results = [
            {
                "document": doc.metadata,
                "score": score,
                "content": doc.page_content
            }
            for doc, score in results
        ]
        
        return json.dumps({
            "results": formatted_results,
            "searchType": "vector",
            "query": query,
            "count": len(formatted_results)
        }, default=str)
        
    except Exception as error:
        print(f"Error in item lookup: {error}")
        import traceback
        traceback.print_exc()
        
        return json.dumps({
            "error": "Failed to search inventory",
            "details": str(error),
            "query": query
        })


def call_agent(client: MongoClient, query: str, thread_id: str) -> str:
    """Main function to create and run the AI agent"""
    try:
        # Database configuration
        db_name = "inventory_database"
        db = client[db_name]
        collection = db["items"]
        
        # Set the global MongoDB collection
        set_mongodb_collection(collection)
        
        # List of available tools
        tools = [item_lookup]
        
        # Initialize the Groq AI model
        model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_retries=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        ).bind_tools(tools)
        
        # Decision function: determines next step
        def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
            """Determine if we should continue to tools or end"""
            messages = state["messages"]
            last_message = messages[-1]
            
            # If the AI wants to use tools, go to tools node
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return "__end__"
        
        # Function that calls the AI model
        def call_model(state: AgentState) -> dict:
            """Call the model with retry logic"""
            def _call():
                # Create prompt template
                prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        """You are a helpful E-commerce Chatbot Agent for a furniture store. 

IMPORTANT TOOL USAGE RULES:
1. You have access to an item_lookup tool that searches the furniture inventory database
2. Use this tool ONCE per user question - do not retry if it returns an error
3. If the tool returns an error or no results, acknowledge this politely and offer general help
4. Never call the same tool multiple times for the same query

Response Guidelines:
- If tool succeeds: Provide helpful details about the furniture items found
- If tool fails or returns empty: Say "I'm having trouble accessing the inventory right now, but I'm here to help with any questions about furniture"
- If database is empty: "Our inventory is currently being updated. Please check back soon"
- Always respond to the user, even if the tool fails

Current time: {time}"""
                    ),
                    MessagesPlaceholder(variable_name="messages")
                ])
                
                # Format the prompt
                formatted_prompt = prompt.format_messages(
                    time=datetime.now().isoformat(),
                    messages=state["messages"]
                )
                
                # Call the model
                result = model.invoke(formatted_prompt)
                return {"messages": [result]}
            
            return retry_with_backoff(_call)
        
        # Build the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(tools))
        
        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "__end__": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Initialize checkpointer
        checkpointer = MongoDBSaver(client=client, db_name=db_name)
        
        # Compile the workflow
        app = workflow.compile(checkpointer=checkpointer)
        
        # Execute the workflow
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 25  # Increased from 15 to handle edge cases
        }
        
        final_state = app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        
        # Extract the final response
        response = final_state["messages"][-1].content
        print(f"Agent response: {response}")
        
        return response
        
    except Exception as error:
        print(f"Error in call_agent: {error}")
        import traceback
        traceback.print_exc()
        
        if hasattr(error, 'status_code'):
            if error.status_code == 429:
                raise Exception("Service temporarily unavailable due to rate limits. Please try again in a minute.")
            elif error.status_code == 401:
                raise Exception("Authentication failed. Please check your API configuration.")
        
        raise Exception(f"Agent failed: {str(error)}")