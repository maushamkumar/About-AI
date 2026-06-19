# """
# Seed MongoDB database with synthetic furniture data using Google Gemini AI
# """
# import os
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# from pymongo import MongoClient
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_mongodb import MongoDBAtlasVectorSearch
# from pydantic import BaseModel, Field
# from langchain_groq import ChatGroq
# from langchain_community.embeddings import OllamaEmbeddings

# # Load environment variables
# load_dotenv()

# # MongoDB client
# client = MongoClient(os.getenv("MONGODB_ATLAS_URI"))


# class ManufacturerAddress(BaseModel):
#     """Manufacturer address schema"""
#     street: str
#     city: str
#     state: str
#     postal_code: str
#     country: str


# class Prices(BaseModel):
#     """Pricing information schema"""
#     full_price: float
#     sale_price: float


# class UserReview(BaseModel):
#     """User review schema"""
#     review_date: str
#     rating: float
#     comment: str


# class Item(BaseModel):
#     """Furniture item schema"""
#     item_id: str
#     item_name: str
#     item_description: str
#     brand: str
#     manufacturer_address: ManufacturerAddress
#     prices: Prices
#     categories: List[str]
#     user_reviews: List[UserReview]
#     notes: str


# # Initialize Gemini chat model
# llm = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0.7,
#     groq_api_key=os.getenv("GROQ_API_KEY")
# )



# # Create parser for list of items
# parser = JsonOutputParser(pydantic_object=List[Item])


# def setup_database_and_collection():
#     """Create database and collection if they don't exist"""
#     print("Setting up database and collection...")
    
#     db = client["inventory_database"]
    
#     # Check if collection exists
#     collections = db.list_collection_names()
    
#     if "items" not in collections:
#         db.create_collection("items")
#         print("Created 'items' collection in 'inventory_database' database")
#     else:
#         print("'items' collection already exists in 'inventory_database' database")


# def create_vector_search_index():
#     """Create vector search index for embeddings"""
#     try:
#         db = client["inventory_database"]
#         collection = db["items"]
        
#         # Drop existing indexes
#         collection.drop_indexes()
        
#         # Define vector search index
#         vector_search_idx = {
#             "name": "vector_index",
#             "type": "vectorSearch",
#             "definition": {
#                 "fields": [
#                     {
#                         "type": "vector",
#                         "path": "embedding",
#                         "numDimensions": 768,
#                         "similarity": "cosine"
#                     }
#                 ]
#             }
#         }
        
#         print("Creating vector search index...")
#         collection.create_search_index(vector_search_idx)
#         print("Successfully created vector search index")
        
#     except Exception as e:
#         print(f"Failed to create vector search index: {e}")


# def generate_synthetic_data() -> List[Dict[str, Any]]:
#     """Generate synthetic furniture data using Gemini AI"""
#     prompt = f"""You are a helpful assistant that generates furniture store item data. 
# Generate 10 furniture store items as a JSON array. Each record should include the following fields:
# - item_id (string): unique identifier
# - item_name (string): name of the furniture
# - item_description (string): detailed description
# - brand (string): brand name
# - manufacturer_address (object): street, city, state, postal_code, country
# - prices (object): full_price (number), sale_price (number)
# - categories (array of strings): furniture categories
# - user_reviews (array of objects): review_date (string), rating (number 1-5), comment (string)
# - notes (string): additional notes

# Ensure variety in the data and realistic values. Return ONLY the JSON array, no markdown formatting.

# Example format:
# [
#   {{
#     "item_id": "FUR001",
#     "item_name": "Modern Sofa",
#     "item_description": "Comfortable 3-seater sofa with soft cushions",
#     "brand": "ComfortPlus",
#     "manufacturer_address": {{
#       "street": "123 Factory Rd",
#       "city": "Los Angeles",
#       "state": "CA",
#       "postal_code": "90001",
#       "country": "USA"
#     }},
#     "prices": {{
#       "full_price": 899.99,
#       "sale_price": 699.99
#     }},
#     "categories": ["Living Room", "Sofas", "Modern"],
#     "user_reviews": [
#       {{
#         "review_date": "2024-01-15",
#         "rating": 4.5,
#         "comment": "Very comfortable and stylish"
#       }}
#     ],
#     "notes": "Available in multiple colors"
#   }}
# ]"""

#     print("Generating synthetic data...")
    
#     response = llm.invoke(prompt)
    
#     # Parse the response content
#     import json
#     content = response.content
    
#     # Remove markdown code blocks if present
#     if "```json" in content:
#         content = content.split("```json")[1].split("```")[0].strip()
#     elif "```" in content:
#         content = content.split("```")[1].split("```")[0].strip()
    
#     data = json.loads(content)
#     return data


# def create_item_summary(item: Dict[str, Any]) -> str:
#     """Create searchable text summary from item data"""
#     manufacturer_details = f"Made in {item['manufacturer_address']['country']}"
#     categories = ", ".join(item['categories'])
    
#     user_reviews = " ".join([
#         f"Rated {review['rating']} on {review['review_date']}: {review['comment']}"
#         for review in item['user_reviews']
#     ])
    
#     basic_info = f"{item['item_name']} {item['item_description']} from the brand {item['brand']}"
#     price = f"At full price it costs: {item['prices']['full_price']} USD, On sale it costs: {item['prices']['sale_price']} USD"
#     notes = item['notes']
    
#     summary = f"{basic_info}. Manufacturer: {manufacturer_details}. Categories: {categories}. Reviews: {user_reviews}. Price: {price}. Notes: {notes}"
    
#     return summary


# def seed_database():
#     """Main function to populate database with AI-generated data"""
#     try:
#         # Connect to MongoDB
#         client.admin.command('ping')
#         print("You successfully connected to MongoDB!")
        
#         # Setup database and collection
#         setup_database_and_collection()
        
#         # Create vector search index
#         create_vector_search_index()
        
#         db = client["inventory_database"]
#         collection = db["items"]
        
#         # Clear existing data
#         collection.delete_many({})
#         print("Cleared existing data from items collection")
        
#         # Generate synthetic data
#         synthetic_data = generate_synthetic_data()
        
#         # Initialize embeddings
#         embeddings = OllamaEmbeddings(model='mxbai-embed-large')
        
#         # Process each item
#         for item in synthetic_data:
#             # Create summary
#             summary = create_item_summary(item)
            
#             # Create document for vector store
#             from langchain_core.documents import Document
#             doc = Document(
#                 page_content=summary,
#                 metadata=item
#             )
            
#             # Add to vector store
#             MongoDBAtlasVectorSearch.from_documents(
#                 documents=[doc],
#                 embedding=embeddings,
#                 collection=collection,
#                 index_name="vector_index",
#                 text_key="embedding_text",
#                 embedding_key="embedding"
#             )
            
#             print(f"Successfully processed & saved record: {item['item_id']}")
        
#         print("Database seeding completed")
        
#     except Exception as e:
#         print(f"Error seeding database: {e}")
#         import traceback
#         traceback.print_exc()
    
#     finally:
#         client.close()


# if __name__ == "__main__":
#     seed_database()







# """
# Seed MongoDB database with synthetic furniture data using Google Gemini AI
# """
# import os
# import json
# import re
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# from pymongo import MongoClient
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_mongodb import MongoDBAtlasVectorSearch
# from pydantic import BaseModel, Field
# from langchain_groq import ChatGroq
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_core.documents import Document

# # Load environment variables
# load_dotenv()

# # MongoDB client
# client = MongoClient(os.getenv("MONGODB_ATLAS_URI"))


# class ManufacturerAddress(BaseModel):
#     """Manufacturer address schema"""
#     street: str
#     city: str
#     state: str
#     postal_code: str
#     country: str


# class Prices(BaseModel):
#     """Pricing information schema"""
#     full_price: float
#     sale_price: float


# class UserReview(BaseModel):
#     """User review schema"""
#     review_date: str
#     rating: float
#     comment: str


# class Item(BaseModel):
#     """Furniture item schema"""
#     item_id: str
#     item_name: str
#     item_description: str
#     brand: str
#     manufacturer_address: ManufacturerAddress
#     prices: Prices
#     categories: List[str]
#     user_reviews: List[UserReview]
#     notes: str


# class ItemList(BaseModel):
#     """Container for list of items"""
#     items: List[Item]


# # Initialize Groq chat model
# llm = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0.7,
#     groq_api_key=os.getenv("GROQ_API_KEY")
# )


# def setup_database_and_collection():
#     """Create database and collection if they don't exist"""
#     print("Setting up database and collection...")
    
#     db = client["inventory_database"]
    
#     # Check if collection exists
#     collections = db.list_collection_names()
    
#     if "items" not in collections:
#         db.create_collection("items")
#         print("Created 'items' collection in 'inventory_database' database")
#     else:
#         print("'items' collection already exists in 'inventory_database' database")


# def create_vector_search_index():
#     """Create vector search index for embeddings"""
#     try:
#         db = client["inventory_database"]
#         collection = db["items"]
        
#         # Drop existing indexes (except _id)
#         for index in collection.list_indexes():
#             if index['name'] != '_id_':
#                 try:
#                     collection.drop_index(index['name'])
#                 except Exception as e:
#                     print(f"Could not drop index {index['name']}: {e}")
        
#         # Define vector search index
#         vector_search_idx = {
#             "name": "vector_index",
#             "type": "vectorSearch",
#             "definition": {
#                 "fields": [
#                     {
#                         "type": "vector",
#                         "path": "embedding",
#                         "numDimensions": 768,
#                         "similarity": "cosine"
#                     }
#                 ]
#             }
#         }
        
#         print("Creating vector search index...")
#         collection.create_search_index(vector_search_idx)
#         print("Successfully created vector search index")
        
#     except Exception as e:
#         print(f"Failed to create vector search index: {e}")


# def clean_json_string(content: str) -> str:
#     """Clean and extract JSON from LLM response"""
#     # Remove markdown code blocks
#     if "```json" in content:
#         content = content.split("```json")[1].split("```")[0].strip()
#     elif "```" in content:
#         content = content.split("```")[1].split("```")[0].strip()
    
#     # Remove any leading/trailing whitespace
#     content = content.strip()
    
#     # Try to find JSON array or object
#     # Look for the first [ or { and last ] or }
#     start_idx = min(
#         content.find('[') if content.find('[') != -1 else len(content),
#         content.find('{') if content.find('{') != -1 else len(content)
#     )
    
#     if content[start_idx] == '[':
#         end_idx = content.rfind(']')
#     else:
#         end_idx = content.rfind('}')
    
#     if start_idx < end_idx:
#         content = content[start_idx:end_idx + 1]
    
#     return content


# def generate_synthetic_data() -> List[Dict[str, Any]]:
#     """Generate synthetic furniture data using Groq AI"""
#     prompt = """You are a helpful assistant that generates furniture store item data. 
# Generate 10 furniture store items as a valid JSON array. Each record must include these exact fields:

# - item_id (string): unique identifier like "FUR001"
# - item_name (string): name of the furniture
# - item_description (string): detailed description
# - brand (string): brand name
# - manufacturer_address (object with): street, city, state, postal_code, country
# - prices (object with): full_price (number), sale_price (number)
# - categories (array of strings): furniture categories
# - user_reviews (array of objects with): review_date (string YYYY-MM-DD), rating (number 1-5), comment (string)
# - notes (string): additional notes

# CRITICAL RULES:
# 1. Return ONLY valid JSON - no markdown, no comments, no extra text
# 2. Use double quotes for all strings and property names
# 3. No trailing commas
# 4. Escape special characters in strings (use \\" for quotes, avoid apostrophes or use \\')
# 5. Make sure all brackets and braces are properly closed

# Example:
# [
#   {
#     "item_id": "FUR001",
#     "item_name": "Modern Sofa",
#     "item_description": "Comfortable 3-seater sofa with soft cushions",
#     "brand": "ComfortPlus",
#     "manufacturer_address": {
#       "street": "123 Factory Rd",
#       "city": "Los Angeles",
#       "state": "CA",
#       "postal_code": "90001",
#       "country": "USA"
#     },
#     "prices": {
#       "full_price": 899.99,
#       "sale_price": 699.99
#     },
#     "categories": ["Living Room", "Sofas", "Modern"],
#     "user_reviews": [
#       {
#         "review_date": "2024-01-15",
#         "rating": 4.5,
#         "comment": "Very comfortable and stylish"
#       }
#     ],
#     "notes": "Available in multiple colors"
#   }
# ]

# Generate 10 different furniture items following this exact structure."""

#     print("Generating synthetic data...")
    
#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             response = llm.invoke(prompt)
#             content = response.content
            
#             # Clean the response
#             cleaned_content = clean_json_string(content)
            
#             # Try to parse JSON
#             data = json.loads(cleaned_content)
            
#             # Validate it's a list
#             if not isinstance(data, list):
#                 if isinstance(data, dict) and 'items' in data:
#                     data = data['items']
#                 else:
#                     raise ValueError("Response is not a list of items")
            
#             # Validate we have items
#             if len(data) == 0:
#                 raise ValueError("No items generated")
            
#             print(f"Successfully generated {len(data)} items")
#             return data
            
#         except json.JSONDecodeError as e:
#             print(f"Attempt {attempt + 1}/{max_retries} - JSON parsing error: {e}")
#             if attempt < max_retries - 1:
#                 print("Retrying...")
#                 continue
#             else:
#                 print(f"Failed to parse JSON after {max_retries} attempts")
#                 print(f"Raw content: {content[:500]}...")
#                 raise
#         except Exception as e:
#             print(f"Attempt {attempt + 1}/{max_retries} - Error: {e}")
#             if attempt < max_retries - 1:
#                 print("Retrying...")
#                 continue
#             else:
#                 raise


# def create_item_summary(item: Dict[str, Any]) -> str:
#     """Create searchable text summary from item data"""
#     manufacturer_details = f"Made in {item['manufacturer_address']['country']}"
#     categories = ", ".join(item['categories'])
    
#     user_reviews = " ".join([
#         f"Rated {review['rating']} on {review['review_date']}: {review['comment']}"
#         for review in item['user_reviews']
#     ])
    
#     basic_info = f"{item['item_name']} {item['item_description']} from the brand {item['brand']}"
#     price = f"At full price it costs: {item['prices']['full_price']} USD, On sale it costs: {item['prices']['sale_price']} USD"
#     notes = item['notes']
    
#     summary = f"{basic_info}. Manufacturer: {manufacturer_details}. Categories: {categories}. Reviews: {user_reviews}. Price: {price}. Notes: {notes}"
    
#     return summary


# def seed_database():
#     """Main function to populate database with AI-generated data"""
#     try:
#         # Connect to MongoDB
#         client.admin.command('ping')
#         print("You successfully connected to MongoDB!")
        
#         # Setup database and collection
#         setup_database_and_collection()
        
#         # Create vector search index
#         create_vector_search_index()
        
#         db = client["inventory_database"]
#         collection = db["items"]
        
#         # Clear existing data
#         collection.delete_many({})
#         print("Cleared existing data from items collection")
        
#         # Generate synthetic data
#         synthetic_data = generate_synthetic_data()
        
#         # Initialize embeddings
#         print("Initializing embeddings model...")
#         embeddings = OllamaEmbeddings(model='mxbai-embed-large')
        
#         # Process each item
#         print(f"Processing {len(synthetic_data)} items...")
#         for idx, item in enumerate(synthetic_data, 1):
#             try:
#                 # Create summary
#                 summary = create_item_summary(item)
                
#                 # Create document for vector store
#                 doc = Document(
#                     page_content=summary,
#                     metadata=item
#                 )
                
#                 # Add to vector store
#                 MongoDBAtlasVectorSearch.from_documents(
#                     documents=[doc],
#                     embedding=embeddings,
#                     collection=collection,
#                     index_name="vector_index",
#                     text_key="embedding_text",
#                     embedding_key="embedding"
#                 )
                
#                 print(f"[{idx}/{len(synthetic_data)}] Successfully processed & saved: {item['item_id']}")
                
#             except Exception as e:
#                 print(f"Error processing item {item.get('item_id', 'unknown')}: {e}")
#                 continue
        
#         print("\n✅ Database seeding completed successfully!")
        
#     except Exception as e:
#         print(f"\n❌ Error seeding database: {e}")
#         import traceback
#         traceback.print_exc()
    
#     finally:
#         client.close()
#         print("MongoDB connection closed")


# if __name__ == "__main__":
#     seed_database()



"""
Seed MongoDB database with synthetic furniture data using Groq AI and Ollama embeddings
"""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_mongodb import MongoDBAtlasVectorSearch
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# MongoDB client
client = MongoClient(os.getenv("MONGODB_ATLAS_URI"))


class ManufacturerAddress(BaseModel):
    """Manufacturer address schema"""
    street: str
    city: str
    state: str
    postal_code: str
    country: str


class Prices(BaseModel):
    """Pricing information schema"""
    full_price: float
    sale_price: float


class UserReview(BaseModel):
    """User review schema"""
    review_date: str
    rating: float
    comment: str


class Item(BaseModel):
    """Furniture item schema"""
    item_id: str
    item_name: str
    item_description: str
    brand: str
    manufacturer_address: ManufacturerAddress
    prices: Prices
    categories: List[str]
    user_reviews: List[UserReview]
    notes: str


# Initialize Groq chat model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Create parser for list of items
parser = JsonOutputParser(pydantic_object=List[Item])


def setup_database_and_collection():
    """Create database and collection if they don't exist"""
    print("Setting up database and collection...")
    
    db = client["inventory_database"]
    
    # Check if collection exists
    collections = db.list_collection_names()
    
    if "items" not in collections:
        db.create_collection("items")
        print("Created 'items' collection in 'inventory_database' database")
    else:
        print("'items' collection already exists in 'inventory_database' database")


def create_vector_search_index():
    """Create vector search index for embeddings"""
    try:
        db = client["inventory_database"]
        collection = db["items"]
        
        # Drop existing indexes
        collection.drop_indexes()
        
        # Define vector search index for mxbai-embed-large (1024 dimensions)
        vector_search_idx = {
            "name": "vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1024,  # Changed from 768 to 1024 for mxbai-embed-large
                        "similarity": "cosine"
                    }
                ]
            }
        }
        
        print("Creating vector search index for 1024 dimensions (mxbai-embed-large)...")
        collection.create_search_index(vector_search_idx)
        print("Successfully created vector search index")
        
    except Exception as e:
        print(f"Failed to create vector search index: {e}")


def generate_synthetic_data() -> List[Dict[str, Any]]:
    """Generate synthetic furniture data using Gemini AI"""
    prompt = f"""You are a helpful assistant that generates furniture store item data. 
Generate 10 furniture store items as a JSON array. Each record should include the following fields:
- item_id (string): unique identifier
- item_name (string): name of the furniture
- item_description (string): detailed description
- brand (string): brand name
- manufacturer_address (object): street, city, state, postal_code, country
- prices (object): full_price (number), sale_price (number)
- categories (array of strings): furniture categories
- user_reviews (array of objects): review_date (string), rating (number 1-5), comment (string)
- notes (string): additional notes

Ensure variety in the data and realistic values. Return ONLY the JSON array, no markdown formatting.

Example format:
[
  {{
    "item_id": "FUR001",
    "item_name": "Modern Sofa",
    "item_description": "Comfortable 3-seater sofa with soft cushions",
    "brand": "ComfortPlus",
    "manufacturer_address": {{
      "street": "123 Factory Rd",
      "city": "Los Angeles",
      "state": "CA",
      "postal_code": "90001",
      "country": "USA"
    }},
    "prices": {{
      "full_price": 899.99,
      "sale_price": 699.99
    }},
    "categories": ["Living Room", "Sofas", "Modern"],
    "user_reviews": [
      {{
        "review_date": "2024-01-15",
        "rating": 4.5,
        "comment": "Very comfortable and stylish"
      }}
    ],
    "notes": "Available in multiple colors"
  }}
]"""

    print("Generating synthetic data...")
    
    response = llm.invoke(prompt)
    
    # Parse the response content
    import json
    content = response.content
    
    # Remove markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    data = json.loads(content)
    return data


def create_item_summary(item: Dict[str, Any]) -> str:
    """Create searchable text summary from item data"""
    manufacturer_details = f"Made in {item['manufacturer_address']['country']}"
    categories = ", ".join(item['categories'])
    
    user_reviews = " ".join([
        f"Rated {review['rating']} on {review['review_date']}: {review['comment']}"
        for review in item['user_reviews']
    ])
    
    basic_info = f"{item['item_name']} {item['item_description']} from the brand {item['brand']}"
    price = f"At full price it costs: {item['prices']['full_price']} USD, On sale it costs: {item['prices']['sale_price']} USD"
    notes = item['notes']
    
    summary = f"{basic_info}. Manufacturer: {manufacturer_details}. Categories: {categories}. Reviews: {user_reviews}. Price: {price}. Notes: {notes}"
    
    return summary


def seed_database():
    """Main function to populate database with AI-generated data"""
    try:
        # Connect to MongoDB
        client.admin.command('ping')
        print("You successfully connected to MongoDB!")
        
        # Setup database and collection
        setup_database_and_collection()
        
        # Create vector search index
        create_vector_search_index()
        
        db = client["inventory_database"]
        collection = db["items"]
        
        # Clear existing data
        collection.delete_many({})
        print("Cleared existing data from items collection")
        
        # Generate synthetic data
        synthetic_data = generate_synthetic_data()
        
        # Initialize Ollama embeddings (1024 dimensions)
        embeddings = OllamaEmbeddings(
            model='mxbai-embed-large'
        )
        
        # Process each item
        for item in synthetic_data:
            # Create summary
            summary = create_item_summary(item)
            
            # Create document for vector store
            from langchain_core.documents import Document
            doc = Document(
                page_content=summary,
                metadata=item
            )
            
            # Add to vector store
            MongoDBAtlasVectorSearch.from_documents(
                documents=[doc],
                embedding=embeddings,
                collection=collection,
                index_name="vector_index",
                text_key="embedding_text",
                embedding_key="embedding"
            )
            
            print(f"Successfully processed & saved record: {item['item_id']}")
        
        print("Database seeding completed")
        
    except Exception as e:
        print(f"Error seeding database: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        client.close()


if __name__ == "__main__":
    seed_database()