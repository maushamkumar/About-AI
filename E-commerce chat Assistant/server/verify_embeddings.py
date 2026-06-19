"""
Verify Ollama embeddings and MongoDB vector index configuration
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
import certifi

load_dotenv()

print("=" * 60)
print("OLLAMA + GROQ CONFIGURATION VERIFICATION")
print("=" * 60)

# Test Ollama embeddings
print("\n1. Testing Ollama Embeddings...")
try:
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    test_embedding = embeddings.embed_query("test")
    print(f"✅ Ollama 'mxbai-embed-large': {len(test_embedding)} dimensions")
    
    if len(test_embedding) == 1024:
        print("✅ CORRECT: 1024 dimensions")
    else:
        print(f"⚠️  WARNING: Expected 1024, got {len(test_embedding)}")
except Exception as e:
    print(f"❌ Ollama connection failed: {e}")
    print("\n🔧 Fix:")
    print("   1. Install Ollama: https://ollama.ai")
    print("   2. Run: ollama pull mxbai-embed-large")
    print("   3. Start server: ollama serve")

# Test Groq API
print("\n2. Testing Groq API...")
try:
    from langchain_groq import ChatGroq
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY not found in .env")
    elif not groq_key.startswith("gsk_"):
        print("⚠️  WARNING: Groq API key should start with 'gsk_'")
    else:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_key,
            max_retries=1
        )
        response = llm.invoke("Say 'test'")
        print(f"✅ Groq API working: {response.content[:50]}...")
except Exception as e:
    print(f"❌ Groq API failed: {e}")
    print("\n🔧 Fix:")
    print("   1. Get API key: https://console.groq.com")
    print("   2. Add to .env: GROQ_API_KEY=gsk_your_key_here")

# Check MongoDB collection
print("\n3. Checking MongoDB Collection...")
try:
    client = MongoClient(
        os.getenv("MONGODB_ATLAS_URI"),
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=5000
    )
    
    db = client["inventory_database"]
    collection = db["items"]
    
    # Get document count
    count = collection.count_documents({})
    print(f"✅ Collection has {count} documents")
    
    # Check for embedding field
    sample = collection.find_one({"embedding": {"$exists": True}})
    if sample:
        embedding_dim = len(sample.get("embedding", []))
        print(f"✅ Existing embeddings have {embedding_dim} dimensions")
        
        if embedding_dim == 1024:
            print("✅ MATCH: Perfect for mxbai-embed-large (1024D)")
        elif embedding_dim == 768:
            print("❌ MISMATCH: Embeddings are 768D (Google), need 1024D (Ollama)")
            print("\n🔧 Fix: Re-seed database")
            print("   python seed_database.py")
        else:
            print(f"⚠️  UNKNOWN: Unexpected dimension count: {embedding_dim}")
    else:
        print("⚠️  No documents with embeddings found")
        print("🔧 Fix: Seed the database")
        print("   python seed_database.py")
    
    # List search indexes
    print("\n4. Checking Vector Search Indexes...")
    try:
        indexes = list(collection.list_search_indexes())
        if indexes:
            for idx in indexes:
                print(f"\nIndex: {idx.get('name', 'unnamed')}")
                print(f"  Type: {idx.get('type', 'unknown')}")
                if 'latestDefinition' in idx:
                    definition = idx['latestDefinition']
                    fields = definition.get('fields', [])
                    for field in fields:
                        if field.get('type') == 'vector':
                            dims = field.get('numDimensions')
                            print(f"  Vector field: {field.get('path')}")
                            print(f"  Dimensions: {dims}")
                            print(f"  Similarity: {field.get('similarity')}")
                            
                            if dims == 1024:
                                print("  ✅ CORRECT for Ollama mxbai-embed-large")
                            elif dims == 768:
                                print("  ❌ WRONG: This is for Google embeddings")
                                print("  🔧 Fix: Re-seed database to recreate index")
        else:
            print("❌ No search indexes found")
            print("\n🔧 Fix: Run seed_database.py to create index")
    except Exception as e:
        print(f"❌ Could not list indexes: {e}")
    
    client.close()
    
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")

