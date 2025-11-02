"""
Test script to verify vectorstore functionality
"""

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

VECTORSTORE_PATH = "data/hswa_vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def test_vectorstore():
    """Test the vectorstore with sample queries"""
    print("🔍 Testing Vectorstore...")

    # Load local embeddings (same model used for building)
    print(f"📥 Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load vectorstore
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings
    )

    # Test queries
    test_queries = [
        "What is a PCBU's primary duty of care?",
        "What are the penalties for health and safety violations?",
        "Who is responsible for workplace safety?"
    ]

    print("\n" + "=" * 60)
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-" * 60)

        # Retrieve top 3 relevant chunks
        results = vectorstore.similarity_search(query, k=3)

        for i, doc in enumerate(results, 1):
            print(f"\n📄 Result {i} (Page {doc.metadata.get('page', 'N/A')}):")
            print(f"   {doc.page_content[:200]}...")

    print("\n" + "=" * 60)
    print("✅ Vectorstore test complete!")

if __name__ == "__main__":
    test_vectorstore()
