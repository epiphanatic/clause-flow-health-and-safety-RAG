"""
WorkSafe NZ AI Assistant - RAG Chain with Claude 4.5 Sonnet
Retrieval-Augmented Generation for HSWA 2015 Q&A with citations
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configuration
VECTORSTORE_PATH = "data/hswa_vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-4-20250514"  # Claude 4.5 Sonnet
TEMPERATURE = 0  # Deterministic for accuracy
TOP_K_CHUNKS = 4  # Number of chunks to retrieve

# Citation-focused prompt template
PROMPT_TEMPLATE = """You are a WorkSafe New Zealand expert assistant specializing in the Health and Safety at Work Act 2015.

Your task is to answer questions based ONLY on the provided context from the Act. You must follow these rules:

1. ALWAYS cite specific sections when making claims (e.g., "According to Section 36...")
2. If the context mentions a page number, include it in your citation
3. If the answer is not in the provided context, say "I don't have enough information in the Act to answer that."
4. Be precise and professional - this is legal/regulatory content
5. Quote directly from the Act when appropriate

Context from HSWA 2015:
{context}

Question: {question}

Answer:"""


def load_vectorstore() -> FAISS:
    """Load the FAISS vectorstore with local embeddings"""
    print("📥 Loading vectorstore...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # Safe: we created this file
    )

    print(f"✅ Vectorstore loaded ({vectorstore.index.ntotal} vectors)")
    return vectorstore


def create_rag_chain(vectorstore: FAISS) -> RetrievalQA:
    """Create RAG chain with Claude 4.5 Sonnet"""
    print(f"🤖 Initializing Claude 4.5 Sonnet...")

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment variables.\n"
            "Please create a .env file with: ANTHROPIC_API_KEY=your_key_here"
        )

    # Initialize Claude LLM
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        anthropic_api_key=api_key,
        max_tokens=2048
    )

    # Create custom prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Configure retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_CHUNKS}
    )

    # Build RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simple concatenation of contexts
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("✅ RAG chain ready!")
    return rag_chain


def ask_question(rag_chain: RetrievalQA, question: str) -> Dict:
    """
    Ask a question and get answer with sources

    Returns:
        {
            'query': str,
            'result': str,
            'source_documents': List[Document]
        }
    """
    print(f"\n❓ Question: {question}")
    print("-" * 80)

    response = rag_chain({"query": question})

    # Display answer
    print(f"\n💬 Answer:\n{response['result']}\n")

    # Display sources
    print("📚 Sources:")
    for i, doc in enumerate(response['source_documents'], 1):
        page = doc.metadata.get('page', 'N/A')
        preview = doc.page_content[:150].replace('\n', ' ')
        print(f"  {i}. Page {page}: {preview}...")

    print("-" * 80)

    return response


def main():
    """Main execution flow"""
    print("=" * 80)
    print("WorkSafe NZ AI Assistant - RAG Chain")
    print("=" * 80)

    try:
        # Step 1: Load vectorstore
        vectorstore = load_vectorstore()

        # Step 2: Create RAG chain
        rag_chain = create_rag_chain(vectorstore)

        # Step 3: Test with sample questions
        test_questions = [
            "What is a PCBU's primary duty of care?",
            "What are the penalties for serious health and safety violations?",
            "Who can be held responsible for workplace safety?",
            "What is the definition of a worker under this Act?",
            "What are the requirements for health and safety representatives?"
        ]

        print("\n" + "=" * 80)
        print("TESTING RAG CHAIN")
        print("=" * 80)

        results = []
        for question in test_questions:
            result = ask_question(rag_chain, question)
            results.append(result)

        print("\n" + "=" * 80)
        print("✅ RAG CHAIN TEST COMPLETE!")
        print("=" * 80)

        return rag_chain, results

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
