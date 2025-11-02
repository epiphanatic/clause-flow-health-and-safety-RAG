"""
WorkSafe NZ AI Assistant - Vector Store Builder
Loads HSWA 2015 PDF, chunks it, and creates FAISS index with embeddings
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configuration
PDF_PATH = "hswa-docs/Health and Safety at Work Act 2015.pdf"
VECTORSTORE_PATH = "data/hswa_vectorstore"
CHUNK_SIZE = 500  # characters (approximates tokens)
CHUNK_OVERLAP = 50  # characters
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local model, fast and efficient

def load_pdf_with_metadata(pdf_path: str) -> List[Document]:
    """Load PDF and extract text with page metadata"""
    print(f"📄 Loading PDF from: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"✅ Loaded {len(documents)} pages from PDF")

    # Enhance metadata
    for i, doc in enumerate(documents):
        doc.metadata['page'] = i + 1
        doc.metadata['source'] = 'Health and Safety at Work Act 2015'
        # Extract section info if present in page text (basic heuristic)
        if 'Section' in doc.page_content[:100]:
            doc.metadata['has_section'] = True

    return documents

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks with overlap"""
    print(f"✂️  Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks")

    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_size'] = len(chunk.page_content)

    return chunks

def create_vectorstore(chunks: List[Document]) -> FAISS:
    """Generate embeddings and create FAISS index using local model"""
    print(f"🧠 Loading local embedding model: {EMBEDDING_MODEL}")
    print(f"   (This will download ~80MB on first run, then cached locally)")

    # Initialize local embeddings (no API key needed!)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create FAISS vectorstore
    print(f"🏗️  Building FAISS index with {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"✅ FAISS index created with {len(chunks)} vectors")

    return vectorstore

def save_vectorstore(vectorstore: FAISS, chunks: List[Document], output_path: str):
    """Save FAISS index and metadata to disk"""
    print(f"💾 Saving vectorstore to: {output_path}")

    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    vectorstore.save_local(output_path)

    # Save chunk metadata separately for inspection
    metadata_path = Path(output_path) / "chunks_metadata.pkl"
    chunk_metadata = [
        {
            'chunk_id': chunk.metadata.get('chunk_id'),
            'page': chunk.metadata.get('page'),
            'chunk_size': chunk.metadata.get('chunk_size'),
            'preview': chunk.page_content[:100] + "..."
        }
        for chunk in chunks
    ]

    with open(metadata_path, 'wb') as f:
        pickle.dump(chunk_metadata, f)

    print(f"✅ Vectorstore saved successfully")
    print(f"   - FAISS index: {output_path}/index.faiss")
    print(f"   - Metadata: {metadata_path}")

def print_statistics(chunks: List[Document]):
    """Print statistics about the chunked data"""
    print("\n📊 Vectorstore Statistics:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Avg chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
    print(f"   Pages covered: {len(set(c.metadata['page'] for c in chunks))}")

    # Sample chunk preview
    print(f"\n📝 Sample Chunk (ID: {chunks[0].metadata['chunk_id']}):")
    print(f"   Page: {chunks[0].metadata['page']}")
    print(f"   Content preview: {chunks[0].page_content[:150]}...")

def main():
    """Main execution flow"""
    print("=" * 60)
    print("WorkSafe NZ AI Assistant - Vector Store Builder")
    print("=" * 60)

    try:
        # Step 1: Load PDF
        documents = load_pdf_with_metadata(PDF_PATH)

        # Step 2: Chunk documents
        chunks = chunk_documents(documents)

        # Step 3: Create vectorstore with embeddings
        vectorstore = create_vectorstore(chunks)

        # Step 4: Save to disk
        save_vectorstore(vectorstore, chunks, VECTORSTORE_PATH)

        # Step 5: Print statistics
        print_statistics(chunks)

        print("\n" + "=" * 60)
        print("✅ VECTORSTORE BUILD COMPLETE!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
