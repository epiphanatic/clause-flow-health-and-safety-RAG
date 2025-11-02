"""
Interactive Chat Interface for WorkSafe NZ AI Assistant
Simple CLI-based chat for testing the RAG chain
"""

import os
import sys
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from rag_chain import load_vectorstore, create_rag_chain

# Load environment variables
load_dotenv()


def print_header():
    """Print welcome header"""
    print("\n" + "=" * 80)
    print("🇳🇿 WorkSafe NZ AI Assistant - HSWA 2015 Q&A")
    print("=" * 80)
    print("\nAsk questions about the Health and Safety at Work Act 2015.")
    print("Type 'quit' or 'exit' to stop.\n")
    print("Example questions:")
    print("  - What is a PCBU's primary duty of care?")
    print("  - What are the penalties for violations?")
    print("  - Who is responsible for workplace safety?")
    print("\n" + "=" * 80 + "\n")


def format_sources(source_documents):
    """Format source documents for display"""
    sources = []
    for i, doc in enumerate(source_documents, 1):
        page = doc.metadata.get('page', 'N/A')
        preview = doc.page_content[:100].replace('\n', ' ').strip()
        sources.append(f"  [{i}] Page {page}: {preview}...")
    return "\n".join(sources)


def chat_loop(rag_chain):
    """Main interactive chat loop"""
    print_header()

    conversation_count = 0

    while True:
        try:
            # Get user input
            question = input("You: ").strip()

            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using WorkSafe NZ AI Assistant!")
                break

            # Skip empty questions
            if not question:
                continue

            # Process question
            print("\n🤔 Thinking...\n")
            response = rag_chain({"query": question})

            # Display answer
            print(f"Assistant: {response['result']}\n")

            # Display sources
            print("📚 Sources:")
            print(format_sources(response['source_documents']))
            print("\n" + "-" * 80 + "\n")

            conversation_count += 1

        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using WorkSafe NZ AI Assistant!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            continue

    print(f"\nTotal questions answered: {conversation_count}")


def main():
    """Initialize and start chat interface"""
    try:
        # Check for API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("\n❌ Error: ANTHROPIC_API_KEY not found in environment variables.")
            print("\nPlease create a .env file with:")
            print("ANTHROPIC_API_KEY=your_key_here")
            sys.exit(1)

        print("\n🚀 Initializing WorkSafe NZ AI Assistant...")

        # Load vectorstore
        print("📥 Loading knowledge base...")
        vectorstore = load_vectorstore()

        # Create RAG chain
        print("🤖 Connecting to Claude 4.5 Sonnet...")
        rag_chain = create_rag_chain(vectorstore)

        # Start chat
        chat_loop(rag_chain)

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
