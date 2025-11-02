"""
WorkSafe NZ AI Assistant - Streamlit Web Interface
Interactive chat interface for HSWA 2015 Q&A with citations
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load environment variables
load_dotenv()

# Import RAG components
from src.rag_chain import load_vectorstore, create_rag_chain

# Page configuration
st.set_page_config(
    page_title="WorkSafe NZ AI Assistant",
    page_icon="🇳🇿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for WorkSafe NZ branding
st.markdown("""
<style>
    /* WorkSafe NZ color scheme */
    .stApp {
        background-color: #f5f5f5;
    }

    .main-header {
        background: linear-gradient(135deg, #0066cc 0%, #004999 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #0066cc;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .user-message {
        background-color: #e8f4f8;
        border-left: 5px solid #0066cc;
    }

    .assistant-message {
        background-color: white;
        border-left: 5px solid #00a651;
    }

    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        border: 1px solid #dee2e6;
    }

    .example-question {
        cursor: pointer;
        padding: 0.8rem;
        background-color: white;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        border: 1px solid #dee2e6;
        transition: all 0.3s;
    }

    .example-question:hover {
        background-color: #e8f4f8;
        border-color: #0066cc;
        transform: translateX(5px);
    }

    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #004999;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_chain():
    """Initialize RAG chain (cached for performance)"""
    with st.spinner("🚀 Initializing WorkSafe NZ AI Assistant..."):
        vectorstore = load_vectorstore()
        rag_chain = create_rag_chain(vectorstore)
    return rag_chain


def format_sources(source_documents):
    """Format source documents for display"""
    sources_html = ""
    for i, doc in enumerate(source_documents, 1):
        page = doc.metadata.get('page', 'N/A')
        content = doc.page_content[:200].replace('\n', ' ')
        sources_html += f"""
        <div class="source-box">
            <strong>📄 Source {i} (Page {page})</strong><br>
            <em>{content}...</em>
        </div>
        """
    return sources_html


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🇳🇿 WorkSafe NZ AI Assistant</h1>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            Ask questions about the Health and Safety at Work Act 2015
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📚 About This Tool")
        st.info(
            "This AI assistant uses Retrieval-Augmented Generation (RAG) "
            "to answer questions about New Zealand's Health and Safety at Work Act 2015. "
            "All answers include citations to specific sections and pages."
        )

        st.markdown("### 💡 Example Questions")
        example_questions = [
            "What is a PCBU's primary duty of care?",
            "What are the penalties for serious violations?",
            "Who can be held responsible for workplace safety?",
            "What is the definition of a worker?",
            "What are the requirements for health and safety representatives?",
            "What are the duties of an officer under the Act?",
            "When must a workplace incident be notified?",
            "What consultation requirements exist for PCBUs?"
        ]

        for i, question in enumerate(example_questions):
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.example_question = question

        st.markdown("---")
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        - **LLM:** Claude 4.5 Sonnet
        - **Embeddings:** Local (sentence-transformers)
        - **Vector DB:** FAISS
        - **Framework:** LangChain
        - **UI:** Streamlit
        """)

        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.warning(
            "This is an educational tool. For legal advice or official "
            "interpretations, consult WorkSafe NZ or a legal professional."
        )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize RAG chain
    if "rag_chain" not in st.session_state:
        try:
            st.session_state.rag_chain = initialize_rag_chain()
        except Exception as e:
            st.error(f"❌ Error initializing RAG chain: {str(e)}")
            st.stop()

    # Handle example question click
    if "example_question" in st.session_state:
        question = st.session_state.example_question
        del st.session_state.example_question

        # Add to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # Get response
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.rag_chain({"query": question})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["result"],
                    "sources": response["source_documents"]
                })
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

            # Show sources in expander
            with st.expander("📚 View Sources"):
                st.markdown(format_sources(message["sources"]), unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about the Health and Safety at Work Act 2015..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get response
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.rag_chain({"query": prompt})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["result"],
                    "sources": response["source_documents"]
                })
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Clear chat button
    if len(st.session_state.messages) > 0:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("""
        ❌ **ANTHROPIC_API_KEY not found!**

        Please create a `.env` file with:
        ```
        ANTHROPIC_API_KEY=your_key_here
        ```
        """)
        st.stop()

    main()
