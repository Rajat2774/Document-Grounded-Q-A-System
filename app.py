import streamlit as st
from ingest.pdf_Ingest import load_pdf
from ingest.web_Ingestion import load_website
from retrieval.retriever import create_vector_store
from retrieval.qa_chain import build_qa_chain
from utils import format_sources
import re

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Upload section card */
    .upload-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Question section card */
    .question-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Answer card */
    .answer-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
    }
    
    /* Math rendering */
    .math-content {
        font-family: 'Courier New', monospace;
        background: rgba(0,0,0,0.05);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 1.1rem;
    }
    
    /* Source card */
    .source-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    
    .source-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Title styling */
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .subtitle {
        color: white;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #9f7aea !important;
        padding: 0.75rem;
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #2d3748 !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* File uploader styling */
    .stFileUploader>div {
        border-radius: 10px;
    }
    
    .stFileUploader>div>div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border: 2px dashed #9f7aea !important;
        border-radius: 10px;
    }
    
    .stFileUploader>div>div:hover {
        border-color: #764ba2 !important;
        background-color: rgba(255, 255, 255, 1) !important;
    }
    
    /* Status messages */
    .success-msg {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .info-msg {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>🤖 Intelligent Q&A System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload documents or provide a website URL to get instant answers powered by AI</p>", unsafe_allow_html=True)

# Initialize session state
if 'docs' not in st.session_state:
    st.session_state.docs = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Your Content")
    
    # PDF Upload with unique key
    uploaded_pdf = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Select a PDF file to analyze",
        key=f"pdf_uploader_{st.session_state.file_uploader_key}"
    )
    
    if uploaded_pdf:
        # Create unique identifier for file
        file_id = f"{uploaded_pdf.name}_{uploaded_pdf.size}"
        
        # Only process if not already processed
        if file_id not in st.session_state.processed_files:
            with st.spinner("📄 Processing PDF..."):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_pdf.read())
                new_docs = load_pdf("temp.pdf")
                st.session_state.docs.extend(new_docs)
                st.session_state.processed_files.add(file_id)
                # Reset vector store to force rebuild
                st.session_state.vector_store = None
                st.session_state.qa_chain = None
                st.markdown(f"<div class='success-msg'>✅ Successfully loaded {len(new_docs)} chunks from PDF!</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='info-msg'>ℹ️ This file is already loaded ({uploaded_pdf.name})</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 🌐 Or Enter a Website")
    
    # Website URL
    url = st.text_input(
        "Enter website URL",
        placeholder="https://example.com",
        help="Provide a website URL to extract content",
        key=f"url_input_{st.session_state.file_uploader_key}"
    )
    
    if url and st.button("🔍 Load Website", key=f"load_btn_{st.session_state.file_uploader_key}"):
        # Create unique identifier for URL
        url_id = f"url_{url}"
        
        if url_id not in st.session_state.processed_files:
            with st.spinner("🌐 Fetching website content..."):
                try:
                    new_docs = load_website(url)
                    st.session_state.docs.extend(new_docs)
                    st.session_state.processed_files.add(url_id)
                    # Reset vector store to force rebuild
                    st.session_state.vector_store = None
                    st.session_state.qa_chain = None
                    st.markdown(f"<div class='success-msg'>✅ Successfully loaded {len(new_docs)} chunks from website!</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error loading website: {str(e)}")
        else:
            st.markdown(f"<div class='info-msg'>ℹ️ This URL is already loaded</div>", unsafe_allow_html=True)

# Display document status
if st.session_state.docs:
    st.markdown("<div class='info-msg'>📚 <strong>Documents Loaded:</strong> {} text chunks ready for analysis</div>".format(len(st.session_state.docs)), unsafe_allow_html=True)
    
    # Create vector store and QA chain
    if st.session_state.vector_store is None:
        with st.spinner("🔧 Building knowledge base..."):
            st.session_state.vector_store = create_vector_store(st.session_state.docs)
            st.session_state.qa_chain = build_qa_chain(st.session_state.vector_store)
    
    # Divider
    st.markdown("---")
    
    # Question section
    st.markdown("### 💬 Ask Your Question")
    
    question = st.text_input(
        "What would you like to know?",
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )
    
    col_ask, col_clear = st.columns([3, 1])
    
    with col_ask:
        ask_button = st.button("🚀 Get Answer", use_container_width=True)
    
    with col_clear:
        if st.button("🔄 Clear All", use_container_width=True):
            # Clear all session state
            st.session_state.docs = []
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            
            # Clear file uploader
            if 'file_uploader_key' not in st.session_state:
                st.session_state.file_uploader_key = 0
            st.session_state.file_uploader_key += 1
            
            # Delete temp file if it exists
            import os
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")
            
            st.rerun()
    
    if question and ask_button:
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.qa_chain.invoke({"input": question})
            
            # Display answer
            st.markdown("### 💡 Answer")
            answer_text = result["answer"]
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        color: white; 
                        font-size: 1.1rem;
                        line-height: 1.6;'>
                {answer_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources
            st.markdown("### 📚 Sources")
            sources = format_sources(result["context"])
            
            for i, src in enumerate(sources, 1):
                st.markdown(f"""
                <div style='background: #f8f9fa; 
                            padding: 1.5rem; 
                            border-radius: 12px; 
                            border-left: 4px solid #667eea; 
                            margin: 1rem 0;'>
                    <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                        <span style='background: #667eea; 
                                     color: white; 
                                     border-radius: 50%; 
                                     width: 30px; 
                                     height: 30px; 
                                     display: inline-flex; 
                                     align-items: center; 
                                     justify-content: center; 
                                     margin-right: 0.5rem;
                                     font-weight: bold;'>{i}</span>
                        <strong style='color: #333;'>📄 {src['source']}</strong>
                    </div>
                    <div style='color: #666; margin-bottom: 0.5rem;'>
                        📍 Page {src['page']} of {src['total_pages']}
                    </div>
                    <div style='background: white; 
                                padding: 1rem; 
                                border-radius: 8px; 
                                font-family: monospace;
                                color: #555;
                                border-left: 3px solid #e0e0e0;
                                white-space: pre-wrap;'>
                                "{src['excerpt']}"
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style='text-align: center; 
                padding: 4rem 2rem; 
                background: white; 
                border-radius: 15px; 
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
        <h2 style='color: #667eea; margin-bottom: 1rem;'>👋 Welcome!</h2>
        <p style='color: #666; font-size: 1.1rem; margin-bottom: 2rem;'>
            Get started by uploading a PDF document or entering a website URL above.
        </p>
        <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;'>
            <div style='text-align: center;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>📄</div>
                <div style='color: #666;'>Upload PDFs</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🌐</div>
                <div style='color: #666;'>Load Websites</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>💬</div>
                <div style='color: #666;'>Ask Questions</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: white; opacity: 0.8;'>Powered by AI • Built with Streamlit</p>", unsafe_allow_html=True)