import streamlit as st
from ingest.pdf_Ingest import load_pdf
from ingest.web_Ingestion import load_website
from retrieval.retriever import create_vector_store
from retrieval.qa_chain import build_qa_chain
from utils import format_sources
import os
import shutil

st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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

# Function to clear all context
def clear_all_context():
    """Clear all documents, vector stores, and temporary files"""
    st.session_state.docs = []
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.retriever = None
    st.session_state.current_source = None
    st.session_state.processing_complete = False
    st.session_state.last_processed_file = None
    st.session_state.active_input_mode = None


    
    # Delete vector store directory
    try:
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
    except Exception as e:
        print(f"Warning: Could not delete vectorstore directory: {e}")
    
    # Clean up any temp PDF files
    try:
        import glob
        for temp_file in glob.glob("temp_*.pdf"):
            try:
                os.remove(temp_file)
            except:
                pass
    except Exception as e:
        print(f"Warning: Could not clean temp files: {e}")

st.markdown("<h1>🤖 Intelligent Q&A System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload documents or provide a website URL to get instant answers powered by AI</p>", unsafe_allow_html=True)

# Initialize session state
if 'docs' not in st.session_state:
    st.session_state.docs = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0
if 'current_source' not in st.session_state:
    st.session_state.current_source = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'last_processed_file' not in st.session_state:
    st.session_state.last_processed_file = None
if "active_input_mode" not in st.session_state:
    st.session_state.active_input_mode = None  # "pdf" or "website"


# Create two columns for better layout
col1, col2 = st.columns([1, 1])

# Check if website button was clicked FIRST (before processing uploads)
website_button_clicked = False
new_url = None

with col2:
    st.markdown("### 🌐 Or Enter a Website")
    url = st.text_input(
        "Enter website URL",
        placeholder="https://example.com",
        help="Provide a website URL to extract content",
        key=f"url_input_{st.session_state.file_uploader_key}"
    )
    
    if st.button("🔍 Load Website", key=f"load_btn_{st.session_state.file_uploader_key}"):
        if url:
            website_button_clicked = True
            new_url = url

# Process website BEFORE PDF to ensure it takes priority
if website_button_clicked and new_url:
    url_id = f"url_{new_url}"

    # Always clear previous context when loading a new website
    clear_all_context()
    
    with st.spinner("🌐 Fetching website content..."):
        try:
            st.session_state.docs = load_website(new_url)
            st.session_state.current_source = url_id
            st.session_state.last_processed_file = url_id
            st.session_state.processing_complete = True
            st.session_state.active_input_mode = "website"
            st.session_state.file_uploader_key += 1  # reset PDF uploader state

            # Force vector store rebuild
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.retriever = None
            
            st.success(f"✅ Successfully loaded {len(st.session_state.docs)} chunks from website!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error loading website: {str(e)}")

with col1:
    st.markdown("### 📤 Upload Your Content")
    
    uploaded_pdf = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Select a PDF file to analyze",
        key=f"pdf_uploader_{st.session_state.file_uploader_key}"
    )
    if uploaded_pdf and st.session_state.active_input_mode != "website":

        # Create unique identifier for file
        file_id = f"pdf_{uploaded_pdf.name}_{uploaded_pdf.size}"
        
        # CRITICAL: Only process if this file hasn't been processed yet
        if file_id != st.session_state.last_processed_file:
            # Clear previous context
            clear_all_context()
            
            with st.spinner("📄 Processing PDF..."):
                # Create unique temp filename to avoid conflicts
                import time
                temp_filename = f"temp_{int(time.time())}_{uploaded_pdf.name}"
                
                try:
                    # Save the uploaded file
                    with open(temp_filename, "wb") as f:
                        f.write(uploaded_pdf.read())
                    
                    # Load and process the PDF
                    st.session_state.docs = load_pdf(temp_filename)
                    st.session_state.current_source = file_id
                    st.session_state.last_processed_file = file_id
                    st.session_state.processing_complete = True
                    st.session_state.active_input_mode = "pdf"

                    
                    # Force vector store rebuild
                    st.session_state.vector_store = None
                    st.session_state.qa_chain = None
                    st.session_state.retriever = None
                    
                    # Clean up the temp file immediately after loading
                    try:
                        os.remove(temp_filename)
                    except:
                        pass  # Best effort cleanup
                    
                    st.success(f"✅ Successfully loaded {len(st.session_state.docs)} chunks from PDF!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    # Clean up on error
                    try:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                    except:
                        pass

# Display document status
if st.session_state.docs and st.session_state.current_source and st.session_state.processing_complete:
    # Show current source with better formatting
    if st.session_state.current_source.startswith("pdf_"):
        source_parts = st.session_state.current_source.split("_", 1)[1].rsplit("_", 1)
        source_display = f"📄 PDF: {source_parts[0]}"
    elif st.session_state.current_source.startswith("url_"):
        source_display = f"🌐 Website: {st.session_state.current_source[4:]}"
    else:
        source_display = "📚 Document"
    
    st.markdown(f"<div class='info-msg'><strong>Current Source:</strong> {source_display}<br><strong>Chunks Loaded:</strong> {len(st.session_state.docs)} text chunks ready for analysis</div>", unsafe_allow_html=True)
    
    # Create vector store and QA chain ONLY if processing is complete
    if st.session_state.vector_store is None:
        with st.spinner("🔧 Building knowledge base..."):
            st.session_state.vector_store = create_vector_store(st.session_state.docs)
            st.session_state.qa_chain, st.session_state.retriever = build_qa_chain(st.session_state.vector_store)
    
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
            # Clear all context
            clear_all_context()
            
            # Increment key to reset file uploader
            st.session_state.file_uploader_key += 1
            
            st.rerun()
    
    if question and ask_button:
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.qa_chain.invoke(question)
            
            # Display answer
            st.markdown("### 💡 Answer")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        color: white; 
                        font-size: 1.1rem;
                        line-height: 1.6;'>
                {result}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources
            st.markdown("### 📚 Sources")
            source_docs = st.session_state.retriever.invoke(question)
            sources = format_sources(source_docs)
            
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