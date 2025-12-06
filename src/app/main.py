import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings
import hashlib
from io import BytesIO
import speech_recognition as sr

# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Optional imports for enhanced features
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = False
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR libraries not available. Install Pillow and pytesseract for OCR support.")

try:
    import PyPDF2
    PASSWORD_PDF_AVAILABLE = False
except ImportError:
    PASSWORD_PDF_AVAILABLE = False
    logging.warning("PyPDF2 not available. Install it for password-protected PDF support.")

try:
    from streamlit_mic_recorder import speech_to_text
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False
    logging.warning("streamlit-mic-recorder not available. Install it for voice input support.")

# Set protobuf environment variable to avoid error messages
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Streamlit page configuration
st.set_page_config(
    page_title="QUERYNEXUS:AI PDF Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- ULTIMATE AI UI WITH ADVANCED ANIMATIONS & EFFECTS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;600;700&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: #000000;
    color: #fff;
}

/* Animated Neural Background */
.main::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: 
        radial-gradient(ellipse at 10% 20%, rgba(99,102,241,0.15), transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(139,92,246,0.15), transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(236,72,153,0.1), transparent 50%),
        linear-gradient(180deg, #000 0%, #0a0a1a 50%, #000 100%);
    z-index: -2;
    animation: bgPulse 20s ease infinite;
}

@keyframes bgPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

/* Animated Grid */
.main::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image: 
        linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    z-index: -1;
    animation: gridMove 20s linear infinite;
}

@keyframes gridMove {
    0% { transform: translate(0, 0); }
    100% { transform: translate(50px, 50px); }
}

/* Holographic Headers */
h1, h2, h3 {
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 25%, #ec4899 50%, #8b5cf6 75%, #6366f1 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: holoShift 8s ease infinite;
}

@keyframes holoShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.main h1 {
    font-size: 3.5em !important;
    text-shadow: 0 0 30px rgba(99,102,241,0.5), 0 0 60px rgba(139,92,246,0.3);
    animation: holoShift 8s ease infinite, floatTitle 6s ease infinite;
}

@keyframes floatTitle {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

/* Glassmorphic Containers */
[data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
    background: rgba(15,15,35,0.6) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    border-radius: 24px !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
}

[data-testid="stVerticalBlock"] > [data-testid="stContainer"]::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.1), transparent);
    transition: left 0.7s;
}

[data-testid="stVerticalBlock"] > [data-testid="stContainer"]:hover::before {
    left: 100%;
}

[data-testid="stVerticalBlock"] > [data-testid="stContainer"]:hover {
    transform: translateY(-4px) scale(1.01);
    border-color: rgba(99,102,241,0.4) !important;
    box-shadow: 0 16px 48px rgba(99,102,241,0.3) !important;
}

/* Advanced Chat Messages */
[data-testid="stChatMessage"] {
    background: rgba(15,15,35,0.6) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    margin: 12px 0 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    position: relative !important;
    overflow: hidden !important;
}

[data-testid="stChatMessage"][class*="user"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3)) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.3) !important;
    animation: slideInRight 0.6s cubic-bezier(0.34,1.56,0.64,1) !important;
}

[data-testid="stChatMessage"][class*="user"]::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
    transform: rotate(45deg);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%) rotate(45deg); }
    100% { transform: translateX(100%) rotate(45deg); }
}

[data-testid="stChatMessage"][class*="assistant"] {
    background: linear-gradient(135deg, rgba(236,72,153,0.2), rgba(139,92,246,0.2)) !important;
    border: 1px solid rgba(236,72,153,0.3) !important;
    box-shadow: 0 8px 32px rgba(236,72,153,0.2) !important;
    animation: slideInLeft 0.6s cubic-bezier(0.34,1.56,0.64,1) !important;
}

[data-testid="stChatMessage"][class*="assistant"]::after {
    content: '✨';
    position: absolute;
    top: 10px; right: 10px;
    font-size: 20px;
    animation: sparkle 2s ease infinite;
}

@keyframes sparkle {
    0%, 100% { opacity: 0.5; transform: scale(1) rotate(0deg); }
    50% { opacity: 1; transform: scale(1.2) rotate(180deg); }
}

@keyframes slideInRight {
    0% { opacity: 0; transform: translateX(50px) scale(0.8); }
    100% { opacity: 1; transform: translateX(0) scale(1); }
}

@keyframes slideInLeft {
    0% { opacity: 0; transform: translateX(-50px) scale(0.8); }
    100% { opacity: 1; transform: translateX(0) scale(1); }
}

/* Futuristic Input */
.stChatInputContainer {
    background: rgba(15,15,35,0.8) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 20px !important;
    border: 2px solid rgba(99,102,241,0.3) !important;
    padding: 8px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(99,102,241,0.2) !important;
    animation: inputGlow 3s ease infinite;
}

@keyframes inputGlow {
    0%, 100% { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(99,102,241,0.2); }
    50% { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 30px rgba(99,102,241,0.4); }
}

.stChatInputContainer::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent);
    animation: scanLine 3s linear infinite;
}

@keyframes scanLine {
    0% { left: -100%; }
    100% { left: 100%; }
}

.stChatInputContainer textarea {
    background: transparent !important;
    color: #fff !important;
    border: none !important;
    font-size: 16px !important;
}

/* Cyber Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%) !important;
    background-size: 200% auto;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 14px 32px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
    overflow: hidden;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.5s;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.05) !important;
    background-position: 100% 0;
    box-shadow: 0 8px 30px rgba(99,102,241,0.6), 0 0 40px rgba(139,92,246,0.4) !important;
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #b91c1c 100%) !important;
    box-shadow: 0 4px 20px rgba(239,68,68,0.4) !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(15,15,35,0.6) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 20px !important;
    border: 2px dashed rgba(99,102,241,0.4) !important;
    padding: 30px !important;
    transition: all 0.4s !important;
    position: relative;
}

[data-testid="stFileUploader"]::before {
    content: '📄';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 60px;
    opacity: 0.1;
    animation: floatUpload 3s ease infinite;
}

@keyframes floatUpload {
    0%, 100% { transform: translate(-50%, -50%) scale(1); }
    50% { transform: translate(-50%, -60%) scale(1.1); }
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.8) !important;
    background: rgba(99,102,241,0.1) !important;
    transform: scale(1.02);
    box-shadow: 0 0 40px rgba(99,102,241,0.3);
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(15,15,35,0.8) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 16px !important;
    color: white !important;
    transition: all 0.3s !important;
}

/* Slider */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899) !important;
}

.stSlider > div > div > div > div {
    background: white !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.6) !important;
    width: 24px !important;
    height: 24px !important;
}

.stSlider > div > div > div > div:hover {
    transform: scale(1.3);
    box-shadow: 0 0 30px rgba(99,102,241,0.8) !important;
}

/* Checkbox */
.stCheckbox {
    background: rgba(15,15,35,0.6) !important;
    backdrop-filter: blur(12px) !important;
    padding: 16px 20px !important;
    border-radius: 16px !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    transition: all 0.3s !important;
}

.stCheckbox:hover {
    background: rgba(99,102,241,0.15) !important;
    transform: translateX(4px);
}

/* Spinner */
.stSpinner > div {
    border-top-color: #6366f1 !important;
    border-right-color: #8b5cf6 !important;
    border-bottom-color: #ec4899 !important;
    width: 50px !important;
    height: 50px !important;
    animation: spinNeutral 1s cubic-bezier(0.68,-0.55,0.27,1.55) infinite !important;
}

@keyframes spinNeutral {
    0% { transform: rotate(0deg) scale(1); box-shadow: 0 0 20px rgba(99,102,241,0.4); }
    50% { transform: rotate(180deg) scale(1.1); box-shadow: 0 0 40px rgba(139,92,246,0.6); }
    100% { transform: rotate(360deg) scale(1); box-shadow: 0 0 20px rgba(236,72,153,0.4); }
}

/* Notifications */
.stSuccess, .stWarning, .stError, .stInfo {
    background: rgba(15,15,35,0.8) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 16px !important;
    border-left: 4px solid !important;
    padding: 20px !important;
    animation: slideNotif 0.5s cubic-bezier(0.4,0,0.2,1) !important;
}

@keyframes slideNotif {
    0% { opacity: 0; transform: translateX(-50px); }
    100% { opacity: 1; transform: translateX(0); }
}

.stSuccess {
    border-left-color: #10b981 !important;
    background: rgba(16,185,129,0.15) !important;
}

.stWarning {
    border-left-color: #f59e0b !important;
    background: rgba(245,158,11,0.15) !important;
}

.stError {
    border-left-color: #ef4444 !important;
    background: rgba(239,68,68,0.15) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: rgba(15,15,35,0.6);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #6366f1, #8b5cf6, #ec4899);
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(99,102,241,0.5);
}

::-webkit-scrollbar-thumb:hover {
    box-shadow: 0 0 20px rgba(139,92,246,0.8);
}

/* Image Enhancement */
.stImage {
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
    transition: all 0.3s !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
}

.stImage:hover {
    transform: scale(1.02);
    box-shadow: 0 12px 48px rgba(99,102,241,0.3) !important;
}

/* Icon Animations */
.icon-float {
    display: inline-block;
    animation: iconFloat 3s ease infinite;
}

@keyframes iconFloat {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-15px) rotate(5deg); }
}

/* Badge Styling */
.doc-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
    margin: 4px;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def calculate_file_hash(file_bytes: bytes) -> str:
    """
    Calculate SHA-256 hash of file content for deduplication.
    
    Args:
        file_bytes: File content in bytes
        
    Returns:
        str: SHA-256 hash of the file
    """
    return hashlib.sha256(file_bytes).hexdigest()


def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info: Response from ollama.list()

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    try:
        if hasattr(models_info, "models"):
            model_names = tuple(model.model for model in models_info.models)
        else:
            model_names = tuple()
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


def unlock_pdf(file_upload, password: str) -> Optional[bytes]:
    """
    Unlock password-protected PDF.
    
    Args:
        file_upload: Streamlit file upload object
        password: Password to unlock the PDF
        
    Returns:
        Optional[bytes]: Unlocked PDF bytes or None if failed
    """
    if not PASSWORD_PDF_AVAILABLE:
        st.error("PyPDF2 not installed. Install it for password-protected PDF support.")
        return None
    
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_upload.getvalue()))
        
        if pdf_reader.is_encrypted:
            if pdf_reader.decrypt(password):
                pdf_writer = PyPDF2.PdfWriter()
                for page in pdf_reader.pages:
                    pdf_writer.add_page(page)
                
                output = BytesIO()
                pdf_writer.write(output)
                output.seek(0)
                return output.getvalue()
            else:
                st.error("❌ Incorrect password!")
                return None
        else:
            return file_upload.getvalue()
    except Exception as e:
        logger.error(f"Error unlocking PDF: {e}")
        st.error(f"Error unlocking PDF: {str(e)}")
        return None


def perform_ocr_on_pdf(file_path: str) -> str:
    """
    Perform OCR on scanned PDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        str: Extracted text from OCR
    """
    if not OCR_AVAILABLE:
        st.warning("OCR libraries not available. Install Pillow and pytesseract.")
        return ""
    
    try:
        import pdf2image
        pages = pdf2image.convert_from_path(file_path)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"
        return text
    except ImportError:
        st.error("pdf2image not installed. Install it for OCR support: pip install pdf2image")
        return ""
    except Exception as e:
        logger.error(f"OCR error: {e}")
        st.error(f"OCR error: {str(e)}")
        return ""


def create_vector_db(file_uploads: List, passwords: Dict[str, str] = None, enable_ocr: bool = False) -> Chroma:
    """
    Create a vector database from uploaded PDF files with deduplication.

    Args:
        file_uploads: List of Streamlit file upload objects
        passwords: Dictionary mapping filenames to passwords
        enable_ocr: Whether to enable OCR for scanned PDFs

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from {len(file_uploads)} file(s)")
    temp_dir = tempfile.mkdtemp()
    all_chunks = []
    processed_hashes = set()
    
    passwords = passwords or {}

    try:
        for file_upload in file_uploads:
            # Calculate file hash for deduplication
            file_bytes = file_upload.getvalue()
            file_hash = calculate_file_hash(file_bytes)
            
            # Check if file already processed
            if file_hash in st.session_state.get("processed_hashes", set()):
                st.warning(f"⚠️ Skipping duplicate: {file_upload.name}")
                continue
            
            # Handle password-protected PDFs
            if file_upload.name in passwords:
                file_bytes = unlock_pdf(file_upload, passwords[file_upload.name])
                if file_bytes is None:
                    continue
            
            path = os.path.join(temp_dir, file_upload.name)
            with open(path, "wb") as f:
                f.write(file_bytes)
                logger.info(f"File saved to temporary path: {path}")
            
            # Try standard PDF loading
            try:
                loader = PyPDFLoader(path)
                data = loader.load()
                
                # If no text extracted and OCR enabled, perform OCR
                if enable_ocr and all(len(doc.page_content.strip()) < 50 for doc in data):
                    st.info(f"🔍 Performing OCR on {file_upload.name}...")
                    ocr_text = perform_ocr_on_pdf(path)
                    if ocr_text:
                        from langchain.schema import Document
                        data = [Document(page_content=ocr_text, metadata={"source": file_upload.name})]
                
            except Exception as e:
                logger.error(f"Error loading PDF {file_upload.name}: {e}")
                st.error(f"Error loading {file_upload.name}: {str(e)}")
                continue

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            
            # Add source metadata
            for chunk in chunks:
                chunk.metadata["source_file"] = file_upload.name
                chunk.metadata["file_hash"] = file_hash
            
            all_chunks.extend(chunks)
            processed_hashes.add(file_hash)
            
            logger.info(f"Processed {file_upload.name}: {len(chunks)} chunks")

        if not all_chunks:
            st.error("❌ No documents to process!")
            return None

        logger.info(f"Total chunks across all documents: {len(all_chunks)}")

        # Create embeddings and vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name="multi_pdf_collection"
        )
        
        # Store processed hashes in session state
        if "processed_hashes" not in st.session_state:
            st.session_state["processed_hashes"] = set()
        st.session_state["processed_hashes"].update(processed_hashes)
        
        logger.info("Vector DB created with persistent storage")
        
        return vector_db
        
    finally:
        shutil.rmtree(temp_dir)
        logger.info(f"Temporary directory {temp_dir} removed")


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    llm = ChatOllama(model="llama3.1")
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload: Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            vector_db.delete_collection()
            
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_uploads", None)
            st.session_state.pop("vector_db", None)
            st.session_state.pop("processed_hashes", None)
            st.session_state.pop("uploaded_files", None)
            
            st.success("✅ Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("⚠️ No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def speech_to_text_input() -> Optional[str]:
    """
    Capture voice input and convert to text using Web Speech API.
    
    Returns:
        Optional[str]: Transcribed text or None
    """
    if VOICE_INPUT_AVAILABLE:
        text = speech_to_text(
            language='en',
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            use_container_width=True,
            key='voice_input'
        )
        return text
    else:
        st.warning("⚠️ Voice input not available. Install: pip install streamlit-mic-recorder")
        return None


def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    # Animated holographic header
    st.markdown(
        """
        <div style='text-align: center; padding: 30px 0 20px 0;'>
            <h1 style='font-size: 4em; margin: 0;'>
                <span class='icon-float'>🤖</span> QUERYNEXUS:AI PDF ASSISTANT
            </h1>
            <p style='color: rgba(255,255,255,0.5); font-size: 1.2em; margin-top: 15px; letter-spacing: 2px;'>
                ⚡ POWERED BY OLLAMA & LANGCHAIN ⚡
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False
    if "processed_hashes" not in st.session_state:
        st.session_state["processed_hashes"] = set()
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    # LEFT COLUMN - PDF Upload & Controls
    with col1:
        st.markdown("### 📄 DOCUMENT MANAGEMENT")
        
        # Model selection
        if available_models:
            selected_model = st.selectbox(
                "🎯 AI MODEL", 
                available_models,
                key="model_select"
            )

        # Sample PDF toggle
        use_sample = st.toggle(
            "📚 USE SAMPLE PDF", 
            key="sample_checkbox"
        )
        
        # Clear vector DB if switching between sample and upload
        if use_sample != st.session_state.get("use_sample"):
            if st.session_state["vector_db"] is not None:
                st.session_state["vector_db"].delete_collection()
                st.session_state["vector_db"] = None
                st.session_state["pdf_pages"] = None
                st.session_state["processed_hashes"] = set()
            st.session_state["use_sample"] = use_sample

        if use_sample:
            # Sample PDF handling
            sample_path = "data/pdfs/sample/scammer-agent.pdf"
            if os.path.exists(sample_path):
                if st.session_state["vector_db"] is None:
                    with st.spinner("🔄 PROCESSING SAMPLE PDF..."):
                        loader = PyPDFLoader(sample_path)
                        data = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                        chunks = text_splitter.split_documents(data)
                        st.session_state["vector_db"] = Chroma.from_documents(
                            documents=chunks,
                            embedding=OllamaEmbeddings(model="nomic-embed-text"),
                            persist_directory=PERSIST_DIRECTORY,
                            collection_name="sample_pdf"
                        )
                        with pdfplumber.open(sample_path) as pdf:
                            st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
                        st.session_state["uploaded_files"] = ["sample_pdf"]
            else:
                st.error("❌ Sample PDF not found")
        else:
            # Multi-PDF upload
            file_uploads = st.file_uploader(
                "📤 UPLOAD PDF(S) (Multi-select enabled)", 
                type="pdf", 
                accept_multiple_files=True,
                key="pdf_uploader"
            )

            if file_uploads:
                # Show uploaded files
                st.markdown("**📑 Uploaded Files:**")
                for file in file_uploads:
                    file_hash = calculate_file_hash(file.getvalue())
                    status = "✅ Already Processed" if file_hash in st.session_state.get("processed_hashes", set()) else "🆕 New"
                    st.markdown(f"<span class='doc-badge'>{file.name} - {status}</span>", unsafe_allow_html=True)

                # OCR Option
                enable_ocr = st.checkbox("🔍 Enable OCR for scanned PDFs", value=False)

                # Password handling for protected PDFs
                passwords = {}
                with st.expander("🔐 Password-Protected PDFs"):
                    for file in file_uploads:
                        pwd = st.text_input(
                            f"Password for {file.name} (leave empty if not protected)",
                            type="password",
                            key=f"pwd_{file.name}"
                        )
                        if pwd:
                            passwords[file.name] = pwd

                # Process button
                if st.button("⚡ PROCESS ALL DOCUMENTS", type="primary", use_container_width=True):
                    with st.spinner("🔄 PROCESSING DOCUMENTS..."):
                        vector_db = create_vector_db(file_uploads, passwords, enable_ocr)
                        if vector_db:
                            st.session_state["vector_db"] = vector_db
                            st.session_state["uploaded_files"] = [f.name for f in file_uploads]
                            # Extract pages from first PDF for preview
                            with pdfplumber.open(file_uploads[0]) as pdf:
                                st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
                            st.success(f"✅ Processed {len(file_uploads)} document(s)!")
                            st.rerun()

        # Display PDF preview if available
        if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
            st.markdown("---")
            st.markdown("**🖼️ PDF PREVIEW**")
            zoom_level = st.slider(
                "🔍 ZOOM", 
                min_value=100, 
                max_value=1000, 
                value=700, 
                step=50,
                key="zoom_slider"
            )

            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

        # Show processed documents info
        if st.session_state.get("uploaded_files"):
            st.markdown("---")
            st.markdown("**📊 ACTIVE DOCUMENTS:**")
            for filename in st.session_state["uploaded_files"]:
                st.markdown(f"<span class='doc-badge'>📄 {filename}</span>", unsafe_allow_html=True)

        # Delete collection button
        st.markdown("---")
        delete_collection = st.button(
            "🗑️ DELETE COLLECTION", 
            type="secondary",
            key="delete_button",
            use_container_width=True
        )

        if delete_collection:
            delete_vector_db(st.session_state["vector_db"])

    # RIGHT COLUMN - Chat Interface
    with col2:
        st.markdown("### 💬 NEURAL CHAT INTERFACE")
        
        # Voice input section
        if VOICE_INPUT_AVAILABLE:
            with st.expander("🎤 VOICE QUERY INPUT"):
                col_voice1, col_voice2 = st.columns([3, 1])
                with col_voice1:
                    voice_text = speech_to_text_input()
                with col_voice2:
                    if st.button("🔄 CLEAR", key="clear_voice"):
                        st.rerun()
                
                if voice_text:
                    st.info(f"📝 Transcribed: {voice_text}")
                    if st.button("✅ USE THIS QUERY", type="primary"):
                        st.session_state["voice_query"] = voice_text
                        st.rerun()
        
        # Chat container
        chat_container = st.container(height=520, border=True)
        
        with chat_container:
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg["content"])

        # Check for voice query
        voice_query = st.session_state.pop("voice_query", None)
        
        # Chat input
        if prompt := st.chat_input("💭 ASK ME ANYTHING...", key="chat_input"):
            query = prompt
        elif voice_query:
            query = voice_query
        else:
            query = None

        if query:
            try:
                st.session_state["messages"].append({"role": "user", "content": query})

                if st.session_state["vector_db"] is not None:
                    with st.spinner("🧠 NEURAL PROCESSING..."):
                        response = process_question(
                            query, st.session_state["vector_db"], selected_model
                        )
                    
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )
                    st.rerun()
                else:
                    st.warning("⚠️ UPLOAD & PROCESS PDF(S) FIRST")

            except Exception as e:
                st.error(f"❌ ERROR: {str(e)}")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.info("👋 UPLOAD & PROCESS PDF(S) TO START CHATTING!")


if __name__ == "__main__":
    main()


# import streamlit as st
# import logging
# import os
# import tempfile
# import shutil
# import pdfplumber
# import ollama
# import warnings

# # Suppress torch warning
# warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from typing import List, Tuple, Dict, Any, Optional

# # Set protobuf environment variable to avoid error messages
# # This might cause some issues with latency but it's a tradeoff
# os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# # Define persistent directory for ChromaDB
# PERSIST_DIRECTORY = os.path.join("data", "vectors")

# # Streamlit page configuration
# st.set_page_config(
#     page_title="Ollama PDF RAG Streamlit UI",
#     page_icon="🎈",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# # Logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# logger = logging.getLogger(__name__)


# def extract_model_names(models_info: Any) -> Tuple[str, ...]:
#     """
#     Extract model names from the provided models information.

#     Args:
#         models_info: Response from ollama.list()

#     Returns:
#         Tuple[str, ...]: A tuple of model names.
#     """
#     logger.info("Extracting model names from models_info")
#     try:
#         # The new response format returns a list of Model objects
#         if hasattr(models_info, "models"):
#             # Extract model names from the Model objects
#             model_names = tuple(model.model for model in models_info.models)
#         else:
#             # Fallback for any other format
#             model_names = tuple()
            
#         logger.info(f"Extracted model names: {model_names}")
#         return model_names
#     except Exception as e:
#         logger.error(f"Error extracting model names: {e}")
#         return tuple()


# def create_vector_db(file_upload) -> Chroma:
#     """
#     Create a vector database from an uploaded PDF file.

#     Args:
#         file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

#     Returns:
#         Chroma: A vector store containing the processed document chunks.
#     """
#     logger.info(f"Creating vector DB from file upload: {file_upload.name}")
#     temp_dir = tempfile.mkdtemp()

#     path = os.path.join(temp_dir, file_upload.name)
#     with open(path, "wb") as f:
#         f.write(file_upload.getvalue())
#         logger.info(f"File saved to temporary path: {path}")
#         loader = PyPDFLoader(path)
#         data = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#     chunks = text_splitter.split_documents(data)
#     logger.info("Document split into chunks")

#     # Updated embeddings configuration with persistent storage
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")

#     vector_db = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=PERSIST_DIRECTORY,
#         collection_name=f"pdf_{hash(file_upload.name)}"  # Unique collection name per file
#     )
#     logger.info("Vector DB created with persistent storage")

#     shutil.rmtree(temp_dir)
#     logger.info(f"Temporary directory {temp_dir} removed")
#     return vector_db


# def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
#     """
#     Process a user question using the vector database and selected language model.

#     Args:
#         question (str): The user's question.
#         vector_db (Chroma): The vector database containing document embeddings.
#         selected_model (str): The name of the selected language model.

#     Returns:
#         str: The generated response to the user's question.
#     """
#     logger.info(f"Processing question: {question} using model: {selected_model}")
    
#     # Initialize LLM
#     llm = ChatOllama(model="llama3.1")
    
#     # Query prompt template
#     QUERY_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""You are an AI language model assistant. Your task is to generate 2
#         different versions of the given user question to retrieve relevant documents from
#         a vector database. By generating multiple perspectives on the user question, your
#         goal is to help the user overcome some of the limitations of the distance-based
#         similarity search. Provide these alternative questions separated by newlines.
#         Original question: {question}""",
#     )

#     # Set up retriever
#     retriever = MultiQueryRetriever.from_llm(
#         vector_db.as_retriever(), 
#         llm,
#         prompt=QUERY_PROMPT
#     )

#     # RAG prompt template
#     template = """Answer the question based ONLY on the following context:
#     {context}
#     Question: {question}
#     """

#     prompt = ChatPromptTemplate.from_template(template)

#     # Create chain
#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     response = chain.invoke(question)
#     logger.info("Question processed and response generated")
#     return response


# @st.cache_data
# def extract_all_pages_as_images(file_upload) -> List[Any]:
#     """
#     Extract all pages from a PDF file as images.

#     Args:
#         file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

#     Returns:
#         List[Any]: A list of image objects representing each page of the PDF.
#     """
#     logger.info(f"Extracting all pages as images from file: {file_upload.name}")
#     pdf_pages = []
#     with pdfplumber.open(file_upload) as pdf:
#         pdf_pages = [page.to_image().original for page in pdf.pages]
#     logger.info("PDF pages extracted as images")
#     return pdf_pages


# def delete_vector_db(vector_db: Optional[Chroma]) -> None:
#     """
#     Delete the vector database and clear related session state.

#     Args:
#         vector_db (Optional[Chroma]): The vector database to be deleted.
#     """
#     logger.info("Deleting vector DB")
#     if vector_db is not None:
#         try:
#             # Delete the collection
#             vector_db.delete_collection()
            
#             # Clear session state
#             st.session_state.pop("pdf_pages", None)
#             st.session_state.pop("file_upload", None)
#             st.session_state.pop("vector_db", None)
            
#             st.success("Collection and temporary files deleted successfully.")
#             logger.info("Vector DB and related session state cleared")
#             st.rerun()
#         except Exception as e:
#             st.error(f"Error deleting collection: {str(e)}")
#             logger.error(f"Error deleting collection: {e}")
#     else:
#         st.error("No vector database found to delete.")
#         logger.warning("Attempted to delete vector DB, but none was found")


# def main() -> None:
#     """
#     Main function to run the Streamlit application.
#     """
#     st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)

#     # Get available models
#     models_info = ollama.list()
#     available_models = extract_model_names(models_info)

#     # Create layout
#     col1, col2 = st.columns([1.5, 2])

#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []
#     if "vector_db" not in st.session_state:
#         st.session_state["vector_db"] = None
#     if "use_sample" not in st.session_state:
#         st.session_state["use_sample"] = False

#     # Model selection
#     if available_models:
#         selected_model = col2.selectbox(
#             "Pick a model available locally on your system ↓", 
#             available_models,
#             key="model_select"
#         )

#     # Add checkbox for sample PDF
#     use_sample = col1.toggle(
#         "Use sample PDF (Scammer Agent Paper)", 
#         key="sample_checkbox"
#     )
    
#     # Clear vector DB if switching between sample and upload
#     if use_sample != st.session_state.get("use_sample"):
#         if st.session_state["vector_db"] is not None:
#             st.session_state["vector_db"].delete_collection()
#             st.session_state["vector_db"] = None
#             st.session_state["pdf_pages"] = None
#         st.session_state["use_sample"] = use_sample

#     if use_sample:
#         # Use the sample PDF
#         sample_path = "data/pdfs/sample/scammer-agent.pdf"
#         if os.path.exists(sample_path):
#             if st.session_state["vector_db"] is None:
#                 with st.spinner("Processing sample PDF..."):
#                     loader = PyPDFLoader(sample_path)

#                     data = loader.load()
#                     text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#                     chunks = text_splitter.split_documents(data)
#                     st.session_state["vector_db"] = Chroma.from_documents(
#                         documents=chunks,
#                         embedding=OllamaEmbeddings(model="nomic-embed-text"),
#                         persist_directory=PERSIST_DIRECTORY,
#                         collection_name="sample_pdf"
#                     )
#                     # Open and display the sample PDF
#                     with pdfplumber.open(sample_path) as pdf:
#                         st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
#         else:
#             st.error("Sample PDF file not found in the current directory.")
#     else:
#         # Regular file upload with unique key
#         file_upload = col1.file_uploader(
#             "Upload a PDF file ↓", 
#             type="pdf", 
#             accept_multiple_files=False,
#             key="pdf_uploader"
#         )

#         if file_upload:
#             if st.session_state["vector_db"] is None:
#                 with st.spinner("Processing uploaded PDF..."):
#                     st.session_state["vector_db"] = create_vector_db(file_upload)
#                     # Store the uploaded file in session state
#                     st.session_state["file_upload"] = file_upload
#                     # Extract and store PDF pages
#                     with pdfplumber.open(file_upload) as pdf:
#                         st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]

#     # Display PDF if pages are available
#     if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
#         # PDF display controls
#         zoom_level = col1.slider(
#             "Zoom Level", 
#             min_value=100, 
#             max_value=1000, 
#             value=700, 
#             step=50,
#             key="zoom_slider"
#         )

#         # Display PDF pages
#         with col1:
#             with st.container(height=410, border=True):
#                 for page_image in st.session_state["pdf_pages"]:
#                     st.image(page_image, width=zoom_level)

#     # Delete collection button
#     delete_collection = col1.button(
#         "⚠️ Delete collection", 
#         type="secondary",
#         key="delete_button"
#     )

#     if delete_collection:
#         delete_vector_db(st.session_state["vector_db"])

#     # Chat interface
#     with col2:
#         message_container = st.container(height=500, border=True)

#         # Display chat history
#         for i, message in enumerate(st.session_state["messages"]):
#             avatar = "🤖" if message["role"] == "assistant" else "😎"
#             with message_container.chat_message(message["role"], avatar=avatar):
#                 st.markdown(message["content"])

#         # Chat input and processing
#         if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
#             try:
#                 # Add user message to chat
#                 st.session_state["messages"].append({"role": "user", "content": prompt})
#                 with message_container.chat_message("user", avatar="😎"):
#                     st.markdown(prompt)

#                 # Process and display assistant response
#                 with message_container.chat_message("assistant", avatar="🤖"):
#                     with st.spinner(":green[processing...]"):
#                         if st.session_state["vector_db"] is not None:
#                             response = process_question(
#                                 prompt, st.session_state["vector_db"], selected_model
#                             )
#                             st.markdown(response)
#                         else:
#                             st.warning("Please upload a PDF file first.")

#                 # Add assistant response to chat history
#                 if st.session_state["vector_db"] is not None:
#                     st.session_state["messages"].append(
#                         {"role": "assistant", "content": response}
#                     )

#             except Exception as e:
#                 st.error(e, icon="⛔️")
#                 logger.error(f"Error processing prompt: {e}")
#         else:
#             if st.session_state["vector_db"] is None:
#                 st.warning("Upload a PDF file or use the sample PDF to begin chat...")


# if __name__ == "__main__":
#     main()



###NEWWWWWWW

# """
# Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.
# Updated to accept multiple PDFs at once, animated AI-background, duplicate Aadhaar detection and better UI.
# Logic of LLM/chain preserved — vector DB used as a single Chroma instance.
# """

# import streamlit as st
# import logging
# import os
# import tempfile
# import shutil
# import pdfplumber
# import ollama
# import warnings
# import re
# from typing import List, Tuple, Any, Optional

# warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# PERSIST_DIRECTORY = os.path.join("data", "vectors")

# st.set_page_config(
#     page_title="AI PDF RAG Assistant",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# # --- Styling & Animated Background (classy + AI blob) ---
# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;600;700&display=swap');
#     * { font-family: 'Space Grotesk', sans-serif; }
#     .stApp { background: #000000; color: #fff; }

#     /* Subtle AI Blob (SVG data URI) */
#     body::before {
#         content: '';
#         position: fixed;
#         inset: 0;
#         background: radial-gradient(circle at 20% 20%, rgba(99,102,241,0.09), transparent 12%),
#                     radial-gradient(circle at 80% 80%, rgba(236,72,153,0.06), transparent 12%),
#                     linear-gradient(180deg, rgba(2,6,23,0.8), rgba(3,7,30,0.85));
#         z-index: -3;
#         animation: blobFloat 25s ease-in-out infinite;
#     }

#     @keyframes blobFloat {
#         0% { transform: scale(1) translateY(0); }
#         50% { transform: scale(1.03) translateY(-12px); }
#         100% { transform: scale(1) translateY(0); }
#     }

#     /* Holographic header + subtle robot watermark */
#     .robot-watermark {
#         position: fixed; right: 5%; top: 10%; opacity: 0.04; font-size: 240px; z-index: -2; transform: rotate(-15deg);
#         pointer-events: none;
#     }

#     /* Keep the rest of your existing CSS (glassmorphism, buttons, uploader etc.) */
#     /* For brevity I'm including the core elements only; the rest from your previous CSS can be merged similarly. */

#     [data-testid="stFileUploader"] {
#         background: rgba(15,15,35,0.6) !important;
#         backdrop-filter: blur(20px) !important;
#         border-radius: 20px !important;
#         border: 2px dashed rgba(99,102,241,0.4) !important;
#         padding: 30px !important;
#         transition: all 0.4s !important;
#         position: relative;
#     }

#     [data-testid="stFileUploader"]::before {
#         content: '📄';
#         position: absolute;
#         top: 50%; left: 50%;
#         transform: translate(-50%, -50%);
#         font-size: 60px;
#         opacity: 0.06;
#         animation: floatUpload 3s ease infinite;
#     }

#     @keyframes floatUpload {
#         0%, 100% { transform: translate(-50%, -50%) scale(1); }
#         50% { transform: translate(-50%, -60%) scale(1.05); }
#     }

#     .stButton > button { /* Cyber button */ }

#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # robot watermark emoji
# st.markdown("""
# <div class="robot-watermark">🤖</div>
# """, unsafe_allow_html=True)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# logger = logging.getLogger(__name__)


# def extract_model_names(models_info: Any) -> Tuple[str, ...]:
#     logger.info("Extracting model names from models_info")
#     try:
#         if hasattr(models_info, "models"):
#             model_names = tuple(model.model for model in models_info.models)
#         else:
#             model_names = tuple()
#         logger.info(f"Extracted model names: {model_names}")
#         return model_names
#     except Exception as e:
#         logger.error(f"Error extracting model names: {e}")
#         return tuple()


# # preserve original single-file creation for compatibility
# def create_vector_db(file_upload) -> Chroma:
#     logger.info(f"Creating vector DB from file upload: {file_upload.name}")
#     temp_dir = tempfile.mkdtemp()

#     path = os.path.join(temp_dir, file_upload.name)
#     with open(path, "wb") as f:
#         f.write(file_upload.getvalue())
#         logger.info(f"File saved to temporary path: {path}")
#         loader = PyPDFLoader(path)
#         data = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#     chunks = text_splitter.split_documents(data)
#     logger.info("Document split into chunks")

#     embeddings = OllamaEmbeddings(model="nomic-embed-text")

#     vector_db = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=PERSIST_DIRECTORY,
#         collection_name=f"pdf_{hash(file_upload.name)}"
#     )
#     logger.info("Vector DB created with persistent storage")

#     shutil.rmtree(temp_dir)
#     logger.info(f"Temporary directory {temp_dir} removed")
#     return vector_db


# # New: create single vector DB from multiple uploaded files (keeps processing logic same downstream)
# def create_vector_db_from_files(file_uploads: List[Any], collection_name: str = "multi_pdf") -> Chroma:
#     logger.info(f"Creating combined vector DB from {len(file_uploads)} files")
#     temp_dir = tempfile.mkdtemp()
#     all_chunks = []

#     for file_upload in file_uploads:
#         path = os.path.join(temp_dir, file_upload.name)
#         with open(path, "wb") as f:
#             f.write(file_upload.getvalue())
#         loader = PyPDFLoader(path)
#         data = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#         chunks = text_splitter.split_documents(data)
#         # annotate metadata with filename so we can trace back
#         for doc in chunks:
#             if not hasattr(doc, 'metadata'):
#                 doc.metadata = {}
#             doc.metadata['source_file'] = file_upload.name
#         all_chunks.extend(chunks)
#         logger.info(f"Processed and split: {file_upload.name}")

#     embeddings = OllamaEmbeddings(model="nomic-embed-text")

#     vector_db = Chroma.from_documents(
#         documents=all_chunks,
#         embedding=embeddings,
#         persist_directory=PERSIST_DIRECTORY,
#         collection_name=collection_name
#     )
#     logger.info("Combined Vector DB created")

#     shutil.rmtree(temp_dir)
#     logger.info(f"Temporary directory {temp_dir} removed")
#     return vector_db


# # Aadhaar detection: simple 12-digit detection (improve as needed)
# def find_aadhaar_numbers_in_text(text: str) -> List[str]:
#     # remove non-digit separators and find 12-digit sequences
#     possible = re.findall(r"\b\d{4}\s?\d{4}\s?\d{4}\b|\b\d{12}\b", text)
#     # normalize: remove spaces
#     normalized = [re.sub(r"\s+", "", p) for p in possible]
#     return normalized


# # Extract pages as images from multiple files
# @st.cache_data
# def extract_pages_from_files(file_uploads: List[Any]) -> List[Any]:
#     logger.info("Extracting pages as images from multiple files")
#     pages = []
#     for file_upload in file_uploads:
#         with pdfplumber.open(file_upload) as pdf:
#             pages.extend([page.to_image().original for page in pdf.pages])
#     logger.info(f"Extracted total pages: {len(pages)}")
#     return pages


# def delete_vector_db(vector_db: Optional[Chroma]) -> None:
#     logger.info("Deleting vector DB")
#     if vector_db is not None:
#         try:
#             vector_db.delete_collection()

#             st.session_state.pop("pdf_pages", None)
#             st.session_state.pop("file_uploads", None)
#             st.session_state.pop("vector_db", None)

#             st.success("✅ Collection and temporary files deleted successfully.")
#             logger.info("Vector DB and related session state cleared")
#             st.rerun()
#         except Exception as e:
#             st.error(f"❌ Error deleting collection: {str(e)}")
#             logger.error(f"Error deleting collection: {e}")
#     else:
#         st.error("⚠️ No vector database found to delete.")
#         logger.warning("Attempted to delete vector DB, but none was found")


# def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
#     # unchanged core logic
#     logger.info(f"Processing question: {question} using model: {selected_model}")

#     llm = ChatOllama(model="llama3.1")

#     QUERY_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""You are an AI language model assistant. Your task is to generate 2
#         different versions of the given user question to retrieve relevant documents from
#         a vector database. By generating multiple perspectives on the user question, your
#         goal is to help the user overcome some of the limitations of the distance-based
#         similarity search. Provide these alternative questions separated by newlines.
#         Original question: {question}""",
#     )

#     retriever = MultiQueryRetriever.from_llm(
#         vector_db.as_retriever(), 
#         llm,
#         prompt=QUERY_PROMPT
#     )

#     template = """Answer the question based ONLY on the following context:
#     {context}
#     Question: {question}
#     """

#     prompt = ChatPromptTemplate.from_template(template)

#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     response = chain.invoke(question)
#     logger.info("Question processed and response generated")
#     return response


# def main() -> None:
#     # Header
#     st.markdown(
#         """
#         <div style='text-align: center; padding: 30px 0 20px 0;'>
#             <h1 style='font-size: 3.8em; margin: 0;'>
#                 <span style='font-size: 0.9em;'>🤖</span> AI PDF RAG ASSISTANT
#             </h1>
#             <p style='color: rgba(255,255,255,0.6); font-size: 1.05em; margin-top: 6px;'>
#                 ⚡ POWERED BY OLLAMA & LANGCHAIN ⚡
#             </p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     models_info = ollama.list()
#     available_models = extract_model_names(models_info)

#     col1, col2 = st.columns([1.6, 2])

#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []
#     if "vector_db" not in st.session_state:
#         st.session_state["vector_db"] = None
#     if "use_sample" not in st.session_state:
#         st.session_state["use_sample"] = False
#     if "file_uploads" not in st.session_state:
#         st.session_state["file_uploads"] = []
#     if "aadhaar_map" not in st.session_state:
#         st.session_state["aadhaar_map"] = {}  # aadhaar_number -> list of filenames

#     with col1:
#         st.markdown("### 📄 DOCUMENT MANAGEMENT")

#         if available_models:
#             selected_model = st.selectbox(
#                 "🎯 AI MODEL", 
#                 available_models,
#                 key="model_select"
#             )

#         use_sample = st.toggle(
#             "📚 USE SAMPLE PDF", 
#             key="sample_checkbox"
#         )

#         if use_sample != st.session_state.get("use_sample"):
#             if st.session_state["vector_db"] is not None:
#                 st.session_state["vector_db"].delete_collection()
#                 st.session_state["vector_db"] = None
#                 st.session_state["pdf_pages"] = None
#                 st.session_state["file_uploads"] = []
#             st.session_state["use_sample"] = use_sample

#         if use_sample:
#             sample_path = "data/pdfs/sample/scammer-agent.pdf"
#             if os.path.exists(sample_path):
#                 if st.session_state["vector_db"] is None:
#                     with st.spinner("🔄 PROCESSING SAMPLE PDF..."):
#                         loader = PyPDFLoader(sample_path)
#                         data = loader.load()
#                         text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#                         chunks = text_splitter.split_documents(data)
#                         st.session_state["vector_db"] = Chroma.from_documents(
#                             documents=chunks,
#                             embedding=OllamaEmbeddings(model="nomic-embed-text"),
#                             persist_directory=PERSIST_DIRECTORY,
#                             collection_name="sample_pdf"
#                         )
#                         with pdfplumber.open(sample_path) as pdf:
#                             st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
#             else:
#                 st.error("❌ Sample PDF not found")
#         else:
#             # Accept multiple PDFs now
#             uploaded_files = st.file_uploader(
#                 "📤 UPLOAD PDF(S)", 
#                 type="pdf", 
#                 accept_multiple_files=True,
#                 key="pdf_uploader"
#             )

#             # show currently added files and allow removal
#             if uploaded_files:
#                 # check duplicates by filename and aadhaar
#                 # first extract aadhaar for each file
#                 new_files = []
#                 new_aadhaar_map = {}
#                 progress = st.progress(0)
#                 for i, f in enumerate(uploaded_files):
#                     # skip if already in session_state by name to avoid reprocessing same upload
#                     if f.name in [x.name for x in st.session_state.get("file_uploads", [])]:
#                         logger.info(f"Skipping already processed file: {f.name}")
#                         progress.progress(int((i+1)/len(uploaded_files)*100))
#                         continue

#                     # extract text for aadhaar detection (fast path using PyPDFLoader)
#                     try:
#                         temp_dir = tempfile.mkdtemp()
#                         path = os.path.join(temp_dir, f.name)
#                         with open(path, "wb") as tmpf:
#                             tmpf.write(f.getvalue())
#                         loader = PyPDFLoader(path)
#                         docs = loader.load()
#                         all_text = "\n".join([d.page_content for d in docs])
#                         found = find_aadhaar_numbers_in_text(all_text)
#                         if found:
#                             new_aadhaar_map[f.name] = found
#                         else:
#                             new_aadhaar_map[f.name] = []
#                     except Exception as e:
#                         logger.error(f"Error extracting text for {f.name}: {e}")
#                         new_aadhaar_map[f.name] = []
#                     finally:
#                         shutil.rmtree(temp_dir)
#                     new_files.append(f)
#                     progress.progress(int((i+1)/len(uploaded_files)*100))

#                 # merge into session_state but avoid aadhaar duplicates across files
#                 duplicates = []
#                 for fname, found_list in new_aadhaar_map.items():
#                     for a in found_list:
#                         existing = st.session_state["aadhaar_map"].get(a, [])
#                         if existing:
#                             duplicates.append((a, existing + [fname]))

#                 if duplicates:
#                     st.error("❌ Duplicate Aadhaar detected across uploaded files. Those files were not added to avoid duplicates.")
#                     for a, files in duplicates:
#                         st.write(f"Aadhaar {a} appears in: {', '.join(files)}")

#                 # add only files that do not contain aadhaar already in aadhaar_map
#                 files_added = []
#                 for f in new_files:
#                     found_list = new_aadhaar_map.get(f.name, [])
#                     conflict = False
#                     for a in found_list:
#                         if a in st.session_state["aadhaar_map"]:
#                             conflict = True
#                             break
#                     if conflict:
#                         continue
#                     # safe to add
#                     st.session_state["file_uploads"].append(f)
#                     # update aadhaar_map
#                     for a in found_list:
#                         st.session_state["aadhaar_map"].setdefault(a, []).append(f.name)
#                     files_added.append(f.name)

#                 if files_added:
#                     st.success(f"✅ Added files: {', '.join(files_added)}")

#                 # provide option to force add duplicates (advanced)
#                 if st.checkbox("Force add duplicates (override Aadhaar checks)", value=False):
#                     for f in new_files:
#                         if f.name not in [x.name for x in st.session_state["file_uploads"]]:
#                             st.session_state["file_uploads"].append(f)
#                     st.warning("⚠️ You forced adding duplicates. Be careful — this may create duplicate records.")

#                 # now, create or update vector_db if any new files were actually added
#                 if st.session_state["file_uploads"] and st.session_state["vector_db"] is None:
#                     with st.spinner("🔄 PROCESSING PDF(S) AND CREATING VECTOR DB..."):
#                         st.session_state["vector_db"] = create_vector_db_from_files(st.session_state["file_uploads"], collection_name="multi_pdf")
#                         st.session_state["pdf_pages"] = extract_pages_from_files(st.session_state["file_uploads"])
#             else:
#                 # no files currently uploaded
#                 pass

#         # show uploaded files summary and basic insights
#         if st.session_state.get("file_uploads"):
#             st.markdown("---")
#             st.markdown("**Uploaded files:**")
#             for f in st.session_state["file_uploads"]:
#                 aad_list = [k for k, v in st.session_state["aadhaar_map"].items() if f.name in v]
#                 badge = f" — Aadhaar found: {', '.join(aad_list)}" if aad_list else ""
#                 st.write(f"• {f.name}{badge}")

#             # quick stats
#             total_pages = len(st.session_state.get("pdf_pages", []))
#             st.write(f"**Total pages:** {total_pages}")

#         if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
#             st.markdown("---")
#             zoom_level = st.slider(
#                 "🔍 ZOOM", 
#                 min_value=100, 
#                 max_value=1000, 
#                 value=700, 
#                 step=50,
#                 key="zoom_slider"
#             )

#             with st.container():
#                 for page_image in st.session_state["pdf_pages"]:
#                     st.image(page_image, width=zoom_level)

#         st.markdown("---")
#         delete_collection = st.button(
#             "🗑️ DELETE COLLECTION", 
#             type="secondary",
#             key="delete_button",
#             use_container_width=True
#         )

#         if delete_collection:
#             delete_vector_db(st.session_state["vector_db"])

#     with col2:
#         st.markdown("### 💬 NEURAL CHAT INTERFACE")

#         chat_container = st.container()

#         with chat_container:
#             for msg in st.session_state["messages"]:
#                 if msg["role"] == "user":
#                     with st.chat_message("user", avatar="👤"):
#                         st.markdown(msg["content"])
#                 else:
#                     with st.chat_message("assistant", avatar="🤖"):
#                         st.markdown(msg["content"])

#         if prompt := st.chat_input("💭 ASK ME ANYTHING...", key="chat_input"):
#             try:
#                 st.session_state["messages"].append({"role": "user", "content": prompt})

#                 if st.session_state["vector_db"] is not None:
#                     with st.spinner("🧠 NEURAL PROCESSING..."):
#                         response = process_question(
#                             prompt, st.session_state["vector_db"], selected_model
#                         )

#                     st.session_state["messages"].append(
#                         {"role": "assistant", "content": response}
#                     )
#                     st.rerun()
#                 else:
#                     st.warning("⚠️ UPLOAD AT LEAST ONE PDF TO START CHATTING!")

#             except Exception as e:
#                 st.error(f"❌ ERROR: {str(e)}")
#                 logger.error(f"Error processing prompt: {e}")
#         else:
#             if st.session_state["vector_db"] is None:
#                 st.info("👋 UPLOAD PDF(S) TO START CHATTING!")


# if __name__ == "__main__":
#     main()



# """
# Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.
# """

# import streamlit as st
# import logging
# import os
# import tempfile
# import shutil
# import pdfplumber
# import ollama
# import warnings

# warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from typing import List, Tuple, Dict, Any, Optional



# def get_ollama_embeddings():
#     os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

#     return OllamaEmbeddings(
#         model="nomic-embed-text",
#         base_url="http://127.0.0.1:11434",
#     )

# PERSIST_DIRECTORY = os.path.join("data", "vectors")

# st.set_page_config(
#     page_title="QUERYNEXUS:AI PDF Assistant",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# # ULTIMATE AI UI
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;600;700&display=swap');

# * { font-family: 'Space Grotesk', sans-serif; }

# .stApp {
#     background: #000000;
#     color: #fff;
# }

# /* Animated Neural Background */
# .main::before {
#     content: '';
#     position: fixed;
#     top: 0; left: 0;
#     width: 100%; height: 100%;
#     background: 
#         radial-gradient(ellipse at 10% 20%, rgba(99,102,241,0.15), transparent 50%),
#         radial-gradient(ellipse at 90% 80%, rgba(139,92,246,0.15), transparent 50%),
#         radial-gradient(ellipse at 50% 50%, rgba(236,72,153,0.1), transparent 50%),
#         linear-gradient(180deg, #000 0%, #0a0a1a 50%, #000 100%);
#     z-index: -2;
#     animation: bgPulse 20s ease infinite;
# }

# @keyframes bgPulse {
#     0%, 100% { opacity: 1; }
#     50% { opacity: 0.8; }
# }

# /* Animated Grid */
# .main::after {
#     content: '';
#     position: fixed;
#     top: 0; left: 0;
#     width: 100%; height: 100%;
#     background-image: 
#         linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
#         linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px);
#     background-size: 50px 50px;
#     z-index: -1;
#     animation: gridMove 20s linear infinite;
# }

# @keyframes gridMove {
#     0% { transform: translate(0, 0); }
#     100% { transform: translate(50px, 50px); }
# }

# /* Holographic Headers */
# h1, h2, h3 {
#     font-weight: 700 !important;
#     background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 25%, #ec4899 50%, #8b5cf6 75%, #6366f1 100%);
#     background-size: 200% auto;
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     animation: holoShift 8s ease infinite;
# }

# @keyframes holoShift {
#     0%, 100% { background-position: 0% 50%; }
#     50% { background-position: 100% 50%; }
# }

# .main h1 {
#     font-size: 3.5em !important;
#     text-shadow: 0 0 30px rgba(99,102,241,0.5), 0 0 60px rgba(139,92,246,0.3);
#     animation: holoShift 8s ease infinite, floatTitle 6s ease infinite;
# }

# @keyframes floatTitle {
#     0%, 100% { transform: translateY(0); }
#     50% { transform: translateY(-10px); }
# }

# /* Glassmorphic Containers */
# [data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
#     background: rgba(15,15,35,0.6) !important;
#     backdrop-filter: blur(20px) saturate(180%) !important;
#     border-radius: 24px !important;
#     border: 1px solid rgba(99,102,241,0.2) !important;
#     box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1) !important;
#     position: relative;
#     overflow: hidden;
#     transition: all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
# }

# [data-testid="stVerticalBlock"] > [data-testid="stContainer"]::before {
#     content: '';
#     position: absolute;
#     top: 0; left: -100%;
#     width: 100%; height: 100%;
#     background: linear-gradient(90deg, transparent, rgba(99,102,241,0.1), transparent);
#     transition: left 0.7s;
# }

# [data-testid="stVerticalBlock"] > [data-testid="stContainer"]:hover::before {
#     left: 100%;
# }

# [data-testid="stVerticalBlock"] > [data-testid="stContainer"]:hover {
#     transform: translateY(-4px) scale(1.01);
#     border-color: rgba(99,102,241,0.4) !important;
#     box-shadow: 0 16px 48px rgba(99,102,241,0.3) !important;
# }

# /* Advanced Chat Messages */
# [data-testid="stChatMessage"] {
#     background: rgba(15,15,35,0.6) !important;
#     backdrop-filter: blur(20px) !important;
#     border-radius: 20px !important;
#     padding: 20px !important;
#     margin: 12px 0 !important;
#     border: 1px solid rgba(99,102,241,0.2) !important;
#     position: relative !important;
#     overflow: hidden !important;
# }

# [data-testid="stChatMessage"][class*="user"] {
#     background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3)) !important;
#     border: 1px solid rgba(99,102,241,0.4) !important;
#     box-shadow: 0 8px 32px rgba(99,102,241,0.3) !important;
#     animation: slideInRight 0.6s cubic-bezier(0.34,1.56,0.64,1) !important;
# }

# [data-testid="stChatMessage"][class*="user"]::before {
#     content: '';
#     position: absolute;
#     top: -50%; left: -50%;
#     width: 200%; height: 200%;
#     background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
#     transform: rotate(45deg);
#     animation: shimmer 3s infinite;
# }

# @keyframes shimmer {
#     0% { transform: translateX(-100%) rotate(45deg); }
#     100% { transform: translateX(100%) rotate(45deg); }
# }

# [data-testid="stChatMessage"][class*="assistant"] {
#     background: linear-gradient(135deg, rgba(236,72,153,0.2), rgba(139,92,246,0.2)) !important;
#     border: 1px solid rgba(236,72,153,0.3) !important;
#     box-shadow: 0 8px 32px rgba(236,72,153,0.2) !important;
#     animation: slideInLeft 0.6s cubic-bezier(0.34,1.56,0.64,1) !important;
# }

# [data-testid="stChatMessage"][class*="assistant"]::after {
#     content: '✨';
#     position: absolute;
#     top: 10px; right: 10px;
#     font-size: 20px;
#     animation: sparkle 2s ease infinite;
# }

# @keyframes sparkle {
#     0%, 100% { opacity: 0.5; transform: scale(1) rotate(0deg); }
#     50% { opacity: 1; transform: scale(1.2) rotate(180deg); }
# }

# @keyframes slideInRight {
#     0% { opacity: 0; transform: translateX(50px) scale(0.8); }
#     100% { opacity: 1; transform: translateX(0) scale(1); }
# }

# @keyframes slideInLeft {
#     0% { opacity: 0; transform: translateX(-50px) scale(0.8); }
#     100% { opacity: 1; transform: translateX(0) scale(1); }
# }

# /* Futuristic Input */
# .stChatInputContainer {
#     background: rgba(15,15,35,0.8) !important;
#     backdrop-filter: blur(20px) !important;
#     border-radius: 20px !important;
#     border: 2px solid rgba(99,102,241,0.3) !important;
#     padding: 8px !important;
#     box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(99,102,241,0.2) !important;
#     animation: inputGlow 3s ease infinite;
# }

# @keyframes inputGlow {
#     0%, 100% { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(99,102,241,0.2); }
#     50% { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 30px rgba(99,102,241,0.4); }
# }

# .stChatInputContainer::before {
#     content: '';
#     position: absolute;
#     top: 0; left: -100%;
#     width: 100%; height: 100%;
#     background: linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent);
#     animation: scanLine 3s linear infinite;
# }

# @keyframes scanLine {
#     0% { left: -100%; }
#     100% { left: 100%; }
# }

# .stChatInputContainer textarea {
#     background: transparent !important;
#     color: #fff !important;
#     border: none !important;
#     font-size: 16px !important;
# }

# /* Cyber Buttons */
# .stButton > button {
#     background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%) !important;
#     background-size: 200% auto;
#     color: white !important;
#     border: none !important;
#     border-radius: 16px !important;
#     padding: 14px 32px !important;
#     font-weight: 600 !important;
#     box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
#     transition: all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
#     overflow: hidden;
#     text-transform: uppercase;
#     letter-spacing: 1px;
# }

# .stButton > button::before {
#     content: '';
#     position: absolute;
#     top: 0; left: -100%;
#     width: 100%; height: 100%;
#     background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
#     transition: left 0.5s;
# }

# .stButton > button:hover::before {
#     left: 100%;
# }

# .stButton > button:hover {
#     transform: translateY(-3px) scale(1.05) !important;
#     background-position: 100% 0;
#     box-shadow: 0 8px 30px rgba(99,102,241,0.6), 0 0 40px rgba(139,92,246,0.4) !important;
# }

# .stButton > button[kind="secondary"] {
#     background: linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #b91c1c 100%) !important;
#     box-shadow: 0 4px 20px rgba(239,68,68,0.4) !important;
# }

# /* File Uploader */
# [data-testid="stFileUploader"] {
#     background: rgba(15,15,35,0.6) !important;
#     backdrop-filter: blur(20px) !important;
#     border-radius: 20px !important;
#     border: 2px dashed rgba(99,102,241,0.4) !important;
#     padding: 30px !important;
#     transition: all 0.4s !important;
#     position: relative;
# }

# [data-testid="stFileUploader"]::before {
#     content: '📄';
#     position: absolute;
#     top: 50%; left: 50%;
#     transform: translate(-50%, -50%);
#     font-size: 60px;
#     opacity: 0.1;
#     animation: floatUpload 3s ease infinite;
# }

# @keyframes floatUpload {
#     0%, 100% { transform: translate(-50%, -50%) scale(1); }
#     50% { transform: translate(-50%, -60%) scale(1.1); }
# }

# [data-testid="stFileUploader"]:hover {
#     border-color: rgba(99,102,241,0.8) !important;
#     background: rgba(99,102,241,0.1) !important;
#     transform: scale(1.02);
#     box-shadow: 0 0 40px rgba(99,102,241,0.3);
# }

# /* Selectbox */
# .stSelectbox > div > div {
#     background: rgba(15,15,35,0.8) !important;
#     backdrop-filter: blur(20px) !important;
#     border: 1px solid rgba(99,102,241,0.3) !important;
#     border-radius: 16px !important;
#     color: white !important;
#     transition: all 0.3s !important;
# }

# /* Slider */
# .stSlider > div > div > div {
#     background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899) !important;
# }

# .stSlider > div > div > div > div {
#     background: white !important;
#     box-shadow: 0 0 20px rgba(99,102,241,0.6) !important;
#     width: 24px !important;
#     height: 24px !important;
# }

# .stSlider > div > div > div > div:hover {
#     transform: scale(1.3);
#     box-shadow: 0 0 30px rgba(99,102,241,0.8) !important;
# }

# /* Checkbox */
# .stCheckbox {
#     background: rgba(15,15,35,0.6) !important;
#     backdrop-filter: blur(12px) !important;
#     padding: 16px 20px !important;
#     border-radius: 16px !important;
#     border: 1px solid rgba(99,102,241,0.3) !important;
#     transition: all 0.3s !important;
# }

# .stCheckbox:hover {
#     background: rgba(99,102,241,0.15) !important;
#     transform: translateX(4px);
# }

# /* Spinner */
# .stSpinner > div {
#     border-top-color: #6366f1 !important;
#     border-right-color: #8b5cf6 !important;
#     border-bottom-color: #ec4899 !important;
#     width: 50px !important;
#     height: 50px !important;
#     animation: spinNeutral 1s cubic-bezier(0.68,-0.55,0.27,1.55) infinite !important;
# }

# @keyframes spinNeutral {
#     0% { transform: rotate(0deg) scale(1); box-shadow: 0 0 20px rgba(99,102,241,0.4); }
#     50% { transform: rotate(180deg) scale(1.1); box-shadow: 0 0 40px rgba(139,92,246,0.6); }
#     100% { transform: rotate(360deg) scale(1); box-shadow: 0 0 20px rgba(236,72,153,0.4); }
# }

# /* Notifications */
# .stSuccess, .stWarning, .stError, .stInfo {
#     background: rgba(15,15,35,0.8) !important;
#     backdrop-filter: blur(20px) !important;
#     border-radius: 16px !important;
#     border-left: 4px solid !important;
#     padding: 20px !important;
#     animation: slideNotif 0.5s cubic-bezier(0.4,0,0.2,1) !important;
# }

# @keyframes slideNotif {
#     0% { opacity: 0; transform: translateX(-50px); }
#     100% { opacity: 1; transform: translateX(0); }
# }

# .stSuccess {
#     border-left-color: #10b981 !important;
#     background: rgba(16,185,129,0.15) !important;
# }

# .stWarning {
#     border-left-color: #f59e0b !important;
#     background: rgba(245,158,11,0.15) !important;
# }

# .stError {
#     border-left-color: #ef4444 !important;
#     background: rgba(239,68,68,0.15) !important;
# }

# /* Scrollbar */
# ::-webkit-scrollbar {
#     width: 12px;
# }

# ::-webkit-scrollbar-track {
#     background: rgba(15,15,35,0.6);
#     border-radius: 10px;
# }

# ::-webkit-scrollbar-thumb {
#     background: linear-gradient(180deg, #6366f1, #8b5cf6, #ec4899);
#     border-radius: 10px;
#     box-shadow: 0 0 10px rgba(99,102,241,0.5);
# }

# ::-webkit-scrollbar-thumb:hover {
#     box-shadow: 0 0 20px rgba(139,92,246,0.8);
# }

# /* Image Enhancement */
# .stImage {
#     border-radius: 16px !important;
#     box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
#     transition: all 0.3s !important;
#     border: 1px solid rgba(99,102,241,0.2) !important;
# }

# .stImage:hover {
#     transform: scale(1.02);
#     box-shadow: 0 12px 48px rgba(99,102,241,0.3) !important;
# }

# /* Icon Animations */
# .icon-float {
#     display: inline-block;
#     animation: iconFloat 3s ease infinite;
# }

# @keyframes iconFloat {
#     0%, 100% { transform: translateY(0) rotate(0deg); }
#     50% { transform: translateY(-15px) rotate(5deg); }
# }
# </style>
# """, unsafe_allow_html=True)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# logger = logging.getLogger(__name__)


# def extract_model_names(models_info: Any) -> Tuple[str, ...]:
#     logger.info("Extracting model names from models_info")
#     try:
#         if hasattr(models_info, "models"):
#             model_names = tuple(model.model for model in models_info.models)
#         else:
#             model_names = tuple()
#         logger.info(f"Extracted model names: {model_names}")
#         return model_names
#     except Exception as e:
#         logger.error(f"Error extracting model names: {e}")
#         return tuple()


# def create_vector_db(file_upload) -> Chroma:
#     logger.info(f"Creating vector DB from file upload: {file_upload.name}")
#     temp_dir = tempfile.mkdtemp()

#     path = os.path.join(temp_dir, file_upload.name)
#     with open(path, "wb") as f:
#         f.write(file_upload.getvalue())
#         logger.info(f"File saved to temporary path: {path}")
#         loader = PyPDFLoader(path)
#         data = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#     chunks = text_splitter.split_documents(data)
#     logger.info("Document split into chunks")

#     embeddings = OllamaEmbeddings(model="nomic-embed-text")

#     vector_db = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=PERSIST_DIRECTORY,
#         collection_name=f"pdf_{hash(file_upload.name)}"
#     )
#     logger.info("Vector DB created with persistent storage")

#     shutil.rmtree(temp_dir)
#     logger.info(f"Temporary directory {temp_dir} removed")
#     return vector_db


# def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
#     logger.info(f"Processing question: {question} using model: {selected_model}")
    
#     llm = ChatOllama(model="llama3.1")
    
#     QUERY_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""You are an AI language model assistant. Your task is to generate 2
#         different versions of the given user question to retrieve relevant documents from
#         a vector database. By generating multiple perspectives on the user question, your
#         goal is to help the user overcome some of the limitations of the distance-based
#         similarity search. Provide these alternative questions separated by newlines.
#         Original question: {question}""",
#     )

#     retriever = MultiQueryRetriever.from_llm(
#         vector_db.as_retriever(), 
#         llm,
#         prompt=QUERY_PROMPT
#     )

#     template = """Answer the question based ONLY on the following context:
#     {context}
#     Question: {question}
#     """

#     prompt = ChatPromptTemplate.from_template(template)

#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     response = chain.invoke(question)
#     logger.info("Question processed and response generated")
#     return response


# @st.cache_data
# def extract_all_pages_as_images(file_upload) -> List[Any]:
#     logger.info(f"Extracting all pages as images from file: {file_upload.name}")
#     pdf_pages = []
#     with pdfplumber.open(file_upload) as pdf:
#         pdf_pages = [page.to_image().original for page in pdf.pages]
#     logger.info("PDF pages extracted as images")
#     return pdf_pages


# def delete_vector_db(vector_db: Optional[Chroma]) -> None:
#     logger.info("Deleting vector DB")
#     if vector_db is not None:
#         try:
#             vector_db.delete_collection()
            
#             st.session_state.pop("pdf_pages", None)
#             st.session_state.pop("file_upload", None)
#             st.session_state.pop("vector_db", None)
            
#             st.success("✅ Collection and temporary files deleted successfully.")
#             logger.info("Vector DB and related session state cleared")
#             st.rerun()
#         except Exception as e:
#             st.error(f"❌ Error deleting collection: {str(e)}")
#             logger.error(f"Error deleting collection: {e}")
#     else:
#         st.error("⚠️ No vector database found to delete.")
#         logger.warning("Attempted to delete vector DB, but none was found")


# def main() -> None:
#     # Animated holographic header
#     st.markdown(
#         """
#         <div style='text-align: center; padding: 30px 0 20px 0;'>
#             <h1 style='font-size: 4em; margin: 0;'>
#                 <span class='icon-float'>🤖</span> QUERYNEXUS:AI PDF ASSISTANT
#             </h1>
#             <p style='color: rgba(255,255,255,0.5); font-size: 1.2em; margin-top: 15px; letter-spacing: 2px;'>
#                 ⚡ POWERED BY OLLAMA & LANGCHAIN ⚡
#             </p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     models_info = ollama.list()
#     available_models = extract_model_names(models_info)

#     col1, col2 = st.columns([1.5, 2])

#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []
#     if "vector_db" not in st.session_state:
#         st.session_state["vector_db"] = None
#     if "use_sample" not in st.session_state:
#         st.session_state["use_sample"] = False

#     with col1:
#         st.markdown("### 📄 DOCUMENT MANAGEMENT")
        
#         if available_models:
#             selected_model = st.selectbox(
#                 "🎯 AI MODEL", 
#                 available_models,
#                 key="model_select"
#             )

#         use_sample = st.toggle(
#             "📚 USE SAMPLE PDF", 
#             key="sample_checkbox"
#         )
        
#         if use_sample != st.session_state.get("use_sample"):
#             if st.session_state["vector_db"] is not None:
#                 st.session_state["vector_db"].delete_collection()
#                 st.session_state["vector_db"] = None
#                 st.session_state["pdf_pages"] = None
#             st.session_state["use_sample"] = use_sample

#         if use_sample:
#             sample_path = "data/pdfs/sample/scammer-agent.pdf"
#             if os.path.exists(sample_path):
#                 if st.session_state["vector_db"] is None:
#                     with st.spinner("🔄 PROCESSING SAMPLE PDF..."):
#                         loader = PyPDFLoader(sample_path)
#                         data = loader.load()
#                         text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#                         chunks = text_splitter.split_documents(data)
#                         st.session_state["vector_db"] = Chroma.from_documents(
#                             documents=chunks,
#                             embedding=OllamaEmbeddings(model="nomic-embed-text"),
#                             persist_directory=PERSIST_DIRECTORY,
#                             collection_name="sample_pdf"
#                         )
#                         with pdfplumber.open(sample_path) as pdf:
#                             st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
#             else:
#                 st.error("❌ Sample PDF not found")
#         else:
#             file_upload = st.file_uploader(
#                 "📤 UPLOAD PDF", 
#                 type="pdf", 
#                 accept_multiple_files=False,
#                 key="pdf_uploader"
#             )

#             if file_upload:
#                 if st.session_state["vector_db"] is None:
#                     with st.spinner("🔄 PROCESSING PDF..."):
#                         st.session_state["vector_db"] = create_vector_db(file_upload)
#                         st.session_state["file_upload"] = file_upload
#                         with pdfplumber.open(file_upload) as pdf:
#                             st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]

#         if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
#             st.markdown("---")
#             zoom_level = st.slider(
#                 "🔍 ZOOM", 
#                 min_value=100, 
#                 max_value=1000, 
#                 value=700, 
#                 step=50,
#                 key="zoom_slider"
#             )

#             with st.container(height=410, border=True):
#                 for page_image in st.session_state["pdf_pages"]:
#                     st.image(page_image, width=zoom_level)

#         st.markdown("---")
#         delete_collection = st.button(
#             "🗑️ DELETE COLLECTION", 
#             type="secondary",
#             key="delete_button",
#             use_container_width=True
#         )

#         if delete_collection:
#             delete_vector_db(st.session_state["vector_db"])

#     with col2:
#         st.markdown("### 💬 NEURAL CHAT INTERFACE")
        
#         chat_container = st.container(height=520, border=True)
        
#         with chat_container:
#             for msg in st.session_state["messages"]:
#                 if msg["role"] == "user":
#                     with st.chat_message("user", avatar="👤"):
#                         st.markdown(msg["content"])
#                 else:
#                     with st.chat_message("assistant", avatar="🤖"):
#                         st.markdown(msg["content"])

#         if prompt := st.chat_input("💭 ASK ME ANYTHING...", key="chat_input"):
#             try:
#                 st.session_state["messages"].append({"role": "user", "content": prompt})

#                 if st.session_state["vector_db"] is not None:
#                     with st.spinner("🧠 NEURAL PROCESSING..."):
#                         response = process_question(
#                             prompt, st.session_state["vector_db"], selected_model
#                         )
                    
#                     st.session_state["messages"].append(
#                         {"role": "assistant", "content": response}
#                     )
#                     st.rerun()
#                 else:
#                     st.warning("⚠️ UPLOAD A PDF FIRST")

#             except Exception as e:
#                 st.error(f"❌ ERROR: {str(e)}")
#                 logger.error(f"Error processing prompt: {e}")
#         else:
#             if st.session_state["vector_db"] is None:
#                 st.info("👋 UPLOAD A PDF TO START CHATTING!")


# if __name__ == "__main__":
#     main()

"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

Enhanced Features:
- Document deduplication (avoid duplicate PDFs)
- Multi-PDF querying (upload and search across multiple documents)
- Password-protected PDF support
- OCR for scanned PDFs
- Voice query input (mic → text → answer)
"""
