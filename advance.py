import os
import streamlit as st
from PyPDF2 import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime
import tempfile
import io
import re
import time
import hashlib
import json
import shutil
import time
import random
from pathlib import Path
from threading import Lock
from collections import defaultdict

# Load environment variables
load_dotenv()

# Configuration
MAX_REQUESTS_PER_MINUTE = 1000
MAX_UPLOAD_SIZE_MB = 50
CACHE_TTL = 3600  # 1 hour
MAX_CACHED_ENTRIES = 10

# Initialize Google API
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

class UserVectorStore:
    """
    Enhanced vector store manager for multi-user support with improved concurrency handling
    and session management.
    """
    def __init__(self, cleanup_interval: int = 3600):
        """
        Initialize the UserVectorStore.
        
        Args:
            cleanup_interval (int): Interval in seconds for cleaning up old stores
        """
        self.base_dir = Path(os.getenv("VECTOR_STORE_PATH", "user_vector_stores"))
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.locks = {}  # Dictionary to store locks per user
        self.global_lock = Lock()  # Global lock for user management
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        
    def _get_user_lock(self, user_id: str) -> Lock:
        """
        Get or create a lock for a specific user.
        
        Args:
            user_id (str): Unique user identifier
            
        Returns:
            threading.Lock: Lock object for the user
        """
        with self.global_lock:
            if user_id not in self.locks:
                self.locks[user_id] = Lock()
            return self.locks[user_id]
    
    def get_user_dir(self, user_id: str) -> Path:
        """
        Get directory for specific user.
        
        Args:
            user_id (str): Unique user identifier
            
        Returns:
            Path: Path object for the user directory
        """
        user_dir = self.base_dir / user_id
        user_dir.mkdir(exist_ok=True, parents=True)
        return user_dir
    
    def get_session_dir(self, user_id: str, session_id: str) -> Path:
        """
        Get directory for specific session.
        
        Args:
            user_id (str): Unique user identifier
            session_id (str): Unique session identifier
            
        Returns:
            Path: Path object for the session directory
        """
        session_dir = self.get_user_dir(user_id) / session_id
        session_dir.mkdir(exist_ok=True, parents=True)
        return session_dir
    
    def cleanup_old_stores(self, max_age: int = 3600):
        """
        Clean up vector stores older than specified age.
        
        Args:
            max_age (int): Maximum age in seconds for stores to keep
        """
        current_time = time.time()
        
        # Only cleanup if enough time has passed since last cleanup
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        with self.global_lock:
            try:
                # Iterate through all user directories
                for user_dir in self.base_dir.iterdir():
                    if not user_dir.is_dir():
                        continue
                        
                    # Get user lock
                    user_id = user_dir.name
                    user_lock = self._get_user_lock(user_id)
                    
                    with user_lock:
                        # Cleanup user's sessions
                        for session_dir in user_dir.iterdir():
                            if not session_dir.is_dir():
                                continue
                                
                            # Check if session is too old
                            if current_time - session_dir.stat().st_mtime > max_age:
                                try:
                                    shutil.rmtree(session_dir)
                                except Exception as e:
                                    st.error(f"Error cleaning up session {session_dir}: {str(e)}")
                                    
                        # Remove empty user directory
                        if not any(user_dir.iterdir()):
                            user_dir.rmdir()
                            with self.global_lock:
                                self.locks.pop(user_id, None)
                                
                self.last_cleanup = current_time
                
            except Exception as e:
                st.error(f"Error during cleanup: {str(e)}")
    
    def save_store(self, user_id: str, session_id: str, store: FAISS, metadata: dict):
        """
        Save vector store for a session.
        
        Args:
            user_id (str): Unique user identifier
            session_id (str): Unique session identifier
            store (FAISS): FAISS vector store object
            metadata (dict): Metadata associated with the store
        """
        user_lock = self._get_user_lock(user_id)
        with user_lock:
            try:
                session_dir = self.get_session_dir(user_id, session_id)
                store.save_local(str(session_dir / "store.faiss"))
                
                # Add timestamp to metadata
                metadata['last_modified'] = datetime.now().isoformat()
                
                with open(session_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f)
                    
            except Exception as e:
                st.error(f"Error saving store: {str(e)}")
                raise
    
    def load_store(self, user_id: str, session_id: str) -> tuple[FAISS | None, dict | None]:
        """
        Load vector store for a session.
        
        Args:
            user_id (str): Unique user identifier
            session_id (str): Unique session identifier
            
        Returns:
            tuple: (FAISS store object, metadata dictionary) or (None, None) if not found
        """
        user_lock = self._get_user_lock(user_id)
        with user_lock:
            try:
                session_dir = self.get_session_dir(user_id, session_id)
                store_path = session_dir / "store.faiss"
                meta_path = session_dir / "metadata.json"
                
                if store_path.exists() and meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if store is still valid (not too old)
                    last_modified = datetime.fromisoformat(metadata.get('last_modified', '2000-01-01'))
                    if datetime.now() - last_modified > timedelta(hours=1):
                        self.delete_store(user_id, session_id)
                        return None, None
                    
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    store = FAISS.load_local(str(store_path), embeddings)
                    return store, metadata
                    
                return None, None
                
            except Exception as e:
                st.error(f"Error loading store: {str(e)}")
                return None, None
    
    def delete_store(self, user_id: str, session_id: str):
        """
        Delete vector store for a session.
        
        Args:
            user_id (str): Unique user identifier
            session_id (str): Unique session identifier
        """
        user_lock = self._get_user_lock(user_id)
        with user_lock:
            try:
                session_dir = self.get_session_dir(user_id, session_id)
                if session_dir.exists():
                    shutil.rmtree(session_dir)
                    
                # Clean up empty user directory if needed
                user_dir = self.get_user_dir(user_id)
                if not any(user_dir.iterdir()):
                    user_dir.rmdir()
                    with self.global_lock:
                        self.locks.pop(user_id, None)
                        
            except Exception as e:
                st.error(f"Error deleting store: {str(e)}")
                raise

    def get_user_stats(self, user_id: str) -> dict:
        """
        Get statistics for a specific user.
        
        Args:
            user_id (str): Unique user identifier
            
        Returns:
            dict: Dictionary containing user statistics
        """
        user_lock = self._get_user_lock(user_id)
        with user_lock:
            try:
                user_dir = self.get_user_dir(user_id)
                active_sessions = 0
                total_stores = 0
                latest_activity = None
                
                for session_dir in user_dir.iterdir():
                    if not session_dir.is_dir():
                        continue
                        
                    meta_path = session_dir / "metadata.json"
                    if meta_path.exists():
                        with open(meta_path, 'r') as f:
                            metadata = json.load(f)
                        last_modified = datetime.fromisoformat(metadata.get('last_modified', '2000-01-01'))
                        
                        if datetime.now() - last_modified <= timedelta(hours=1):
                            active_sessions += 1
                            
                        if latest_activity is None or last_modified > latest_activity:
                            latest_activity = last_modified
                            
                        total_stores += 1
                
                return {
                    'active_sessions': active_sessions,
                    'total_stores': total_stores,
                    'latest_activity': latest_activity.isoformat() if latest_activity else None
                }
                
            except Exception as e:
                st.error(f"Error getting user stats: {str(e)}")
                return {
                    'active_sessions': 0,
                    'total_stores': 0,
                    'latest_activity': None
                }
# Initialize vector store manager
vector_store_manager = UserVectorStore(cleanup_interval=3600)  # Cleanup every hour

def get_session_id():
    """Generate or retrieve session ID"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.sha256(
            str(time.time() + random.random()).encode()
        ).hexdigest()[:16]
    return st.session_state.session_id

def initialize_session_state():
    """Initialize or reset session state with user and session management"""
    
    # Initialize user_id if not present
    if 'user_id' not in st.session_state:
        st.session_state.user_id = hashlib.sha256(
            str(time.time()).encode()
        ).hexdigest()[:16]
    
    # Initialize session_id if not present
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.sha256(
            str(time.time() + random.random()).encode()
        ).hexdigest()[:16]
    
    # Initialize session state variables
    if 'session_init' not in st.session_state:
        st.session_state.session_init = True
        # Clean up previous session data with both user_id and session_id
        vector_store_manager.delete_store(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id
        )
    
    # Initialize other session state variables
    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = []
    
    if 'figures' not in st.session_state:
        st.session_state.figures = []
    
    if 'asked_questions' not in st.session_state:
        st.session_state.asked_questions = {}
def validate_pdf_size(pdf):
    return pdf.size <= MAX_UPLOAD_SIZE_MB * 1024 * 1024

def get_study_chain(answer_length="medium"):
    length_guides = {
        "small": """
        Provide a concise response (150-200 words) that directly answers the question.
        Focus on clarity, key points, and relevance to the question.
        Include diagrams if explicitly mentioned in the question .
        Present the answer using a combination of paragraphs and bullet points.
        Ensure the response is suitable for theoretical exam answers.
        If no relevant content is found in the PDF, state clearly: 'No relevant content found in the provided document.'
        """,
        
        "medium": """
        Provide a balanced response (250-300 words) with clarity and focus.
        Address the question comprehensively while maintaining readability.
        Include diagrams if explicitly mentioned in the question or if they significantly enhance understanding.
        Present the answer using a combination of paragraphs and bullet points.
        Ensure the response is suitable for theoretical exam answers.
        If no relevant content is found in the PDF, state clearly: 'No relevant content found in the provided document.'
        """,
        
        "large": """
        Provide a detailed response (400-500 words) with depth and clarity.
        Address all important aspects of the question comprehensively.
        Include diagrams if explicitly mentioned in the question or if they significantly enhance understanding.
        Present the answer using a combination of paragraphs and bullet points.
        Ensure the response is clear, logically structured, and suitable for theoretical exam answers.
        If no relevant content is found in the PDF, state clearly: 'No relevant content found in the provided document.'
        """
    }
    
    prompt_template = f"""
    Answer the question thoughtfully with clarity and focus:
    
    {length_guides[answer_length]}
    
    Only respond if relevant content is found.
    Include diagrams if explicitly mentioned in the question or if they significantly enhance understanding.
    Present the answer using a combination of paragraphs and bullet points.
    Include text references (e.g., Section X.Y or X.Y.Z) from the PDF at the end .
    If no relevant content is found, state: 'No relevant content found in the provided document.'
    
    Context: {{context}}
    Question: {{question}}
    
    Answer:
    """
    
    token_limits = {
        "small": 300,
        "medium": 600,
        "large": 1000
    }
    
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_output_tokens=token_limits[answer_length]
    )
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_pdf_text(pdf_docs):
    text = ""
    figures = []
    
    for pdf in pdf_docs:
        if not validate_pdf_size(pdf):
            st.error(f"PDF {pdf.name} exceeds size limit of {MAX_UPLOAD_SIZE_MB}MB")
            continue
            
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(pdf.read())
                reader = PdfReader(tmp.name)
                
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        for match in re.finditer(r"(?:Figure|Fig\.?)\s+(\d+(?:\.\d+)?)\s*(?:[:.]\s*)?([^.]+)", page_text, re.I):
                            figures.append({
                                "id": match.group(1),
                                "caption": match.group(2).strip(),
                                "page": page_num
                            })
                        text += f"\n{page_text}"
            os.unlink(tmp.name)
        except Exception as e:
            st.error(f"Error processing PDF {pdf.name}: {str(e)}")
            
    return text, figures

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", ", ", " "]
    )
    return splitter.split_text(text)

def get_vector_store(pdf_docs):
    try:
        user_id = st.session_state.get('user_id')
        session_id = get_session_id()
        
        # Process PDFs and create vector store
        text, figures = get_pdf_text(pdf_docs)
        chunks = get_text_chunks(text)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        store = FAISS.from_texts(chunks, embedding=embeddings)
        
        metadata = {
            'created_at': datetime.now().isoformat(),
            'num_chunks': len(chunks),
            'figures': figures
        }
        
        vector_store_manager.save_store(user_id, session_id, store, metadata)
        return store, metadata
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None, None
    
def process_question(vector_store, question, answer_length):
    try:
        k_docs = {"small": 2, "medium": 4, "large": 6}
        docs = vector_store.similarity_search(question, k=k_docs[answer_length])
        chain = get_study_chain(answer_length)
        response = chain(
            {"input_documents": docs, "question": question},
            return_only_outputs=True
        )
        return response["output_text"], [], []
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return f"Error processing question: {str(e)}", [], []

def create_pdf(qa_pairs):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2E5090')
    )
    
    answer_style = ParagraphStyle(
        'AnswerStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=10,
        leading=16
    )
    
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontSize=12,
        leftIndent=20,
        spaceBefore=5,
        spaceAfter=5,
        bulletIndent=10,
        leading=16
    )
    
    # Helper function to process text and create paragraphs
    def format_text(text):
        formatted_content = []
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            # Handle bullet points
            if para.strip().startswith('*') or para.strip().startswith('-'):
                lines = para.split('\n')
                for line in lines:
                    if line.strip():
                        formatted_content.append(
                            Paragraph(
                                line.strip().replace('*', '•').replace('-', '•'),
                                bullet_style
                            )
                        )
            # Handle numbered lists
            elif any(line.strip().startswith(str(i) + '.') for i in range(1, 10) for line in para.split('\n')):
                lines = para.split('\n')
                for line in lines:
                    if line.strip():
                        formatted_content.append(
                            Paragraph(
                                line.strip(),
                                bullet_style
                            )
                        )
            # Regular paragraphs
            else:
                formatted_content.append(Paragraph(para.strip(), answer_style))
                
        return formatted_content
    
    # Build PDF content
    content = []
    content.append(Paragraph("Study Notes", title_style))
    content.append(Spacer(1, 20))
    
    for qa in qa_pairs:
        # Add question
        content.append(Paragraph(f"Question: {qa['question']}", question_style))
        
        # Add answer with formatting
        answer_paragraphs = format_text(qa['answer'])
        content.extend(answer_paragraphs)
        
        # Add references if present
        if qa.get('references') or qa.get('figures'):
            content.append(Spacer(1, 10))
            if qa.get('references'):
                content.append(
                    Paragraph(
                        f"<b>References:</b> {', '.join(f'[{r}]' for r in qa['references'])}",
                        answer_style
                    )
                )
            if qa.get('figures'):
                content.append(
                    Paragraph(
                        f"<b>Figures:</b> {', '.join(f'[{f}]' for f in qa['figures'])}",
                        answer_style
                    )
                )
        
        content.append(Spacer(1, 20))
        content.append(Paragraph("<hr/>", answer_style))
        content.append(Spacer(1, 20))
    
    try:
        doc.build(content)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def delete_qa(index):
    st.session_state.qa_pairs.pop(index)
    st.session_state.asked_questions = {
        f"{qa['question'].strip().lower()}_{qa['answer_length']}": True 
        for qa in st.session_state.qa_pairs
    }

import streamlit as st

import streamlit as st
import os

import streamlit as st
import os

import streamlit as st
import os

def main():
    st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")
    initialize_session_state()
    
    st.title("📚 Study Assistant")
    st.caption("Upload PDFs and ask questions about the content")
    st.caption("please upload module wise content and don't upload whole book")
    
    
    with st.sidebar:
        st.header("Usage Statistics")
        st.metric("Questions Asked", len(st.session_state.qa_pairs))
        if hasattr(st.session_state, 'vector_store'):
            st.success("PDFs Processed ✓")
        else:
            st.warning("No PDFs Processed")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
            help=f"Maximum file size: {MAX_UPLOAD_SIZE_MB}MB per file"
        )
        
        if pdf_docs:
            if st.button("Process PDFs"):
                with st.spinner("Processing PDFs..."):
                    vector_store, metadata = get_vector_store(pdf_docs)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.figures = metadata.get('figures', [])
                        st.session_state.asked_questions = {}
                        st.session_state.qa_pairs = []
                        st.success("PDFs processed successfully!")
        
        question = st.text_area("Your Question:", help="Enter your question about the PDF content")
        answer_length = st.select_slider(
            "Answer Length:",
            options=["small", "medium", "large"],
            value="medium",
            help="Small: Brief overview, Medium: Comprehensive explanation, Large: In-depth analysis"
        )
        
        if st.button("Get Answer"):
            if hasattr(st.session_state, 'vector_store') and question:
                question_key = f"{question.strip().lower()}_{answer_length}"
                
                if question_key not in st.session_state.asked_questions:
                    with st.spinner("Finding answer..."):
                        answer, section_refs, figure_refs = process_question(
                            st.session_state.vector_store,
                            question,
                            answer_length
                        )
                        
                        if not answer.startswith("Error"):
                            qa_entry = {
                                "question": question,
                                "answer": answer,
                                "references": section_refs,
                                "figures": figure_refs,
                                "timestamp": datetime.now().isoformat(),
                                "answer_length": answer_length
                            }
                            
                            st.session_state.qa_pairs.append(qa_entry)
                            st.session_state.asked_questions[question_key] = True
                            
                            st.markdown("### Answer")
                            st.markdown(answer)
                            
                            if figure_refs and st.session_state.figures:
                                st.markdown("### Referenced Figures")
                                for ref in figure_refs:
                                    matching_figs = [f for f in st.session_state.figures if f["id"] == ref]
                                    for fig in matching_figs:
                                        st.info(f"Figure {fig['id']}: {fig['caption']}")
                else:
                    st.warning(f"You already have an answer for this question with {answer_length} length. Try a different length or ask a new question.")
            else:
                st.warning("Please upload and process PDFs first, then enter a question!")

    with col2:
        st.header("Session History")
        
        if st.session_state.qa_pairs:
            if st.button("Export Session to PDF"):
                pdf = create_pdf(st.session_state.qa_pairs)
                if pdf:
                    st.download_button(
                        "Download Session Notes (PDF)",
                        pdf,
                        "study_session_notes.pdf",
                        "application/pdf"
                    )
            
            for i, qa in enumerate(reversed(st.session_state.qa_pairs)):
                with st.expander(f"Q: {qa['question']} ({qa.get('answer_length', 'default').capitalize()})", expanded=True):
                    col1, col2 = st.columns([0.9, 0.1])
                    with col1:
                        st.markdown(qa['answer'])
                        if qa.get('references') or qa.get('figures'):
                            st.divider()
                            if qa['references']:
                                st.markdown("**Sections:** " + ", ".join(f"[{r}]" for r in qa['references']))
                            if qa['figures']:
                                st.markdown("**Figures:** " + ", ".join(f"[{f}]" for f in qa['figures']))
                    with col2:
                        if st.button("🗑️", key=f"delete_{i}"):
                            delete_qa(len(st.session_state.qa_pairs) - 1 - i)
                            st.rerun()
        else:
            st.info("Ask questions to see your history here")

    # Developer Profile Section
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        image_path = "developer_resized.jpg"
        if os.path.exists(image_path):
            st.image(image_path, width=150, caption="Ritik(r.itik__)")
        else:
            st.warning("Developer image not found. Please check the path.")
    
    with col2:
        st.markdown(
            """
            <h3 style="color: #2E5090; margin: 10px 0;">ABHIJEET</h3>
            <p style="color: #666; margin: 5px 0;">AI & ML Developer</p>
            <p style="color: #666; font-size: 0.9em;">Contact: abhku21aiml@cmrit.ac.in<br>Location: Your AECS Layout, India</p>
            """,
            unsafe_allow_html=True
        )
    
    st.caption("© 2025 Study Assistant - All PDFs are processed locally and securely")

if __name__ == '__main__':
    main()


