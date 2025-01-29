# Study Assistant 📚

A Streamlit-based application that helps users study and analyze PDF documents using AI-powered question-answering capabilities. The app processes PDF content and allows users to ask questions about the material, generating comprehensive answers with references to specific sections and figures.
![image](https://github.com/user-attachments/assets/2a06e990-4a3d-4743-af04-68b736141e10)

## Features

### 📄 PDF Processing & Analysis
- Upload and process multiple PDF documents
- Extract text and figure references
- Maximum file size: 50MB per file
- Secure local processing of documents

### 🤖 AI-Powered Question Answering
- Contextual answers based on PDF content
- Three response length options: small, medium, large
- Figure and section references in answers
- Duplicate question detection

### 📊 Session Management
- Persistent question-answer history
- Export session to formatted PDF
- Delete individual Q&A entries
- Usage statistics tracking

### 🔒 Security & Performance
- Rate limiting protection
- Concurrent user support
- Automatic cleanup of old sessions
- Secure vector store management

---

## Prerequisites

Ensure you have the following dependencies installed:

```bash
# Core dependencies
streamlit
PyPDF2
reportlab
langchain
langchain-google-genai
faiss-cpu
python-dotenv
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/study-assistant.git
   cd study-assistant
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   Create a `.env` file in the project root with:
   ```
   GEMINI_API_KEY=your_google_api_key
   VECTOR_STORE_PATH=path_to_store_vectors
   ```

---

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open `http://localhost:8501` in your web browser.
3. Upload PDF documents and click **"Process PDFs"**.
4. Enter questions about the content and select desired answer length.
5. Review answers and export session to PDF if needed.

---

## Configuration

Key configuration variables are defined at the top of the script:
```python
MAX_REQUESTS_PER_MINUTE = 1000
MAX_UPLOAD_SIZE_MB = 50
CACHE_TTL = 3600  # 1 hour
MAX_CACHED_ENTRIES = 10
```

---

## Architecture

The application consists of several key components:
- **UserVectorStore**: Manages vector stores for multiple users with concurrent access support.
- **get_study_chain**: Configures the question-answering chain with different response lengths.
- **get_pdf_text**: Extracts text and figures from PDF documents.
- **get_vector_store**: Creates and manages FAISS vector stores for document embeddings.
- **process_question**: Handles question processing and answer generation.
- **create_pdf**: Generates formatted PDF exports of Q&A sessions.

---

## Security Considerations
- All PDFs are processed locally.
- No data is stored permanently.
- Automatic cleanup of old sessions.
- Rate limiting protection.
- Concurrent access management.
- File size restrictions.

---

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to the branch.
5. Create a Pull Request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Abhijeet**  
Email: [abhku21aiml@cmrit.ac.in](mailto:abhku21aiml@cmrit.ac.in)  
Location: AECS Layout, India  

---

## Acknowledgments
- Built with **Streamlit**
- Powered by **Google's Generative AI**
- Uses **FAISS** for vector storage
- Utilizes **Langchain** for document processing

© 2025 Study Assistant - All PDFs are processed locally and securely.

