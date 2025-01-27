# medical_assistant.py
import streamlit as st
import os
import PyPDF2
import requests
import tempfile
import pinecone
import logging
from bs4 import BeautifulSoup
from groq import Groq
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Suppress warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load environment variables
load_dotenv()

# ==================== Backend Components ====================
class DocumentProcessor:
    @staticmethod
    def process_pdf(file):
        """Process PDF file and return text"""
        try:
            text = ""
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                if page_text := page.extract_text():
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"PDF processing error: {str(e)}")
            return None

    @staticmethod
    def process_url(url):
        """Process web URL content"""
        try:
            if url.lower().endswith('.pdf'):
                response = requests.get(url, timeout=10)
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    return DocumentProcessor.process_pdf(tmp_file.name)

            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
                
            main_content = soup.find(['article', 'main']) or soup.body
            return main_content.get_text(separator='\n', strip=True)
        except Exception as e:
            st.error(f"URL processing error: {str(e)}")
            return None

class AIAssistant:
    def __init__(self):
        self.embeddings = None
        self.groq_client = None
        self.pinecone_index = None
        self.initialize_components()

    def initialize_components(self):
        """Initialize AI components"""
        try:
            # Validate environment variables
            if not os.getenv("GROQ_API_KEY") or not os.getenv("PINECONE_API_KEY"):
                raise ValueError("Missing required API keys in environment variables")

            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize Groq client
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            
            # Initialize Pinecone
            pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.pinecone_index = "pine"
            
            # Create index if not exists
            if self.pinecone_index not in pc.list_indexes().names():
                pc.create_index(
                    name=self.pinecone_index,
                    dimension=384,
                    metric="cosine",
                    spec=pinecone.ServerlessSpec(
                        cloud="aws",
                        region="us-west-2"
                    )
                )
                
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.stop()

    def create_vector_store(self, text):
        """Create Pinecone vector store from text"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(text)
            
            return PineconeVectorStore.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                index_name=self.pinecone_index,
                namespace="medical-docs"
            )
        except Exception as e:
            st.error(f"Vectorization error: {str(e)}")
            return None

    def get_answer(self, question, vector_store):
        """Get answer from AI model"""
        try:
            docs = vector_store.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""Context: {context}
        
        Question: {question}
        
        Answer based on the context. If unsure, say 'I don't know'."""
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.2,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Answer generation error: {str(e)}")
            return None

# ==================== UI Components ====================
class MedicalUI:
    def __init__(self):
        self.assistant = AIAssistant()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []

    def show_sidebar(self):
        """Sidebar controls for document input"""
        with st.sidebar:
            st.header("🗂️ Data Source")
            input_method = st.radio(
                "Select input method:",
                ["Default Dataset", "Upload PDF", "Web URL"],
                index=0
            )
            
            process_clicked = False
            uploaded_file = None
            url = None
            
            if input_method == "Upload PDF":
                uploaded_file = st.file_uploader("Choose medical PDF", type=["pdf"])
                if uploaded_file:
                    process_clicked = st.button("Process Document")
                    
            elif input_method == "Web URL":
                url = st.text_input("Enter document URL:")
                if url:
                    process_clicked = st.button("Process URL")
                    
            else:  # Default dataset
                process_clicked = st.button("Load Default Dataset")
            
            return process_clicked, input_method, uploaded_file, url

    def process_document(self, input_method, uploaded_file=None, url=None):
        """Handle document processing"""
        with st.spinner("Processing document..."):
            try:
                text = ""
                if input_method == "Upload PDF":
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        text = DocumentProcessor.process_pdf(tmp_file.name)
                elif input_method == "Web URL":
                    text = DocumentProcessor.process_url(url)
                else:
                    try:
                        with open(r"C:\1\Infosys Homework\model\medical_book.pdf", "rb") as f:
                            text = DocumentProcessor.process_pdf(f)
                    except FileNotFoundError:
                        st.error("Default document not found")
                        text = None
                
                if text:
                    st.session_state.vector_store = self.assistant.create_vector_store(text)
                    st.success("Document processed successfully!")
                    
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

    def show_main_interface(self):
        """Main query interface"""
        st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💊 Medical Query Assistant 💊</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        question = st.text_area("Enter your question:", height=150, placeholder="Type your medical question here...")
        
        if st.button("Get Answer", use_container_width=True):
            self.handle_query(question)
            
        self.show_query_history()

    def handle_query(self, question):
        """Handle user query"""
        if not question.strip():
            st.warning("Please enter a valid question")
            return
            
        if not st.session_state.vector_store:
            st.warning("Please process a document first")
            return
            
        try:
            with st.spinner("Analyzing..."):
                answer = self.assistant.get_answer(question, st.session_state.vector_store)
                
                # Store in history
                st.session_state.query_history.insert(0, {
                    "question": question,
                    "answer": answer
                })
                
                # Display results
                st.markdown("### 🤖 AI Response")
                st.markdown(f'<div class="response-box">{answer}</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")

    def show_query_history(self):
        """Display query history"""
        if st.session_state.query_history:
            st.markdown("---")
            st.markdown("### 📜 Question History")
            
            for idx, entry in enumerate(st.session_state.query_history[:5]):
                with st.expander(f"Q{idx+1}: {entry['question'][:50]}..."):
                    st.markdown(f"**Question:** {entry['question']}")
                    st.markdown(f"**Answer:** {entry['answer']}")

    def show_footer(self):
        """Application footer"""
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; margin-top: 2rem;'>"
            "🔒 Secure processing | 🚀 Powered by Groq & Pinecone | 📚 Medical AI Assistant v1.0"
            "</div>",
            unsafe_allow_html=True
        )

# ==================== Main Application ====================
def main():
    # Configure page
    st.set_page_config(
        page_title="Medical Query Assistant",
        layout="centered",
        page_icon="💊",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
            .main { padding: 2rem; }
            .stButton>button { background-color: #4CAF50; color: white; }
            .stTextInput>div>div>input { padding: 10px; }
            .sidebar .sidebar-content { background-color: #f0f2f6; }
            .response-box { 
                border-left: 4px solid #4CAF50;
                padding: 1rem;
                margin: 1rem 0;
                background-color: #f8f9fa;
            }
        </style>
    """, unsafe_allow_html=True)

    try:
        ui = MedicalUI()
        process_clicked, input_method, uploaded_file, url = ui.show_sidebar()
        
        if process_clicked:
            ui.process_document(input_method, uploaded_file, url)
            
        ui.show_main_interface()
        ui.show_footer()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()