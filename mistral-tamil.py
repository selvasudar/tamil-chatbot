import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber  # Better for Tamil text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os
import dotenv
import re
import shutil
import hashlib
import tempfile
import chromadb
from chromadb.config import Settings

# Load environment variables from .env file
dotenv.load_dotenv()

# Function to create a unique collection name based on file content
def get_file_hash(uploaded_file):
    """Generate a hash of the uploaded file for unique collection naming"""
    uploaded_file.seek(0)
    file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return f"pdf_collection_{file_hash[:8]}"

# Function to extract text from PDF with multiple methods
def extract_text_from_pdf(uploaded_file):
    text = ""
    
    # Try pdfplumber first (better for complex layouts)
    try:
        import pdfplumber
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            st.success(f"✅ Text extracted using pdfplumber: {len(text)} characters")
            return text
    except Exception as e:
        st.warning(f"pdfplumber failed: {e}")
    
    # Fallback to PyPDF2
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if text.strip():
            st.success(f"✅ Text extracted using PyPDF2: {len(text)} characters")
            return text
    except Exception as e:
        st.error(f"PyPDF2 also failed: {e}")
    
    return text

# Function to check if text contains Tamil characters
def contains_tamil(text):
    tamil_range = re.compile(r'[\u0B80-\u0BFF]')
    return bool(tamil_range.search(text))

# Function to initialize embedding model with fallback options
@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings with multiple fallback options"""
    embedding_options = [
        {
            "name": "mistral-embed",
            "description": "Mistral AI Embeddings (Multilingual, 1024-dim)",
            "type": "mistral",
            "fallback": False
        },
        {
            "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "description": "Multilingual (Best for Tamil)",
            "fallback": False
        },
        {
            "name": "sentence-transformers/all-MiniLM-L6-v2", 
            "description": "English (Fallback 1)",
            "fallback": True
        },
        {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "description": "English (Fallback 2)", 
            "fallback": True
        }
    ]
    
    for option in embedding_options:
        try:
            st.info(f"🔄 Trying to load: {option['description']}")
            if option["type"] == "mistral":
                from mistralai import Mistral
                client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
                
                # Create a custom embedding class compatible with langchain
                class MistralEmbeddings:
                    def __init__(self, client, model_name):
                        self.client = client
                        self.model_name = model_name
                    
                    def embed_documents(self, texts):
                        try:
                            response = self.client.embeddings.create(
                                model=self.model_name,
                                inputs=texts
                            )
                            return [data.embedding for data in response.data]
                        except Exception as e:
                            st.error(f"❌ Mistral API error: {str(e)}")
                            raise
                    
                    def embed_query(self, text):
                        try:
                            response = self.client.embeddings.create(
                                model=self.model_name,
                                inputs=[text]
                            )
                            return response.data[0].embedding
                        except Exception as e:
                            st.error(f"❌ Mistral API error: {str(e)}")
                            raise
                
                embeddings = MistralEmbeddings(client, option["name"])
                # Test the embedding
                test_embedding = embeddings.embed_query("test")
                st.success(f"✅ Successfully loaded: {option['description']}")
                return embeddings, option["name"]
            
            elif option["type"] == "huggingface":
                embeddings = HuggingFaceEmbeddings(
                    model_name=option["name"],
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                # Test the embedding
                test_embedding = embeddings.embed_query("test")
                st.success(f"✅ Successfully loaded: {option['description']}")
                return embeddings, option["name"]
                
        except Exception as e:
            st.warning(f"❌ Failed to load {option['name']}: {str(e)}")
            continue
    
    # If all HuggingFace models fail, try OpenAI embeddings as last resort
    try:
        st.info("🔄 Trying OpenAI embeddings as final fallback...")
        openai_api_key = st.sidebar.text_input("OpenAI API Key (Fallback)", type="password")
        if openai_api_key:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            test_embedding = embeddings.embed_query("test")
            st.success("✅ Successfully loaded OpenAI embeddings")
            return embeddings, "OpenAI"
        else:
            st.error("❌ OpenAI API key required for fallback embeddings")
    except ImportError:
        st.warning("OpenAI package not installed. Install with: pip install langchain-openai")
    except Exception as e:
        st.error(f"OpenAI embeddings failed: {str(e)}")
    
    # If everything fails, provide instructions
    st.error("❌ All embedding options failed!")
    st.markdown("""
    ### 🔧 Troubleshooting Steps:
    
    1. **Check Internet Connection**: Ensure you can access huggingface.co
    2. **Use VPN**: If behind firewall, try using a VPN
    3. **Download Model Manually**: 
       ```bash
       python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
       ```
    4. **Use Local Model**: Place model files in local directory
    5. **Alternative**: Use OpenAI embeddings (requires API key)
    """)
    return None, None

# Function to initialize ChromaDB client
@st.cache_resource
def initialize_chroma_client():
    """Initialize ChromaDB client with persistent storage"""
    # Create a persistent directory for ChromaDB
    persist_directory = os.path.join(tempfile.gettempdir(), "tamil_chatbot_chroma")
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=persist_directory)
    return client, persist_directory

# Function to create or get existing collection
def get_or_create_collection(client, collection_name, embeddings):
    """Get existing collection or create new one"""
    try:
        # Try to get existing collection
        collection = client.get_collection(name=collection_name)
        st.info(f"📚 Found existing collection: {collection_name}")
        
        # Create Langchain Chroma wrapper
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        return vectorstore, True
        
    except Exception:
        # Collection doesn't exist, will create new one
        st.info(f"🆕 Creating new collection: {collection_name}")
        return None, False

# Streamlit app
st.title("Tamil Language Chatbot with ChromaDB 📚")
st.markdown("*Persistent vector storage with ChromaDB for better performance*")

# Sidebar for API keys
st.sidebar.header("API Keys")
gemini_api_key = st.sidebar.text_input("Google API Key", type="password", 
                                      value=os.getenv("GOOGLE_API_KEY", ""))
hf_api_token = st.sidebar.text_input("HuggingFace API Token", type="password",
                                    value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))
mistral_api_key = st.sidebar.text_input("Mistral AI API Key", type="password",
                                       value=os.getenv("MISTRAL_API_KEY", ""))

if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
if hf_api_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token
if mistral_api_key:
    os.environ["MISTRAL_API_KEY"] = mistral_api_key

# Check if API keys are provided
if not gemini_api_key or not hf_api_token or not mistral_api_key:
    st.error("Please provide both API keys in the sidebar!")
    st.stop()

# Initialize ChromaDB client
try:
    chroma_client, persist_directory = initialize_chroma_client()
    st.sidebar.success(f"🔗 ChromaDB initialized")
    st.sidebar.info(f"📁 Storage: {persist_directory}")
except Exception as e:
    st.error(f"❌ Failed to initialize ChromaDB: {e}")
    st.stop()

# Sidebar: Collection Management
st.sidebar.header("📊 Collection Management")
if st.sidebar.button("🗑️ Clear All Collections"):
    try:
        # Delete all collections
        collections = chroma_client.list_collections()
        for collection in collections:
            chroma_client.delete_collection(collection.name)
        st.sidebar.success("All collections cleared!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error clearing collections: {e}")

# Show existing collections
try:
    existing_collections = chroma_client.list_collections()
    if existing_collections:
        st.sidebar.write("📚 Existing Collections:")
        for collection in existing_collections:
            st.sidebar.write(f"- {collection.name}")
except Exception as e:
    st.sidebar.warning(f"Could not list collections: {e}")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF (தமிழில்)", type="pdf")

if uploaded_file is not None:
    # Generate unique collection name based on file content
    collection_name = get_file_hash(uploaded_file)
    
    # Initialize embeddings with fallback options
    if "embeddings" not in st.session_state:
        with st.spinner("🔄 Loading embeddings model..."):
            embeddings, model_name = initialize_embeddings()
            if embeddings is None:
                st.error("❌ Could not load any embedding model. Please check your internet connection or try the solutions above.")
                st.stop()
            st.session_state.embeddings = embeddings
            st.session_state.embedding_model = model_name
    
    # Check if we already have this PDF processed
    vectorstore, collection_exists = get_or_create_collection(
        chroma_client, 
        collection_name, 
        st.session_state.embeddings
    )
    
    if not collection_exists:
        with st.spinner("Processing PDF (PDF செயலாக்கம்)..."):
            try:
                # Extract text
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                # Debug: Show extracted text info
                if not pdf_text.strip():
                    st.error("❌ No text could be extracted from the PDF. The PDF might contain only images.")
                    st.info("💡 Try using an OCR tool to convert scanned PDFs to text first.")
                    st.stop()
                
                # Check for Tamil content
                has_tamil = contains_tamil(pdf_text)
                st.info(f"Tamil characters detected: {'✅ Yes' if has_tamil else '❌ No'}")
                
                # Show a preview of extracted text
                with st.expander("🔍 Preview of Extracted Text (First 500 characters)"):
                    st.text(pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text)
                
                # Split text with better separators for Tamil
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=[
                        "\n\n",  # Double newlines
                        "\n",    # Single newlines  
                        "।",     # Devanagari full stop
                        "।।",    # Double devanagari
                        ".",     # English full stop
                        "!",     # Exclamation
                        "?",     # Question mark
                        ";",     # Semicolon
                        ",",     # Comma
                        " ",     # Space
                        ""       # Character level
                    ]
                )
                chunks = text_splitter.split_text(pdf_text)
                
                st.info(f"📄 Text split into {len(chunks)} chunks")
                
                # Show chunk preview
                with st.expander(f"📋 Preview of Chunks (First 3 of {len(chunks)})"):
                    for i, chunk in enumerate(chunks[:3]):
                        st.text(f"Chunk {i+1} ({len(chunk)} chars): {chunk[:200]}...")
                
                # Create ChromaDB vector store
                st.info("📄 Creating embeddings and storing in ChromaDB...")
                
                vectorstore = Chroma.from_texts(
                    texts=chunks,
                    embedding=st.session_state.embeddings,
                    client=chroma_client,
                    collection_name=collection_name,
                    metadatas=[{"chunk_id": i, "source": uploaded_file.name} for i in range(len(chunks))]
                )
                
                st.session_state.chunks = chunks  # Store for debugging
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.exception(e)
                st.stop()
    else:
        st.info("♻️ Using existing processed PDF from ChromaDB")
    
    # Store vectorstore in session state
    st.session_state.vectorstore = vectorstore
    st.success(f"✅ PDF successfully processed using {st.session_state.embedding_model}! (PDF வெற்றிகரமாக செயலாக்கப்பட்டது!)")

    # Display collection statistics
    try:
        collection_info = chroma_client.get_collection(collection_name)
        doc_count = collection_info.count()
        embedding_dim = 1024 if st.session_state.embedding_model == "mistral-embed" else 384  # Adjust based on model
        st.info(f"📊 Collection '{collection_name}' contains {doc_count} document chunks")
    except Exception as e:
        st.warning(f"Could not get collection info: {e}")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("PDF பற்றி கேள்வி கேளுங்கள் (Ask a question about the PDF)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("சிந்திக்கிறது (Thinking...)"):
                try:
                    # LLM using Gemini
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        temperature=0.1,  # Lower temperature for more focused answers
                        max_tokens=1000
                    )
                    
                    # Enhanced bilingual prompt template
                    prompt_template = """You are a helpful assistant that can read and understand both Tamil and English text. 
                    Answer the question based on the provided context from the PDF document.
                    
                    Important instructions:
                    - If the context is in Tamil, you can answer in Tamil or English as appropriate
                    - If the context is in English, you can answer in English or Tamil as requested
                    - If you don't know the answer from the context, say so clearly
                    - Be specific and cite relevant parts of the context
                    - Keep your answer concise but complete
                    
                    Context from PDF:
                    {context}
                    
                    Question: {question}
                    
                    Answer:"""
                    
                    PROMPT = PromptTemplate(
                        template=prompt_template, 
                        input_variables=["context", "question"]
                    )
                    
                    # Retrieval QA chain with more retrieved documents
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 5}  # Retrieve more chunks
                        ),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": PROMPT}
                    )
                    
                    # Get response
                    result = qa_chain({"query": prompt})
                    response = result["result"]
                    source_docs = result.get("source_documents", [])
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show source information in expander
                    if source_docs:
                        with st.expander(f"📚 Source Information ({len(source_docs)} chunks used)"):
                            for i, doc in enumerate(source_docs):
                                st.text(f"Source {i+1}: {doc.page_content[:200]}...")
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    response = "Sorry, I encountered an error while processing your question. Please try again."
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Debug section
    if st.sidebar.checkbox("🔧 Debug Mode"):
        st.sidebar.subheader("Debug Information")
        if st.sidebar.button("Test Tamil Embedding"):
            tamil_text = "இந்த ஆவணம் தமிழ் மொழியில் உள்ளது."
            try:
                embedding = st.session_state.embeddings.embed_query(tamil_text)
                st.sidebar.success(f"✅ Tamil text embedded successfully! Embedding length: {len(embedding)}")
                st.sidebar.text(f"First 5 values: {embedding[:5]}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to embed Tamil text: {e}")

else:
    st.info("📄 தயவு செய்து ஒரு PDF ஐ அப்லோட் செய்து இரண்டு API விசைகளையும் உள்ளிடவும் (Please upload a PDF and enter both API keys to start).")

# Installation requirements
st.sidebar.markdown("""
### 📦 Required Packages:
```bash
pip install streamlit PyPDF2 pdfplumber langchain langchain-google-genai langchain-huggingface chromadb sentence-transformers python-dotenv mistralai

# Optional: For OpenAI fallback
pip install langchain-openai
```

### 🔧 Troubleshooting Connection Issues:
1. **Check internet connection**
2. **Try VPN if behind firewall**
3. **Pre-download models**:
   ```python
   from sentence_transformers import SentenceTransformer
   SentenceTransformer('all-MiniLM-L6-v2')
   ```
4. **Use OpenAI embeddings** (enter API key above)

### 🆕 ChromaDB Features:
- ✅ Persistent storage
- ✅ Better scalability  
- ✅ Metadata support
- ✅ Collection management
- ✅ Automatic deduplication
- ✅ Multiple embedding fallbacks
""")

# Footer
st.markdown("---")
st.markdown("*Powered by ChromaDB for persistent vector storage* 🚀")