import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber  # Better for Tamil text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os
import dotenv
import re

# Load environment variables from .env file
dotenv.load_dotenv()

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
            st.success(f"✓ Text extracted using pdfplumber: {len(text)} characters")
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
            st.success(f"✓ Text extracted using PyPDF2: {len(text)} characters")
            return text
    except Exception as e:
        st.error(f"PyPDF2 also failed: {e}")
    
    return text

# Function to check if text contains Tamil characters
def contains_tamil(text):
    tamil_range = re.compile(r'[\u0B80-\u0BFF]')
    return bool(tamil_range.search(text))

# Streamlit app
st.title("Tamil Language Chatbot (Tamil Support) 📚")

# Sidebar for API keys
st.sidebar.header("API Keys")
gemini_api_key = st.sidebar.text_input("Google API Key", type="password", 
                                      value=os.getenv("GOOGLE_API_KEY", ""))
hf_api_token = st.sidebar.text_input("HuggingFace API Token", type="password",
                                    value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))

if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
if hf_api_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token

# Check if API keys are provided
if not gemini_api_key or not hf_api_token:
    st.error("Please provide both API keys in the sidebar!")
    st.stop()

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF (தமிழில்)", type="pdf")

if uploaded_file is not None:
    if "vectorstore" not in st.session_state:
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
                st.info(f"Tamil characters detected: {'✓ Yes' if has_tamil else '❌ No'}")
                
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
                
                # Embeddings using multilingual model
                st.info("🔄 Creating embeddings...")
                
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # Vector store
                vectorstore = FAISS.from_texts(chunks, embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.chunks = chunks  # Store for debugging
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.exception(e)
                st.stop()
        
        st.success("✅ PDF successfully processed! (PDF வெற்றிகரமாக செயலாக்கப்பட்டது!)")

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
    if st.session_state.get("vectorstore") and st.sidebar.checkbox("🔧 Debug Mode"):
        st.sidebar.subheader("Debug Information")
        if st.sidebar.button("Test Retrieval"):
            test_query = "test"
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(test_query)
            st.sidebar.write(f"Retrieved {len(docs)} documents")
            for i, doc in enumerate(docs):
                st.sidebar.text(f"Doc {i+1}: {doc.page_content[:100]}...")

else:
    st.info("📄 தயவு செய்து ஒரு PDF ஐ அப்லோட் செய்து இரண்டு API விசைகளையும் உள்ளிடவும் (Please upload a PDF and enter both API keys to start).")

# Installation requirements
st.sidebar.markdown("""
### 📦 Required Packages:
```bash
pip install streamlit PyPDF2 pdfplumber langchain langchain-google-genai langchain-huggingface faiss-cpu sentence-transformers python-dotenv
```
""")