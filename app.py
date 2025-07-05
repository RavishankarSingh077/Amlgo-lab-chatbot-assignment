import os
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Modular pipeline import
from src.pipeline import build_pipeline

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Streamlit config
st.set_page_config(page_title="RAG Chatbot - Amlgo Labs", layout="wide")
st.title("Amlgo Labs RAG Chatbot (PDF Embedded)")

# Sidebar setup
with st.sidebar:
    st.header("Model & Vector Info")
    st.markdown("- LLM: `phi` via Ollama")
    st.markdown("- Embedding Model: `all-MiniLM-L6-v2`")
    st.markdown("- Vector DB: `Chroma`")
    if st.button(" Reset / Clear"):
        st.session_state.query = ""
        st.session_state.answer = ""
        st.rerun()

# Cleaning function
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page\s+\d+", "", text)
    return text.strip()

# Step 1: Load PDF & prepare chunks (only once using session)
if "vectorstore" not in st.session_state:
    pdf_path = "data/AI Training Document.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Clean document text
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Save raw chunks to /chunks/chunks.json
    chunk_texts = [chunk.page_content for chunk in chunks]
    os.makedirs("chunks", exist_ok=True)
    with open("chunks/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunk_texts, f, ensure_ascii=False, indent=2)

    # Embeddings + Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

    # Save in session
    st.session_state.vectorstore = vectorstore
    st.session_state.chunks = chunks

# Step 2: Load retriever and QA chain from /src modules
retriever, qa_chain = build_pipeline(st.session_state.vectorstore)

# Step 3: Input + Answer
if "query" not in st.session_state:
    st.session_state.query = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

query = st.text_input("Ask a question from the document", value=st.session_state.query)

if query:
    st.session_state.query = query
    with st.spinner("answering..."):
        docs = retriever.get_relevant_documents(query)
        result = qa_chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        st.session_state.answer = result["output_text"]
        st.session_state.sources = docs

# Output Answer
if st.session_state.answer:
    st.subheader("Answer:")
    st.write(st.session_state.answer)

# Output Sources
if "sources" in st.session_state:
    st.subheader("Source Snippets:")
    for i, doc in enumerate(st.session_state.sources):
        st.markdown(f"**Source {i+1}:**")
        st.code(doc.page_content[:500])

# Chunk Preview
with st.expander("Preview Chunks"):
    st.write(f"Total Chunks: {len(st.session_state.chunks)}")
    st.code(st.session_state.chunks[0].page_content[:500])
