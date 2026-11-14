import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def ingest():
    print("🚀 Starting Ingestion Pipeline...")
    
    # 1. Load a real PDF (A public HR Policy)
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    print(f"Downloading {url}...")
    
    loader = PyPDFLoader(url)
    docs = loader.load()
    
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} document chunks.")
    
    # 3. Embed and Store in ChromaDB
    print("🧠 Embedding and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("🎉 Ingestion Complete! Vector DB saved to ./chroma_db")

if __name__ == "__main__":
    ingest()