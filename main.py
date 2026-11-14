from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DocuMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Connect to the local ChromaDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
try:
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
except Exception as e:
    print(f"⚠️ Could not load ChromaDB. Error: {e}")
    vectorstore = None

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vector database not initialized.")

    try:
        # 2. Retrieve the top 3 most relevant document chunks
        retriever = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
        )
        source_docs = retriever.invoke(request.message)
        
        # 3. Format the retrieved chunks into a single context string
        context = "\n\n".join([doc.page_content for doc in source_docs])
        
        # 4. Build the prompt manually
                # Change the prompt to this:
        prompt = f"""You are an expert AI assistant reading technical documents. 
Use the following pieces of context to answer the question. Note that tables in PDFs sometimes have disconnected text (e.g., numbers separated from their labels).

If you find the answer in the context, provide it clearly. If you truly cannot find the answer, say "I don't have enough information to answer that based on the provided documents."

Context:
{context}

Question: {request.message}
Helpful Answer:"""
        
        # 5. Call the LLM directly
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        answer = llm.invoke(prompt).content
        
        # 6. Format citations
        sources = []
        for doc in source_docs:
            sources.append({
                "page": doc.metadata.get("page", "Unknown"),
                "snippet": doc.page_content[:150] + "..."
            })

        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "DocuMind API is running"}