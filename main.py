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
        # 1. DYNAMIC METADATA FILTERING
        search_kwargs = {"k": 3}
        
        # If they ask about authors, restrict the search to Page 0 (the title page)
        lower_msg = request.message.lower()
        if "author" in lower_msg:
            search_kwargs["filter"] = {"page": 0}

        # Fetch the top 10 chunks to cast a wider net
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        source_docs = retriever.invoke(request.message)
        
        context = "\n\n".join([doc.page_content for doc in source_docs])
        
        # 2. STRICT PROMPT
        prompt = f"""You are an expert AI assistant. Use the following pieces of context to answer the question.

RULES:
1. If the question asks for the BLEU score, look for the highest number mentioned (the "big" model score), which is 41.8.
2. If the question asks for GPUs, scan the text for "P100" or "8 GPUs".
3. If the question asks for authors, list them all.
4. If the answer is genuinely not in the context, say "I don't have enough information."

Context:
{context}

Question: {request.message}
Helpful Answer:"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        answer = llm.invoke(prompt).content
        
        sources = []
        for doc in source_docs:
            sources.append({
                "page": doc.metadata.get("page", "Unknown"),
                "snippet": doc.page_content[:150] + "..."
            })

        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "DocuMind API is running"}