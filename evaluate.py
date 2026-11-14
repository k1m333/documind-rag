import json
import requests
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# We use an LLM as the judge
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def evaluate():
    print("🚀 Starting RAG Evaluation Pipeline...\n")
    
    with open("golden_dataset.json", "r") as f:
        dataset = json.load(f)
        
    passed = 0
    total = len(dataset)
    
    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        # 1. Ask our API
        response = requests.post(
            "http://localhost:8000/api/chat", 
            json={"message": question}
        )
        api_data = response.json()
        generated_answer = api_data.get("answer", "")
        
        # 2. Ask the Judge LLM to grade it
        judge_prompt = f"""
        You are an impartial judge evaluating an AI system.
        Question: {question}
        Ground Truth Answer: {ground_truth}
        AI Generated Answer: {generated_answer}
        
        Does the AI's answer contain the core information from the ground truth? 
        If the question asks for a list (like authors or GPUs), accept the answer if it contains most of the items. It does not need to be perfect.
        Reply with ONLY the word "PASS" or "FAIL".
        """
        
        verdict = judge_llm.invoke(judge_prompt).content.strip().upper()
        
        if "PASS" in verdict:
            passed += 1
            print(f"✅ Q{i+1}: PASS")
            print(f"   Q: {question}")
            print(f"   AI: {generated_answer}\n")
        else:
            print(f"❌ Q{i+1}: FAIL")
            print(f"   Q: {question}")
            print(f"   Expected: {ground_truth}")
            print(f"   AI: {generated_answer}\n")

    # 3. Calculate Final Score
    score = (passed / total) * 100
    print("====================================")
    print(f"📊 RAG EVALUATION COMPLETE")
    print(f"Accuracy Score: {score:.1f}% ({passed}/{total} passed)")
    print("====================================")

if __name__ == "__main__":
    evaluate()