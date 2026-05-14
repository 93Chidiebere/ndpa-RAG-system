import os
import json
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, GenerationConfig
from rank_bm25 import BM25Okapi

app = FastAPI(title="NDPA RAG System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models and data
chunks = []
index = None
embedding_model = None
bm25 = None
text_generator = None
generation_config = None

@app.on_event("startup")
def load_models_and_data():
    global chunks, index, embedding_model, bm25, text_generator, generation_config
    
    print("Loading chunks.json...")
    try:
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"Error loading chunks.json: {e}. Make sure to run save_data.py first.")
        chunks = []
        
    print("Loading FAISS index...")
    try:
        index = faiss.read_index("ndpa_faiss.index")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        
    print("Initializing BM25...")
    if chunks:
        tokenized_chunks = [chunk.split(" ") for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        
    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading TinyLlama text generator locally (this might take a minute)...")
    # Setup generation config to avoid memory/timeout issues if possible
    generation_config = GenerationConfig(
        max_new_tokens=200,
        do_sample=False
    )
    
    text_generator = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=-1 # CPU
    )
    print("Startup complete!")

def hybrid_retrieve(query, top_k=5):
    # Dense retrieval
    query_embedding = embedding_model.encode([query])
    query_embedding = query_embedding.astype("float32")
    
    distances, dense_indices = index.search(query_embedding, top_k)
    dense_results = [chunks[idx] for idx in dense_indices[0]]
    
    # BM25 retrieval
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [chunks[idx] for idx in bm25_indices]
    
    # Merged Result
    merged = list(dict.fromkeys(dense_results + bm25_results))
    return merged[:top_k]

def build_prompt(query, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""<|system|>
You are a legal assistant specialized in the Nigerian Data Protection Act 2023. Answer ONLY using the provided context. If the answer is not in the context, say: 'I could not find the answer in the provided document.'</s>
<|user|>
Context:
{context_text}

Question:
{query}</s>
<|assistant|>
"""
    return prompt

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    if not chunks or index is None or text_generator is None:
        return {"error": "System is not fully initialized. Check server logs."}
        
    query = request.query
    contexts = hybrid_retrieve(query)
    prompt = build_prompt(query, contexts)
    
    response = text_generator(
        prompt,
        generation_config=generation_config,
        clean_up_tokenization_spaces=False
    )
    
    generated_text = response[0]["generated_text"]
    
    # Extract only the assistant's response part
    answer = generated_text.split("<|assistant|>\n")[-1].strip()
    
    return {
        "query": query,
        "answer": answer,
        "sources": contexts
    }

# HTML UI
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NDPA RAG System</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f9fafb; color: #111827; }
        h1 { color: #2563eb; text-align: center; }
        .container { background-color: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .chat-box { height: 400px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 20px; display: flex; flex-direction: column; gap: 15px; background-color: #f3f4f6; }
        .message { padding: 12px 16px; border-radius: 8px; max-width: 80%; line-height: 1.5; }
        .user-message { background-color: #2563eb; color: white; align-self: flex-end; border-bottom-right-radius: 0; }
        .bot-message { background-color: white; color: #1f2937; align-self: flex-start; border-bottom-left-radius: 0; border: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
        .input-group { display: flex; gap: 10px; }
        input[type="text"] { flex: 1; padding: 12px; border: 1px solid #d1d5db; border-radius: 8px; outline: none; font-size: 16px; }
        input[type="text"]:focus { border-color: #2563eb; }
        button { padding: 12px 24px; background-color: #2563eb; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: 500; transition: background-color 0.2s; }
        button:hover { background-color: #1d4ed8; }
        button:disabled { background-color: #93c5fd; cursor: not-allowed; }
        .loading { font-size: 14px; color: #6b7280; text-align: center; display: none; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>NDPA 2023 Legal Assistant</h1>
    <div class="container">
        <p style="text-align: center; color: #4b5563; margin-bottom: 20px;">Ask any question about the Nigerian Data Protection Act 2023</p>
        <div class="chat-box" id="chatBox">
            <div class="message bot-message">Hello! I am an AI legal assistant trained on the Nigerian Data Protection Act (NDPA) 2023. What would you like to know?</div>
        </div>
        <div class="input-group">
            <input type="text" id="queryInput" placeholder="E.g., What are the rights of a data subject?" onkeypress="handleKeyPress(event)">
            <button id="sendBtn" onclick="askQuestion()">Ask</button>
        </div>
        <div class="loading" id="loadingIndicator">Generating answer... this might take a moment. (Using local TinyLlama, please be patient)</div>
    </div>

    <script>
        async function askQuestion() {
            const queryInput = document.getElementById('queryInput');
            const chatBox = document.getElementById('chatBox');
            const sendBtn = document.getElementById('sendBtn');
            const loadingIndicator = document.getElementById('loadingIndicator');
            
            const query = queryInput.value.trim();
            if (!query) return;

            // Add user message
            appendMessage(query, 'user-message');
            queryInput.value = '';
            
            // Disable input and show loading
            queryInput.disabled = true;
            sendBtn.disabled = true;
            loadingIndicator.style.display = 'block';
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    appendMessage("Error: " + data.error, 'bot-message');
                } else {
                    appendMessage(data.answer, 'bot-message');
                }
            } catch (error) {
                appendMessage("Error connecting to the server.", 'bot-message');
            } finally {
                // Enable input and hide loading
                queryInput.disabled = false;
                sendBtn.disabled = false;
                loadingIndicator.style.display = 'none';
                queryInput.focus();
            }
        }
        
        function appendMessage(text, className) {
            const chatBox = document.getElementById('chatBox');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${className}`;
            msgDiv.textContent = text;
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                askQuestion();
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTML_CONTENT