import os
import time
import uuid
import asyncio
# from datetime import datetime
import datetime 
from typing import Optional

import httpx
import aiosqlite
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel, Field 
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

DATABASE_PATH = os.getenv("DATABASE_PATH", "./ai_requests.db") 
LLM_API_KEY = os.getenv("LLM_API_KEY","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") # set this in your env
# LLM_API_URL = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions") 
LLM_API_URL = os.getenv("LLM_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions") 
# LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini") 
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct") 
# REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))  # seconds 
REQUEST_TIMEOUT = 3000.0

app = FastAPI(title="Reusable AI Backend Meta Version", version="1.0") 

# --- Pydantic models ---
class ChatRequest(BaseModel):
    # user_id: str = Field(..., description="Unique identifier for the user")
    # enforce a minimum length and maximum length for prmopt text to avoid empty or excessively long inputs
    prompt: str = Field(..., min_length=1, max_length=8000, description="User input prompt text")   
    # define an optional system prompt feild with a default value of None and a maximum length of 2000 characters to provide context or instructions to the LLM
    system_prompt: Optional[str] = Field(None, max_length=1024) 
    # max_tokens: Optional[int] = Field(150, description="Maximum tokens for the response")
    temperature: float = Field(0.2, ge=0.0, le=2.0) 

class ChatResponse(BaseModel):
    request_id: str 
    response: str 
    model: str 
    latency_ms: int 
    timestamp: str 

# --- DB setup --- 
async def init_db():
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                prompt TEXT NOT NULL,
                system_prompt TEXT, 
                model TEXT NOT NULL,               
                response TEXT,                
                status TEXT NOT NULL,
                latency_ms INTEGER,
                error TEXT 
            )
        """)
        await db.commit()  

@app.on_event("startup")
async def startup_event():
    await init_db()
    if not LLM_API_KEY: 
        print("WARNING: LLM_API_KEY environment variable is not set. /chat will fail until you set it.")

# --- Core Logic --- 
async def call_llm(prompt: str, system_prompt: Optional[str], temperature: float) -> dict:
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt}) 
    messages.append({"role": "user", "content": prompt}) 
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages, 
        "temperature": temperature
    }   

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(LLM_API_URL, json=payload, headers=headers) 
        response.raise_for_status() 
        data = response.json() 

        content = data["choices"][0]["message"]["content"]    
        return {"response": content, "model": data.get("model", LLM_MODEL)} 


async def log_request(req_id: str, prompt: str, system_prompt: Optional[str], 
                            model: str, response: Optional[str], status: str, latency_ms: int, error: Optional[str]):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT INTO requests (id, timestamp, prompt, system_prompt, model, response, status, latency_ms, error)
            VALUES (?,?,?,?,?,?,?,?,?) 
        """, (
            req_id,
            datetime.datetime.now(datetime.UTC).isoformat(),
            prompt,
            system_prompt,
            model,
            response,
            status,
            latency_ms,
            error
        )) 
        
        await db.commit()

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.datetime.now(datetime.UTC).isoformat()}  

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    llm_response = None 
    model_used = LLM_MODEL 
    status = "success"
    error = None 

    try: 
        if not LLM_API_KEY:
            raise HTTPException(status_code=500, detail="LLM_API_KEY is not configured")         

        llm_result = await call_llm(chat_request.prompt, chat_request.system_prompt, chat_request.temperature)
        llm_response = llm_result["response"] 
        model_used = llm_result["model"] 
        
        #latency_ms = int((time.perf_counter() - start_time) * 1000)
        #await log_request_to_db(request_id, chat_request.prompt, chat_request.system_prompt, llm_result["response"], llm_result["model"], "success", latency_ms, None)
        
    except httpx.HTTPStatusError as e: 
        status = "llm_error"
        error = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
        raise HTTPException(status_code=502, detail=error) 
    except httpx.TimeoutException:
        status = "timeout"
        error = "LLM request timed out"
        raise HTTPException(status_code=504, detail="LLM timeout") 
    except Exception as e:
        status = "internal error"
        error = str(e)                    
        raise HTTPException(status_code=500, detail=str(e)) 
    finally:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await log_request(request_id, chat_request.prompt, chat_request.system_prompt, model_used, llm_response, status, latency_ms, error)

    return ChatResponse(
                request_id=request_id,
                response=llm_response, 
                model=model_used,
                latency_ms=latency_ms,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat()
            )

@app.get("/logs")
async def get_logs(limit: int = 50):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM requests ORDER BY timestamp DESC LIMIT ?", (limit,))

        rows = await cursor.fetchall()        
        return [dict(row) for row in rows]