from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()
ai_client = genai.Client()

app = FastAPI(title="Gen AI Engineer API", version="1.0")

# Request schema from frontend
class ChatRequest(BaseModel):
    prompt: str

# Response schema to frontend
class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    try:
        response = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=payload.prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful AI Engineer assistant. Keep answers concise.",
                temperature=0.3,
            ),
        )
        return ChatResponse(response=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)