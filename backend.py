import asyncio
import os
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from pydantic import BaseModel
import pymupdf

load_dotenv()

app = FastAPI(title="PDF RAG & Summarization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

# Lightweight in-memory vector store (avoids ChromaDB 512MB RAM crashes on Render)
app_state = {
    "full_pdf_text": "",
    "filename": "",
    "chunks": [],
    "metadatas": [],
    "embeddings": None,
}


class ChatRequest(BaseModel):
    question: str


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        pdf_bytes = await file.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

        texts, metadatas, full_text_pages = [], [], []
        for page_idx in range(len(doc)):
            page_text = doc[page_idx].get_text("text").strip()
            if not page_text:
                continue
            full_text_pages.append(f"--- Page {page_idx + 1} ---\n{page_text}")
            for chunk in splitter.split_text(page_text):
                texts.append(chunk)
                metadatas.append({"page": page_idx + 1})

        if not texts:
            raise HTTPException(status_code=400, detail="No readable text found in PDF.")

        # Embed in batches of 50 to avoid Gemini API batch size limits
        embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb_res = client.models.embed_content(
                model="gemini-embedding-001", contents=batch
            )
            embeddings.extend([e.values for e in emb_res.embeddings])

        emb_matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_matrix = emb_matrix / np.maximum(norms, 1e-10)

        app_state["full_pdf_text"] = "\n\n".join(full_text_pages)
        app_state["filename"] = file.filename
        app_state["chunks"] = texts
        app_state["metadatas"] = metadatas
        app_state["embeddings"] = emb_matrix

        return {
            "status": "success",
            "filename": file.filename,
            "pages": len(doc),
            "chunks": len(texts),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")


@app.post("/api/chat")
def chat_stream(req: ChatRequest):
    question = req.question
    is_summary = any(
        w in question.lower()
        for w in ["summary", "sumary", "summarize", "overview", "about"]
    )

    if is_summary and app_state["full_pdf_text"]:
        prompt_to_send = (
            f"Summarize the key points of this document in 5-7 concise bullet points:\n\n"
            f"{app_state['full_pdf_text'][:25000]}"
        )
        sys_prompt = "You are a concise technical analyst. Be brief and direct."
    elif app_state["embeddings"] is not None and len(app_state["chunks"]) > 0:
        q_emb = np.array(
            client.models.embed_content(
                model="gemini-embedding-001", contents=question
            ).embeddings[0].values,
            dtype=np.float32,
        )
        q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-10)

        scores = app_state["embeddings"] @ q_emb
        top_k = min(5, len(app_state["chunks"]))
        top_indices = np.argsort(scores)[::-1][:top_k]

        chunks = [
            f"[Page {app_state['metadatas'][idx]['page']}]: {app_state['chunks'][idx]}"
            for idx in top_indices
        ]
        context_block = "\n\n---\n\n".join(chunks)
        prompt_to_send = (
            f"Context from uploaded PDF:\n{context_block}\n\nQuestion: {question}"
        )
        sys_prompt = (
            "Answer using the provided PDF context and cite page numbers like [Page X]. "
            "If the answer is not in the context, say 'Not found in the document.'"
        )
    else:
        prompt_to_send = question
        sys_prompt = "You are a helpful AI Engineer assistant."

    def token_generator():
        stream = client.models.generate_content_stream(
            model="gemini-3.5-flash-lite",
            contents=prompt_to_send,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                max_output_tokens=600,
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    return StreamingResponse(token_generator(), media_type="text/plain")


# Keep this mount last so the API routes match first.
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")