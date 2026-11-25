import asyncio
import os
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import pymupdf

load_dotenv()

app = FastAPI(title="PDF RAG & Summarization API")

# Allow React frontend (localhost:5173 or Vercel) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="api_pdf_store")

# Store extracted full text in memory for summary routing
app_state = {"full_pdf_text": "", "filename": ""}


class ChatRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

    ids, texts, metadatas, full_text_pages = [], [], [], []
    for page_idx in range(len(doc)):
        page_text = doc[page_idx].get_text("text").strip()
        if not page_text:
            continue
        full_text_pages.append(f"--- Page {page_idx + 1} ---\n{page_text}")
        for chunk_idx, chunk in enumerate(splitter.split_text(page_text)):
            ids.append(f"p{page_idx+1}_c{chunk_idx}")
            texts.append(chunk)
            metadatas.append({"page": page_idx + 1})

    if not texts:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")

    app_state["full_pdf_text"] = "\n\n".join(full_text_pages)
    app_state["filename"] = file.filename

    emb_res = client.models.embed_content(
        model="gemini-embedding-001", contents=texts
    )
    embeddings = [e.values for e in emb_res.embeddings]

    if collection.count() > 0:
        collection.delete(ids=collection.get()["ids"])

    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return {
        "status": "success",
        "filename": file.filename,
        "pages": len(doc),
        "chunks": len(texts),
    }


@app.post("/chat")
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
    elif collection.count() > 0:
        q_emb = client.models.embed_content(
            model="gemini-embedding-001", contents=question
        ).embeddings[0].values
        results = collection.query(query_embeddings=[q_emb], n_results=5)

        chunks = [
            f"[Page {meta['page']}]: {doc}"
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
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


# Mount StaticFiles last so /upload and /chat match first.
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")