import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
import streamlit as st

load_dotenv()

st.set_page_config(page_title="RAG AI Workspace", page_icon="📄")
st.title("📄 Chat with Any PDF")

@st.cache_resource
def get_clients():
    ai_client = genai.Client()
    chroma_client = chromadb.Client()
    return ai_client, chroma_client

client, chroma_client = get_clients()
collection = chroma_client.get_or_create_collection(name="uploaded_pdf")

# --- SIDEBAR: PDF INGESTION & CHUNKING ---
with st.sidebar:
    st.header("1. Upload Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file and st.button("Process & Index PDF"):
        with st.spinner("Extracting, chunking, and embedding PDF..."):
            doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
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

            st.session_state.full_pdf_text = "\n\n".join(full_text_pages)

            emb_res = client.models.embed_content(
                model="gemini-embedding-001", contents=texts
            )
            embeddings = [e.values for e in emb_res.embeddings]

            if collection.count() > 0:
                collection.delete(ids=collection.get()["ids"])

            collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
            st.success(f"Indexed {len(texts)} chunks across {len(doc)} pages!")

# --- MAIN CHAT: ROUTER FOR FULL SUMMARY VS. VECTOR RAG ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def rag_stream(question: str):
    is_summary = any(
        w in question.lower() for w in ["summary", "sumary", "summarize", "overview", "about"]
    )

    if is_summary and "full_pdf_text" in st.session_state:
        prompt_to_send = (
    f"Summarize the key points of this document in 5-7 concise bullet points:\n\n"
    f"{st.session_state.full_pdf_text[:25000]}"
)
        sys_prompt = "You are an expert technical analyst. Summarize the provided document accurately using bullet points."
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
        prompt_to_send = f"Context from uploaded PDF:\n{context_block}\n\nQuestion: {question}"
        sys_prompt = (
            "Answer using the provided PDF context and cite page numbers like [Page X]. "
            "If the answer is not in the context, say 'Not found in the document.'"
        )
    else:
        prompt_to_send = question
        sys_prompt = "You are a helpful AI Engineer assistant."

    stream = client.models.generate_content_stream(
        model="gemini-3.5-flash-lite",
        contents=prompt_to_send,
        config=types.GenerateContentConfig(system_instruction=sys_prompt),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text

if prompt := st.chat_input("Ask a question or type 'summarize this pdf'..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        reply = st.write_stream(rag_stream(prompt))

    st.session_state.messages.append({"role": "assistant", "content": reply})