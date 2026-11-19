from dotenv import load_dotenv
import chromadb
from google import genai
from google.genai import types

load_dotenv()
ai_client = genai.Client()

# 1. Connect to the existing ChromaDB created in Day 3
chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(name="pdf_knowledge_base")

def retrieve_context(query: str, top_k: int = 3) -> str:
    """Embeds the user query and formats the top matching chunks with page numbers."""
    query_emb = ai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    ).embeddings[0].values

    results = collection.query(query_embeddings=[query_emb], n_results=top_k)

    # Combine chunks with their page numbers so the LLM can cite them
    formatted_chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        page_num = meta.get("page", "Unknown")
        formatted_chunks.append(f"\n{doc}")

    return "\n\n---\n\n".join(formatted_chunks)

# 2. Strict System Prompt to Prevent Hallucinations
SYSTEM_PROMPT = """
You are a factual technical assistant. Answer the user's question ONLY using the provided context below.
Rules:
1. Always cite the page number at the end of your sentences like [Page X].
2. If the answer is not explicitly contained in the context, reply exactly: "I cannot find this information in the provided document." Do not use outside knowledge.
"""

def ask_pdf(question: str) -> str:
    # Step A: Retrieve top chunks
    context_str = retrieve_context(question, top_k=3)

    # Step B: Augment the prompt
    augmented_prompt = f"""
Context from PDF:
{context_str}

User Question: {question}
"""

    # Step C: Generate grounded response
    response = ai_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=augmented_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,  # 0.0 ensures factual adherence to context
        ),
    )
    return response.text

# 3. Interactive RAG Loop
print("PDF RAG Bot Ready! Type 'exit' to quit.\n")
while True:
    user_q = input("Ask your PDF: ")
    if user_q.strip().lower() == "exit":
        break
    print("\nAnswer:")
    print(ask_pdf(user_q))
    print("-" * 50)