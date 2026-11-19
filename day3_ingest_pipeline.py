from dotenv import load_dotenv
import chromadb
from google import genai
from day3_chunking import chunks_with_metadata  # Reuses chunks from Step 3

load_dotenv()
ai_client = genai.Client()

# 1. Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./rag_db")
collection = chroma_client.get_or_create_collection(name="pdf_knowledge_base")

# 2. Extract texts, IDs, and metadata lists
ids = [item["id"] for item in chunks_with_metadata]
texts = [item["text"] for item in chunks_with_metadata]
metadatas = [item["metadata"] for item in chunks_with_metadata]

# 3. Batch embed all chunks using Gemini
print(f"Embedding {len(texts)} chunks...")
embedding_response = ai_client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts,
)
embeddings = [e.values for e in embedding_response.embeddings]

# 4. Store in Vector DB
collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
)
print("Ingestion complete! Testing retrieval...\n")

# 5. Test Query to verify chunk + page number retrieval
query = "How fast is the database recovery during an outage?"
query_emb = ai_client.models.embed_content(
    model="gemini-embedding-001", contents=query
).embeddings[0].values

results = collection.query(query_embeddings=[query_emb], n_results=2)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"Found on Page {meta['page']}: {doc}")