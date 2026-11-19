from dotenv import load_dotenv
import chromadb
from google import genai

load_dotenv()
ai_client = genai.Client()

# 1. Initialize a local persistent ChromaDB (saves data to a ./chroma_db folder)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_policies")

# 2. Sample knowledge base documents + metadata
documents = [
    "Employees can claim up to 2,500 INR per month for home broadband internet bills.",
    "All production database deployments must be approved by the DevOps lead on Slack.",
    "Laptops showing hardware defects can be swapped at the IT desk on Floor 3.",
    "Annual paid leave balances reset on December 31st; up to 5 days can be carried forward.",
    "Contractor invoices must be uploaded to the finance portal before the 25th of every month.",
]

metadatas = [
    {"department": "HR", "doc_id": "POL-101"},
    {"department": "Engineering", "doc_id": "POL-102"},
    {"department": "IT", "doc_id": "POL-103"},
    {"department": "HR", "doc_id": "POL-104"},
    {"department": "Finance", "doc_id": "POL-105"},
]

ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]

# 3. Helper function to generate embeddings via Gemini API
def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = ai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )
    return [e.values for e in response.embeddings]

# 4. Upsert documents + their embeddings into ChromaDB
print("Embedding and storing documents...")
doc_embeddings = get_embeddings(documents)

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=doc_embeddings,
    metadatas=metadatas,
)

# 5. Semantic Search Query (Notice: ZERO overlapping keywords with the Wi-Fi policy!)
user_query = "Can I get reimbursed for my home Wi-Fi connection?"
query_embedding = get_embeddings([user_query])[0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,  # Fetch top 2 closest matches
    # Optional metadata filter: where={"department": "HR"}
)

print(f"\nUser Question: '{user_query}'\n")
for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    meta = results["metadatas"][0][i]
    distance = results["distances"][0][i]
    print(f"Match {i+1} (Distance: {distance:.4f} | Dept: {meta['department']}):")
    print(f" -> {doc}\n")