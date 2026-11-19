import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    """Extracts text page-by-page along with page number metadata."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text").strip()
        if text:
            pages.append({"page_number": page_num + 1, "text": text})
    return pages

# Fallback sample text if you don't have a PDF handy right this second
sample_pages = [
    {
        "page_number": 1,
        "text": (
            "Project Orion Architecture Specification.\n\n"
            "The backend service is built on FastAPI and runs on AWS ECS Fargate. "
            "All incoming client requests are rate-limited to 120 requests per minute using Redis. "
            "Authentication is handled via OAuth2 JWT tokens that expire every 15 minutes."
        ),
    },
    {
        "page_number": 2,
        "text": (
            "Database & Disaster Recovery Policy.\n\n"
            "Primary transactional data is stored in PostgreSQL 16 with automated daily snapshots. "
            "In the event of a regional outage, the failover replica in Mumbai (ap-south-1) "
            "is promoted within 10 minutes (RTO = 10 mins, RPO = 5 mins)."
        ),
    },
]

# Load from sample.pdf if present, otherwise use sample_pages
pages = extract_pages_from_pdf("sample.pdf") if os.path.exists("sample.pdf") else sample_pages

# Initialize RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,       # In real PDFs, use 500 to 1000 characters
    chunk_overlap=30,     # Typically 10% to 20% of chunk_size
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks_with_metadata = []
for page in pages:
    sub_chunks = splitter.split_text(page["text"])
    for idx, chunk_text in enumerate(sub_chunks):
        chunks_with_metadata.append({
            "id": f"page_{page['page_number']}_chunk_{idx}",
            "text": chunk_text,
            "metadata": {"page": page["page_number"], "chunk_index": idx},
        })

print(f"Created {len(chunks_with_metadata)} chunks:\n")
for c in chunks_with_metadata:
    print(f"[{c['id']} | Page {c['metadata']['page']}]: {c['text']}\n")