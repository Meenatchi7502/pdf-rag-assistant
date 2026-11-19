from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

# Convert a sentence into a numerical vector
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="How do I reset my company VPN password?",
)

vector = result.embeddings[0].values
print(f"Vector Length (Dimensions): {len(vector)}")
print(f"First 5 numbers of the vector: {vector[:5]}")