from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
client = genai.Client()

# 1. Define the exact JSON structure you want using Pydantic
class CandidateProfile(BaseModel):
    full_name: str = Field(description="Candidate's full name")
    years_of_experience: int = Field(description="Total years of work experience")
    skills: list[str] = Field(description="List of technical programming skills")
    is_python_developer: bool = Field(description="True if they know Python")

# 2. Messy, unstructured input text
raw_resume_text = """
Hey! I'm Priya Sharma. I've been working as a backend developer in Bangalore for the last 4 years.
Mostly I build APIs using FastAPI, Python, PostgreSQL, and Docker, and recently started learning LangChain.
"""

# 3. Force the LLM to respond strictly matching the Pydantic schema
response = client.models.generate_content(
    model="gemini-3.8-flash", # <-- Update this line
    contents=f"Extract the candidate details from this text:\n{raw_resume_text}",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CandidateProfile,
        temperature=0.0,
    ),
)

# 4. Print the raw JSON string or use the parsed Pydantic object directly
print("Raw JSON Output:\n", response.text)

parsed_data: CandidateProfile = response.parsed
print("\nAccessing as Python Object:")
print(f"Name: {parsed_data.full_name} | Exp: {parsed_data.years_of_experience} yrs | Skills: {parsed_data.skills}")