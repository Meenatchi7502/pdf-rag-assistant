from dotenv import load_dotenv
from ddgs import DDGS
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
client = genai.Client()

def search_web(query: str) -> list[dict]:
    """Searches the web for live company news, product launches, and pricing."""
    print(f"  [Researcher Agent] Searching: '{query}'...")
    with DDGS() as ddgs:
        return [
            {"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")}
            for r in list(ddgs.text(query, max_results=3))
        ]

# Output schema for Agent 2
class CompetitorBrief(BaseModel):
    company_name: str
    key_recent_announcements: list[str] = Field(description="Top 3 recent product or business updates")
    target_audience: str
    strategic_threat_level: str = Field(description="LOW, MEDIUM, or HIGH with a 1-sentence reason")
    source_urls: list[str]

def run_multi_agent_pipeline(company: str) -> CompetitorBrief:
    # --- AGENT 1: The Web Researcher ---
    print(f"\n1. Starting Researcher Agent for '{company}'...")
    research_response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=f"Research the latest AI product announcements, pricing, and news for {company}.",
        config=types.GenerateContentConfig(
            system_instruction="You are a senior technical researcher. Gather detailed facts and exact URLs using search_web.",
            tools=[search_web],
            temperature=0.1,
        ),
    )
    raw_research_notes = research_response.text
    print("  [Researcher Agent] Research complete! Handing notes to Analyst Agent...")

    # --- AGENT 2: The Executive Analyst (Structured Synthesis) ---
    print("2. Starting Executive Analyst Agent...")
    analyst_response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=f"Convert these raw research notes into an executive brief:\n\n{raw_research_notes}",
        config=types.GenerateContentConfig(
            system_instruction="You are a Chief Product Officer. Synthesize raw research into a sharp competitor brief.",
            response_mime_type="application/json",
            response_schema=CompetitorBrief,
            temperature=0.0,
        ),
    )
    return analyst_response.parsed

if __name__ == "__main__":
    brief = run_multi_agent_pipeline("Perplexity AI")
    print("\n=== FINAL STRUCTURED MULTI-AGENT OUTPUT ===")
    print(brief.model_dump_json(indent=2))