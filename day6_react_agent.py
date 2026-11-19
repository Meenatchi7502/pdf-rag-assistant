from dotenv import load_dotenv
from ddgs import DDGS
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

# 1. Live Web Search Tool (Free via DuckDuckGo)
def search_web(query: str) -> list[dict]:
    """Searches the live internet for recent news, company data, or technical topics."""
    print(f"\n[AGENT ACTION] Searching web for: '{query}'...")
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        return [
            {"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")}
            for r in results
        ]

def calculate_growth_percentage(old_val: float, new_val: float) -> float:
    """Calculates the exact percentage change between two numbers."""
    print(f"\n[AGENT ACTION] Calculating % change from {old_val} to {new_val}...")
    return round(((new_val - old_val) / old_val) * 100, 2)

# 2. Configure the ReAct Agent
agent_config = types.GenerateContentConfig(
    system_instruction=(
        "You are an autonomous AI Market Research Agent. "
        "Break complex questions down into multiple search queries if needed, "
        "verify facts using search_web, and always cite source URLs."
    ),
    tools=[search_web, calculate_growth_percentage],
    # Allow the agent to chain up to 5 consecutive tool calls autonomously
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        maximum_remote_calls=5
    ),
    temperature=0.1,
)

# 3. Give the Agent a Multi-Step Goal
goal = (
    "Find the latest major AI model releases from Google and Anthropic, "
    "and compare their main developer features in 4 bullet points."
)
print(f"Goal: {goal}")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=goal,
    config=agent_config,
)

print(f"\n--- Final Agent Report ---\n{response.text}")