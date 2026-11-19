from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

# 1. Define your Python tools with strict type hints and clear docstrings
def get_stock_price(ticker: str) -> dict:
    """Fetches the live stock price and currency for a given stock ticker symbol (e.g., 'TCS', 'INFY', 'NVDA')."""
    print(f"\n[TOOL EXECUTED] Fetching stock price for: {ticker}...")
    mock_prices = {
        "TCS": {"price": 4120.50, "currency": "INR"},
        "INFY": {"price": 1850.00, "currency": "INR"},
        "NVDA": {"price": 138.25, "currency": "USD"},
    }
    return mock_prices.get(ticker.upper(), {"error": f"Ticker {ticker} not found"})

def calculate_loan_emi(principal: float, annual_rate_percent: float, tenure_years: int) -> dict:
    """Calculates the monthly EMI and total interest payable for a loan."""
    print(f"\n[TOOL EXECUTED] Calculating EMI for {principal} at {annual_rate_percent}% for {tenure_years} yrs...")
    monthly_rate = (annual_rate_percent / 12) / 100
    months = tenure_years * 12
    emi = (principal * monthly_rate * ((1 + monthly_rate) ** months)) / (((1 + monthly_rate) ** months) - 1)
    total_payment = emi * months
    return {
        "monthly_emi": round(emi, 2),
        "total_interest": round(total_payment - principal, 2),
        "total_payable": round(total_payment, 2),
    }

# 2. Attach the Python functions to the chat configuration
config = types.GenerateContentConfig(
    system_instruction="You are a helpful financial assistant. Use your tools whenever exact data or math is needed.",
    tools=[get_stock_price, calculate_loan_emi],
    temperature=0.0,
)

chat = client.chats.create(model="gemini-2.5-flash", config=config)

# 3. Ask a question that requires calling BOTH tools in a single turn (Parallel Tool Calling)
prompt = (
    "What is the stock price of TCS and INFY right now? "
    "Also, if I take a loan of 10,00,000 INR at 9% interest for 5 years, what will my monthly EMI be?"
)
print(f"User: {prompt}")

response = chat.send_message(prompt)
print(f"\nFinal AI Response:\n{response.text}")