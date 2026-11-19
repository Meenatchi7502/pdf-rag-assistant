from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

def refund_order(order_id: str, reason: str) -> dict:
    """Processes a customer refund for a given order ID and reason."""
    return {"status": "SUCCESS", "order_id": order_id, "refunded_amount_inr": 1499}

# Map function names to actual Python callables
TOOL_MAP = {"refund_order": refund_order}

# Disable automatic execution so WE control when the function runs
config = types.GenerateContentConfig(
    tools=[refund_order],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    temperature=0.0,
)

# 1. Initial user message
history = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Please refund my order ORD-9921 because the screen arrived cracked.")],
    )
]

response = client.models.generate_content(
    model="gemini-3.8-flash", contents=history, config=config
)

# 2. Check if the LLM wants to call a function
if response.function_calls:
    # Always append the model's tool-call request to history first
    history.append(response.candidates[0].content)

    for fn_call in response.function_calls:
        print(f"LLM requested tool: {fn_call.name} with args: {fn_call.args}")

        # Human-in-the-loop approval check
        confirm = input(f"Approve executing {fn_call.name}? (y/n): ")
        if confirm.lower() == "y":
            tool_output = TOOL_MAP[fn_call.name](**fn_call.args)
        else:
            tool_output = {"status": "CANCELLED_BY_USER"}

        # 3. Append the tool execution result to history
        history.append(
            types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=fn_call.name, response={"result": tool_output}
                    )
                ],
            )
        )

    # 4. Send the tool result back to the LLM for the final answer
    final_response = client.models.generate_content(
        model="gemini-3.8-flash", contents=history, config=config
    )
    print(f"\nAI: {final_response.text}")