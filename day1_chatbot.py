import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # Loads GEMINI_API_KEY from .env if you use one

# 1. Initialize the client
client = genai.Client()

# 2. Configure System Instructions & Temperature
config = types.GenerateContentConfig(
    system_instruction="You are a concise, practical Senior AI Engineer mentor. Keep answers under 3 sentences.",
    temperature=0.3,  # Lower = more deterministic/factual, Higher (0.8+) = more creative
)

# 3. Start a stateful chat session
chat = client.chats.create(model="gemini-3.8-flash", config=config)

print("Chatbot initialized! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        break

    # Send message (the SDK automatically appends user_input and the response to history)
    response = chat.send_message(user_input)
    print(f"\nAI: {response.text}\n")

# Optional: Inspect how the history array looks under the hood
print("--- Conversation History Stored in Memory ---")
for message in chat.get_history():
    print(f"{message.role}: {message.parts[0].text[:60]}...")