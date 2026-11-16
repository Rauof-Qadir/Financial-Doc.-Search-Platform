import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

print("\nAVAILABLE GEMINI FLASH MODELS")
print("-" * 45)

try:
    found = False

    for model in client.models.list():
        name = getattr(model, "name", "") or ""
        actions = getattr(model, "supported_actions", []) or []

        if "generateContent" in actions and "flash" in name.lower():
            print(name)
            found = True

    if not found:
        print("No Flash models found in the returned model list.")

except Exception as error:
    print("\nMODEL LIST REQUEST FAILED")
    print(type(error).__name__)
    print(str(error))
