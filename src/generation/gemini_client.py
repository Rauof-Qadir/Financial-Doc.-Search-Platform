import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


class GeminiGenerator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is missing. Check your project's .env file."
            )

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        )

        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text.strip()


if __name__ == "__main__":
    generator = GeminiGenerator()

    answer = generator.generate(
        "Reply with exactly: Gemini API connection successful."
    )

    print("\nGemini API Test")
    print("------------------------------")
    print("Model:", generator.model)
    print("Response:", answer)
