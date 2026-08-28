import os
from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()


# Get API key
api_key = os.getenv("GEMINI_API_KEY")


if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")


# Create Gemini client
client = genai.Client(api_key=api_key)


def ask_gemini(prompt):

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        print("Gemini API Error:", e)
        return "Sorry, the AI service is temporarily unavailable. Please try again later."


if __name__ == "__main__":

    answer = ask_gemini(
        "Explain Artificial Intelligence in simple words."
    )

    print("\nGemini Answer:")
    print(answer)