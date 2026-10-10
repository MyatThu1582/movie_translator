
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

client = Groq(api_key=api_key)

model = "openai/gpt-oss-20b"

try:
    print(f"Testing model: {model}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional movie subtitle translator. "
                    "Translate dialogue into natural, conversational Burmese. "
                    "Return only the Burmese translation."
                ),
            },
            {
                "role": "user",
                "content": "I never thought I would see you again. I missed you so much.",
            },
        ],
        temperature=0.3,
    )

    print("\nSUCCESS! Model responded.")
    print("Burmese translation:")
    print(response.choices[0].message.content)

except Exception as e:
    print("\nTEST FAILED")
    print(type(e).__name__, str(e))
