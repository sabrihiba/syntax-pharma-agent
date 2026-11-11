
from openai import OpenAI
from app.config import API_KEY, BASE_URL, MODEL

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=30.0
)


def ask_llm(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content or ""
