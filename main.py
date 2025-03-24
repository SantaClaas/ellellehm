import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


QUESTION = "What is the capital of France?"
SYSTEM_PROMPT = "You are a helpful assistant that can answer quesions."
MODEL = "gpt-4o"


def main():
    print("Starting ellellehm!")
    load_dotenv()

    # This is the default and can be omitted
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION}
    ]

    # response = client.responses.create(
    #     model="gpt-4o",
    #     instructions="You are a coding assistant that talks like a pirate.",
    #     input="How do I check if a Python object is an instance of a class?",
    # )
    # print(response.output_text)

    response = client.chat.completions.create(model=MODEL, messages=messages)
    text = response.choices[0].message.content
    print("Response:", text)
    print("Completed ellellehm")


if __name__ == "__main__":
    main()
