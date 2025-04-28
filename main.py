import os
from dotenv import load_dotenv
from openai import OpenAI

AGENT_1_PROMPT = "You are Agent 1, a curious assistant who asks thoughtful questions."
AGENT_2_PROMPT = "You are Agent 2, a knowledgeable assistant who provides detailed answers."
MODEL = "gpt-4o-mini"
CONVERSATION_TURNS = 5


class Agent:
    def __init__(self, client):
        self.client = client
        # Instruct the character
        self.history = [
            {
                "role": "system",
                "content": "You are a helpful and curious assistant. Answer short and with a question so that the conversation continues"
            },
        ]

    def message(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=self.history
        )
        content = response.choices[0].message.content
        self.history.append({"role": "system", "content": content})
        return content


def main():
    print("Starting AI conversation!")
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    agent_1 = Agent(client)
    agent_2 = Agent(client)

    # Ask initial question to get the chat rolling
    last_message = "How are you doing?"
    print("[Agent 1]:", last_message)

    last_message = agent_2.message(last_message)
    print("[Agent 2]:", last_message)

    last_message = agent_1.message(last_message)
    print("[Agent 1]:", last_message)

    last_message = agent_2.message(last_message)
    print("[Agent 1]:", last_message)

    print("Conversation completed!")


if __name__ == "__main__":
    main()
