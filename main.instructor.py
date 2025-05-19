import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from openai.types.responses import ResponseInputParam
import instructor

MODEL = "gpt-4o-mini"
CONVERSATION_TURNS = 20


class Response(BaseModel):
    inner_thoughts: str = Field(
        description="Your inner thoughts before forming the answer you respond with in utterance. This will not be visible to anyone else than you.")
    utterance: str = Field(
        description="The response that you say after forming your thoughts from inner thoughts. This wil lbe spoken to the chat partner.")


class Agent:
    client: any
    history: ResponseInputParam

    def __init__(self, client: any, message: str):
        self.client = client
        # Instruct the character
        self.history = [
            {
                "role": "system",
                "content": message
            },
        ]

    def message(self, message: str) -> tuple[str, str]:
        self.history.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=self.history,
            response_model=Response
        )

        content = response.utterance
        self.history.append(
            {"role": "system", "content": content})
        return (content, response.inner_thoughts)


def main():
    print("Starting AI conversation!")
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    client = instructor.from_openai(client)

    system_prompt = "You are a helpful and curious assistant. Answer short and with a question so that the conversation continues."

    agent_1 = Agent(
        client, system_prompt)
    agent_2 = Agent(
        client, system_prompt)
    agents = [agent_1, agent_2]

    # Iterable that switches between 0 and 1 to switch between agent 0 and 1
    # to switch for each chat turn
    agent_turn_indices = map(lambda turn_index: turn_index %
                             len(agents), range(CONVERSATION_TURNS))

    # Ask initial question to get the chat rolling
    last_message = "How are you doing?"
    for agent_index in agent_turn_indices:
        current_agent = agents[agent_index]
        (last_message, inner_thoughts) = current_agent.message(last_message)
        print(f"\n[Agent {agent_index + 1} inner thoughts]: {inner_thoughts}")
        print(
            f"[Agent {agent_index + 1}]: {last_message}")

    print("Conversation completed!")


if __name__ == "__main__":
    main()
