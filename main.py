import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

AGENT_1_PROMPT = "You are Agent 1, a curious assistant who asks thoughtful questions."
AGENT_2_PROMPT = "You are Agent 2, a knowledgeable assistant who provides detailed answers."
MODEL = "gpt-4o-mini"
CONVERSATION_TURNS = 20


class Response(BaseModel):
    inner_thoughts: str = Field(
        description="Your inner thoughts before forming the answer you respond with in utterance. This will not be visible to anyone else than you.")
    utterance: str = Field(
        description="The response that you say after forming your thoughts from inner thoughts. This wil lbe spoken to the chat partner.")


class Agent:
    def __init__(self, client: OpenAI, message: str):
        self.client = client
        # Instruct the character
        self.history = [
            {
                "role": "system",
                "content": message
            },
        ]

    def message(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})

        response = self.client.responses.parse(
            model=MODEL,
            input=self.history,
            text_format=Response
        )

        self.history.append(
            {"role": "system", "content": response.output_parsed.utterance})
        return response.output_parsed.utterance


def main():
    print("Starting AI conversation!")
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": "You are a helpful and curious assistant. Answer short and with a question so that the conversation continues. When you answer use your inner thoughts and what you speak in the utterance. The utterance is what your chat partner can see. Your inner thoughts are only visible to you."
            },
            {
                "role": "user",
                "content": "hey how is it going?"
            }],
        text_format=Response
    )

    print("[Response utterance]:\t", response.output_parsed.utterance)
    print("[Inner thought]:\t", response.output_parsed.inner_thoughts)

    system_prompt = "You are a helpful and curious assistant. Answer short and with a question so that the conversation continues. When you answer use your inner thoughts and what you speak in the utterance. The utterance is what your chat partner can see. Your inner thoughts are only visible to you."
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
        last_message = current_agent.message(last_message)
        print(
            f"[Agent {agent_index + 1}]: {last_message}")

    print("Conversation completed!")


if __name__ == "__main__":
    main()
