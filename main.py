import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from openai.types.responses import ResponseInputParam

MODEL = "gpt-4o-mini"
CONVERSATION_TURNS = 20


class RoleSeed(BaseModel):
    role_1_description: str = Field(
        description="The description of the first role. The role has opposing interest to the second role.")
    role_2_description: str = Field(
        description="The description of the second role. The role has opposing interests to the first role")
    role_1_initial_message: str = Field(
        "The initial message role 1 should start the conversation with")


class Response(BaseModel):
    # inner_thoughts: str = Field(
    #     description="Your inner thoughts before forming the answer you respond with in utterance. This will not be visible to anyone else than you.")
    utterance: str = Field(
        description="The response that you say after forming your thoughts from inner thoughts. This wil lbe spoken to the chat partner.")


class Agent:
    client: OpenAI
    history: ResponseInputParam

    def __init__(self, client: OpenAI, message: str, initial_message: str | None = None):
        self.client = client
        # Instruct the character
        self.history = [
            {
                "role": "system",
                "content": message
            },
        ]
        if initial_message != None:
            self.history.append({"role": "user", "content": initial_message})

    def message(self, message: str) -> tuple[str, str]:
        self.history.append({"role": "user", "content": message})

        response = self.client.responses.parse(
            model=MODEL,
            input=self.history,
            text_format=Response
        )

        content = response.output_parsed.utterance
        self.history.append(
            {"role": "system", "content": content})
        # return (content, response.output_parsed.inner_thoughts)
        return (content, "")


def main():
    print("Starting AI conversation!")
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    response = client.responses.parse(
        model=MODEL,
        input=[{"role": "user", "content": "Create two role instructions for two opposing that have conflicting interests in an awkward confrontation. The descriptions are handed to actors to act out the roles in an improvised conversation. The opposing interests should create a funny ongoing conversation. Write it in a way that each role is aware of the other without revealing too much detail. One of the roles tries to manipulate and cheat the other. Instruct each role to be short and concise."}],
        text_format=RoleSeed,
    )

    role_1 = response.output_parsed.role_1_description
    role_1_initial = response.output_parsed.role_1_initial_message
    role_2 = response.output_parsed.role_2_description

    print(f"Role 1: {role_1}")
    print(f"Role 2: {role_2}")

    agent_1 = Agent(
        client, role_1, role_1_initial)
    agent_2 = Agent(
        client, role_2)
    agents = [agent_1, agent_2]

    # Iterable that switches between 0 and 1 to switch between agent 0 and 1
    # to switch for each chat turn
    agent_turn_indices = map(lambda turn_index: turn_index %
                             len(agents), range(CONVERSATION_TURNS))

    # Ask initial question to get the chat rolling
    last_message = role_1_initial
    print(f"[Agent 1 Initial Message]:\t{last_message}")
    for agent_index in agent_turn_indices:
        current_agent = agents[agent_index]
        (last_message, inner_thoughts) = current_agent.message(last_message)
        # print(f"\n[Agent {agent_index + 1} inner thoughts]: {inner_thoughts}")
        print(
            f"\n[Agent {agent_index + 1}]: {last_message}")

    print("Conversation completed!")


if __name__ == "__main__":
    main()
