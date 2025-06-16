import os
from pathlib import Path
import time
from dotenv import load_dotenv
import nanoid
from openai import OpenAI
from pydantic import BaseModel, Field
from pydub import AudioSegment

from agent import Agent
from constants import CONVERSATION_TURNS, MODEL
import database


class RoleSeed(BaseModel):
    role_1_description: str = Field(
        description="The description of the first role. The role has opposing interest to the second role.")
    role_2_description: str = Field(
        description="The description of the second role. The role has opposing interests to the first role")
    role_1_initial_message: str = Field(
        "The initial message role 1 should start the conversation with")


def ensure_directory_exists(path: Path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    print("Starting AI conversation!")
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    connection = database.initialize()

    response = client.responses.parse(
        model=MODEL,
        input=[{"role": "user", "content": "Create two role instructions for two opposing people that have conflicting interests in an awkward confrontation. The descriptions are handed to actors to act out the roles in an improvised conversation. The opposing interests should create a funny ongoing conversation. Write it in a way that each role is aware of the other without revealing too much detail. One of the roles tries to manipulate and cheat the other. Instruct each role to be short and concise."}],
        text_format=RoleSeed,
    )

    database.create_usage(
        connection, response.usage.input_tokens, response.usage.output_tokens)

    role_1_description = response.output_parsed.role_1_description
    role_1_initial_message = response.output_parsed.role_1_initial_message
    role_2_description = response.output_parsed.role_2_description

    print(f"Role 1: {role_1_description}")
    print(f"Role 2: {role_2_description}")

    agent_1 = Agent(
        client, connection, role_1_description, role_1_initial_message)
    agent_2 = Agent(
        client, connection, role_2_description)
    agents = [agent_1, agent_2]

    # Ask initial question to get the chat rolling
    last_message = role_1_initial_message
    print(f"[Agent 1 Initial Message]:\t{last_message}")
    conversation_id = nanoid.generate()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    conversations_directory = Path.cwd() / "conversations" / \
        f"{timestamp}-{conversation_id}"
    turns_directory = conversations_directory / "turns"
    ensure_directory_exists(turns_directory)
    print(f"Writing audio files to: {conversations_directory}")

    def generate_audio(agent: Agent, turn: int, message: str) -> Path:
        turn_path = turns_directory / f"{turn}.mp3"
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=agent.text_to_speech_voice,
            input=message,
            instructions=agent.role_description,
        ) as response:
            response.stream_to_file(turn_path)

        return turn_path

    conversation_audio = AudioSegment.empty()
    try:
        turn_files = []

        turn_path = generate_audio(agent_1, 0, last_message)
        turn_files.append(turn_path)
        segment = AudioSegment.from_mp3(turn_path)
        conversation_audio += segment

        for turn in range(1, CONVERSATION_TURNS):
            agent_index = turn % len(agents)
            current_agent = agents[agent_index]
            (last_message, inner_thoughts) = current_agent.message(last_message)
            print(
                f"\n[Agent {agent_index + 1} inner thoughts]:\t{inner_thoughts}")
            print(
                f"\n[Agent {agent_index + 1}]:\t{last_message}")
            turn_path = generate_audio(current_agent, turn, last_message)
            turn_files.append(turn_path)
            segment = AudioSegment.from_mp3(turn_path)
            conversation_audio += segment

        print("Conversation completed!")
    finally:
        conversation_audio.export(conversations_directory / "full.mp3")


if __name__ == "__main__":
    main()
