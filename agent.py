
import random
from sqlite3 import Connection
import typing
from openai import OpenAI
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, Field

from constants import MODEL
import database

type OpenAiTextToSpeechVoices = typing.Literal['alloy', 'ash',
                                               'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer']


voice_options = list(typing.get_args(
    # Duplicating type because get_args does not work with type alias of OpenAiTextToSpeechVoices and Idl anymore
    typing.Literal['alloy', 'ash', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer']))
print("VOICE OPTIONS", voice_options)
random.shuffle(voice_options)


class Response(BaseModel):
    inner_thoughts: str = Field(
        description="Your inner thoughts before forming the answer you respond with in utterance. This will not be visible to anyone else than you.")
    utterance: str = Field(
        description="The response that you say after forming your thoughts from inner thoughts. This wil lbe spoken to the chat partner.")


class Agent:
    client: OpenAI
    history: ResponseInputParam
    connection: Connection
    role_description: str
    text_to_speech_voice: OpenAiTextToSpeechVoices

    def __init__(self, client: OpenAI, connection: Connection, role_description: str, initial_message: str | None = None):
        assert isinstance(role_description, str)
        self.client = client
        self.connection = connection
        self.role_description = role_description

        if len(voice_options) == 0:
            raise "All voices for the conversation have been taken. There is no more voice available to differentiate agents in the conversation"

        self.text_to_speech_voice = voice_options.pop()

        # Instruct the character
        self.history = [
            {
                "role": "system",
                "content": role_description
            },
        ]
        if initial_message != None:
            self.history.append({"role": "user", "content": initial_message})

    def message(self, message: str) -> tuple[str, str]:
        assert isinstance(message, str)
        self.history.append({"role": "user", "content": message})

        response = self.client.responses.parse(
            model=MODEL,
            input=self.history,
            text_format=Response
        )

        database.create_usage(
            self.connection, response.usage.input_tokens, response.usage.output_tokens)

        content = response.output_parsed.utterance
        self.history.append(
            {"role": "system", "content": content})

        return (content, response.output_parsed.inner_thoughts)
        # return (content, "")
