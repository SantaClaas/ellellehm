
from sqlite3 import Connection
from openai import OpenAI
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, Field

from constants import MODEL
import database


class Response(BaseModel):
    inner_thoughts: str = Field(
        description="Your inner thoughts before forming the answer you respond with in utterance. This will not be visible to anyone else than you.")
    utterance: str = Field(
        description="The response that you say after forming your thoughts from inner thoughts. This wil lbe spoken to the chat partner.")


class Agent:
    client: OpenAI
    history: ResponseInputParam
    connection: Connection

    def __init__(self, client: OpenAI, connection: Connection, message: str, initial_message: str | None = None):
        assert isinstance(message, str)
        self.client = client
        self.connection = connection
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
