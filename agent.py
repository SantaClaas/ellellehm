
from openai import OpenAI
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, Field

from constants import MODEL


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
