"""Contains the data contracts used across the app."""

from dataclasses import asdict, dataclass

from pydantic import BaseModel


@dataclass
class Message:
    """Represents a single message in a conversation."""

    content: str
    role: str

    def to_dict(self) -> dict:
        """Converts the message to a dictionary."""
        return asdict(self)


@dataclass
class UserMessage(Message):
    """Wrapper for a message sent by the user."""

    role: str = "user"


@dataclass
class AssistantMessage(Message):
    """Wrapper for a message sent by the assistant."""

    role: str = "assistant"


@dataclass(frozen=True)
class CompletionRequest:
    """Bundles the common parameters for a completion call.

    Built by ``lmtk.core.get_response`` and threaded through the provider
    layer so that adding a new parameter is a single-field change here.
    """

    model_id: str
    messages: list[Message]
    system_instruction: str | None
    output_schema: type[BaseModel] | None
    generation_kwargs: dict


@dataclass
class CompletionResponse:
    """The result of a completion call, including usage and timing."""

    content: str
    input_tokens: int
    output_tokens: int
    latency: float
    parsed: BaseModel | None = None

    @property
    def message(self):
        """Converts the response to an AssistantMessage."""
        return AssistantMessage(self.content)
