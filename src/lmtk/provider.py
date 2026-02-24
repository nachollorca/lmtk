"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from pydantic import BaseModel

from lmtk.datatypes import Message, ModelResponse
from lmtk.errors import MissingAPIKeyError
from lmtk.secrets import resolve_api_key


# TODO: check if the methods should be static apart from abstract
# ? Maybe the api key validation can actually run on every generate_response and then we just dont need an __init__?
class Provider(ABC):
    """Interface that all LLM providers must implement.

    Subclasses must define the following class attributes:

        models: A list of supported model identifiers.
        api_key_name: The environment variable name for the provider's API key.
    """

    models: list[str]
    api_key_name: str

    def __init__(self) -> None:
        """Initialize the provider, loading any required credentials."""
        self.api_key = resolve_api_key(self.api_key_name)
        if not self.api_key:
            raise MissingAPIKeyError(self.api_key_name)

    @abstractmethod
    def get_response(
        self,
        model: str,
        messages: list[Message],
        system_instruction: str | None = None,
        output_schema: BaseModel | None = None,
        stream: bool = False,
        generation_kargs: dict = {"temperature": 0},
    ) -> ModelResponse:
        """Docstring."""
        ...

    @abstractmethod
    def _stream(self, *args, **kwargs) -> Iterator[str]:
        """Stream chat completion tokens from the provider.

        Yields:
            Individual content tokens as they arrive.
        """
