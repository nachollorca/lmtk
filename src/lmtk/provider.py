"""Abstract base class for LLM providers."""

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

from lmtk.datatypes import CompletionRequest, CompletionResponse


class Provider(ABC):
    """Interface that all LLM providers must implement.

    Subclasses must define the following class attributes:
        api_key_name: The environment variable name for the provider's API key.

    The base class handles credential resolution and validation.
    Concrete providers only implement ``_get_response`` and ``_stream``,
    receiving the API key as a parameter.
    """

    api_key_name: str

    @classmethod
    def get_response(
        cls, request: CompletionRequest, stream: bool
    ) -> CompletionResponse | Iterator[str]:
        """Resolve API credentials and delegate to the provider implementation.

        See ``lmtk.core.get_response`` for parameter docs and defaults.
        """
        api_key = os.getenv(cls.api_key_name)
        if not api_key:
            raise ValueError(f"Environment variable {cls.api_key_name} not found.")

        if stream:
            return cls._stream(request, api_key)
        return cls._get_response(request, api_key)

    @classmethod
    @abstractmethod
    def _get_response(cls, request: CompletionRequest, api_key: str) -> CompletionResponse:
        """Provider-specific response generation."""
        ...

    @classmethod
    @abstractmethod
    def _stream(cls, request: CompletionRequest, api_key: str) -> Iterator[str]:
        """Stream chat completion tokens from the provider."""
        ...
