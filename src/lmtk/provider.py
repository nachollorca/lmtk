"""Abstract base class for LLM providers."""

import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import requests

from lmtk.datatypes import CompletionRequest, CompletionResponse
from lmtk.errors import STATUS_TO_ERROR, AuthenticationError, ProviderError


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
        api_key = cls._resolve_api_key()

        if stream:
            return cls._stream_response(request, api_key)
        return cls._get_full_response(request, api_key)

    @classmethod
    @abstractmethod
    def _get_full_response(cls, request: CompletionRequest, api_key: str) -> CompletionResponse:
        """Provider-specific response generation."""
        ...

    @classmethod
    @abstractmethod
    def _stream_response(cls, request: CompletionRequest, api_key: str) -> Iterator[str]:
        """Stream chat completion tokens from the provider."""
        ...

    @classmethod
    def _make_request(
        cls, url: str, *, json: dict, headers: dict | None = None, **kwargs
    ) -> requests.Response:
        """Send a POST request and raise on non-200 responses.

        This is an ergonomic wrapper around ``requests.post`` that maps
        common HTTP status codes to specific ``ProviderError`` subclasses
        so callers get uniform error handling across providers.

        Returns the :class:`requests.Response` on success.
        """
        response = requests.post(url, json=json, headers=headers, **kwargs)

        if response.status_code != 200:
            error_cls = STATUS_TO_ERROR.get(response.status_code, ProviderError)
            raise error_cls(
                status_code=response.status_code,
                message=f"{cls.__name__}: HTTP {response.status_code} - {response.reason}",
                provider=cls.__name__,
                body=response.text,
            )

        return response

    @classmethod
    def _resolve_api_key(cls) -> str:
        """Read the API key from the environment.

        Raises :class:`AuthenticationError` when the variable is unset or empty.
        """
        api_key = os.getenv(cls.api_key_name)
        if not api_key:
            raise AuthenticationError(
                status_code=0,
                message=f"Environment variable {cls.api_key_name} not set.",
                provider=cls.__name__,
            )
        return api_key


def load_provider(name: str) -> type[Provider]:
    """Gets the appropriate Provider class for the given provider name.

    Imports ``<Name>Provider`` from ``lmtk.providers.<name>``
    (e.g. ``"mistral"`` -> ``MistralProvider``).

    Raises:
        ImportError: If no module matches *name*.
        AttributeError: If the module does not contain the expected class.
    """
    module = importlib.import_module(f"lmtk.providers.{name}")
    class_name = f"{name.capitalize()}Provider"
    return getattr(module, class_name)
