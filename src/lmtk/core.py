"""Contains the main logic to call language model APIs."""

from collections.abc import Iterator

from pydantic import BaseModel

from lmtk.datatypes import CompletionRequest, CompletionResponse, Message
from lmtk.provider import Provider


def _load_provider(name: str) -> Provider:
    """Gets the appropriate Provider class for the given string."""
    # load the corresponding Provider
    raise NotImplementedError()


def get_response(
    model: str,
    messages: list[Message],
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    stream: bool = False,
    generation_kwargs: dict | None = None,
) -> CompletionResponse | Iterator[str]:
    """Generate a response from a language model.

    Args:
        model: Provider-prefixed model identifier (e.g. ``"mistral:devstral-latest"``).
        messages: The conversation history.
        system_instruction: Optional system prompt prepended to the conversation.
        output_schema: Optional Pydantic model class for structured output.
            Mutually exclusive with *stream*.
        stream: If ``True``, return an iterator of content tokens instead of
            a complete ``ModelResponse``. Mutually exclusive with *output_schema*.
        generation_kwargs: Additional generation parameters forwarded to the
            provider (e.g. ``temperature``, ``max_tokens``).
            Defaults to ``{"temperature": 0}``.

    Returns:
        A ``CompletionResponse`` with the generated content and metadata, or an
        iterator of string tokens when *stream* is ``True``.
    """
    if output_schema and stream:
        raise ValueError("Only `stream` or `output_schema` can be set, not both.")

    if generation_kwargs is None:
        generation_kwargs = {"temperature": 0}

    provider_name, model_id = model.split(":")
    provider = _load_provider(name=provider_name)
    request = CompletionRequest(
        model_id=model_id,
        messages=messages,
        system_instruction=system_instruction,
        output_schema=output_schema,
        generation_kwargs=generation_kwargs,
    )
    provider.get_response(request=request, stream=stream)

    raise NotImplementedError()
