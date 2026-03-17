"""Contains the main logic to call language model APIs."""

import importlib
from collections.abc import Iterator

from pydantic import BaseModel

from lmtk.datatypes import CompletionRequest, CompletionResponse, Message, UserMessage
from lmtk.errors import AllModelsFailedError
from lmtk.provider import Provider


def _load_provider(name: str) -> type[Provider]:
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


def get_response(
    model: str | list[str],
    messages: list[Message] | str,
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    stream: bool = False,
    generation_kwargs: dict | None = None,
) -> CompletionResponse | Iterator[str]:
    """Generate a response from a language model.

    Args:
        model: Provider-prefixed model identifier (e.g. ``"mistral:devstral-latest"``)
            or a list of identifiers to try in order as fallbacks.
        messages: The conversation history, or a plain string which is
            interpreted as a single user message.
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

    Raises:
        AllModelsFailedError: If every model in the list fails.
    """
    # early stop
    if output_schema and stream:
        raise ValueError("Only `stream` or `output_schema` can be set, not both.")

    # set defaults and normalize overloaded params
    models = [model] if isinstance(model, str) else model
    if isinstance(messages, str):
        messages = [UserMessage(content=messages)]
    if generation_kwargs is None:
        generation_kwargs = {"temperature": 0}

    # fallback loop
    errors: dict[str, Exception] = {}
    for m in models:
        provider_name, model_id = m.split(":")
        provider = _load_provider(name=provider_name)
        request = CompletionRequest(
            model_id=model_id,
            messages=messages,
            system_instruction=system_instruction,
            output_schema=output_schema,
            generation_kwargs=generation_kwargs,
        )
        try:
            # call provider and hopefully return response
            return provider.get_response(request=request, stream=stream)
        except Exception as exc:
            errors[m] = exc

    # raise paricular error if one model or error summary if many
    if len(errors) == 1:
        raise next(iter(errors.values()))
    raise AllModelsFailedError(errors)
