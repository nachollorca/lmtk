"""Contains the main logic to call language model APIs."""

from collections.abc import Iterator

from pydantic import BaseModel

from lmtk.datatypes import Message, ModelResponse
from lmtk.provider import Provider


def _get_provider(model: str) -> Provider:
    """Gets the appropriate Provider class for the given model string."""
    provider_name = model.split(":")
    # load the corresponding Provider
    raise NotImplementedError()


# TODO: think if this should be get_response or generate_response
def get_response(
    model: str,
    messages: list[Message],
    system_instruction: str | None = None,
    output_schema: type[BaseModel] | None = None,
    stream: bool = False,
    generation_kwargs: dict | None = None,
) -> ModelResponse | Iterator[str]:
    """Docstring."""
    if output_schema and stream:
        raise ValueError("Only `stream` or `output_schema` can be set, not both.")

    if generation_kwargs is None:
        generation_kwargs = {"temperature": 0}

    # get provider, attempt to call the get_response
    raise NotImplementedError()
