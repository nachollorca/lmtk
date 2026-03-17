"""Implements the provider to use models hosted in Mistral API."""

import json
import time
import urllib.request
from collections.abc import Iterator

from lmtk.datatypes import CompletionRequest, CompletionResponse
from lmtk.provider import Provider

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralProvider(Provider):
    """Provider for models hosted on the Mistral API."""

    api_key_name = "MISTRAL_API_KEY"

    @classmethod
    def _get_response(cls, request: CompletionRequest, api_key: str) -> CompletionResponse:
        """Send a chat completion request to the Mistral API."""
        api_messages: list[dict] = []
        if request.system_instruction:
            api_messages.append({"role": "system", "content": request.system_instruction})
        api_messages.extend(m.to_dict() for m in request.messages)

        payload = {
            "model": request.model_id,
            "messages": api_messages,
            **(request.generation_kwargs or {}),
        }

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            MISTRAL_API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        start = time.perf_counter()
        with urllib.request.urlopen(req) as response:
            body = json.loads(response.read())
        latency = time.perf_counter() - start

        return CompletionResponse(
            content=body["choices"][0]["message"]["content"],
            input_tokens=body["usage"]["prompt_tokens"],
            output_tokens=body["usage"]["completion_tokens"],
            latency=latency,
        )

    @classmethod
    def _stream(cls, request: CompletionRequest, api_key: str) -> Iterator[str]:
        """Not yet implemented."""
        raise NotImplementedError
