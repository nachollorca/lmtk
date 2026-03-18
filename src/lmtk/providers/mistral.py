"""Implements the provider to use models hosted in Mistral API."""

import json
import time
from collections.abc import Iterator

from lmtk.datatypes import CompletionRequest, CompletionResponse
from lmtk.provider import Provider

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralProvider(Provider):
    """Provider for models hosted on the Mistral API."""

    api_key_name = "MISTRAL_API_KEY"

    @classmethod
    def _build_messages(cls, request: CompletionRequest) -> list[dict]:
        """Build the API messages list from a CompletionRequest."""
        api_messages: list[dict] = []
        if request.system_instruction:
            api_messages.append({"role": "system", "content": request.system_instruction})
        api_messages.extend(m.to_dict() for m in request.messages)
        return api_messages

    @classmethod
    def _get_full_response(cls, request: CompletionRequest, api_key: str) -> CompletionResponse:
        payload: dict = {
            "model": request.model_id,
            "messages": cls._build_messages(request),
            **(request.generation_kwargs or {}),
        }

        if request.output_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema.__name__,
                    "schema": request.output_schema.model_json_schema(),
                },
            }

        start = time.perf_counter()
        response = cls._make_request(
            MISTRAL_API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        latency = time.perf_counter() - start

        body = response.json()
        content = body["choices"][0]["message"]["content"]

        parsed = None
        if request.output_schema:
            parsed = request.output_schema.model_validate_json(content)

        return CompletionResponse(
            content=content,
            input_tokens=body["usage"]["prompt_tokens"],
            output_tokens=body["usage"]["completion_tokens"],
            latency=latency,
            parsed=parsed,
        )

    @classmethod
    def _stream_response(cls, request: CompletionRequest, api_key: str) -> Iterator[str]:
        payload: dict = {
            "model": request.model_id,
            "messages": cls._build_messages(request),
            "stream": True,
            **(request.generation_kwargs or {}),
        }

        response = cls._make_request(
            MISTRAL_API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            stream=True,
        )

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices", [])
            if choices:
                token = choices[0].get("delta", {}).get("content", "")
                if token:
                    yield token
