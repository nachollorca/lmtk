# lmtk

What it offers:
- **Simplest interface to call different Language Model APIs**
- Minimal dependencies: HTTP requests only, no third party packages
- Streaming
- Comfy structured outputs, **only if the provider / model supports it natively**
- Parallel calls
- Unified HTTP error handling
- Locations (for AWS Bedrock, GCP Vertex and Azure)
- Fallbacks
- Bring Your Own Token (for each provider)

What it does **NOT** offer:
- Tools / function calling / MCP
- Agents
- Multimodality (only text-in, text-out)
- Shady under-the-hood prompt modification (e.g. to force structured output)
- API gateways

If you are looking for a more constrained but out-of-the-box agent interface, I'd recommend [pydantic-ai](https://ai.pydantic.dev) or [haystack-ai](https://docs.haystack.deepset.ai/docs/generators).
If you are looking to keep granular control but extend on tools or multimodality, I'd recommend [litellm](https://docs.litellm.ai/docs/) or leveraging the OpenAI-compatible endpoints that providers normally set up.
If you want a unified a token for all providers and are willing to give away telemetry data, check Gateways like [openrouter](https://openrouter.ai).

## Install
`uv add lmtk`

## Basic usage
```python
from lmtk import get_response

model = "mistral:mistral-small-2603"
# supports locations as in "vertex:gemini-2.5-flash@europe-west4"

# single prompt
response = get_response(model=model, messages="Tell me a joke")

# multi-turn
messages = [
    UserMessage("My name is Alice."),
    AssistantMessage("Nice to meet you, Alice!"),
    UserMessage("What is my name?"),
]
response = get_response(model=model, messages=messages)

# system prompt / generation kwargs
response = get_response(
    model=model,
    messages="Hi!",
    system_instruction="Talk like a pirate",
    generation_kwargs={"temperature": 0.9, "max_tokens": 10}
)

# streaming
token_iter = get_response(model=model, messages="Count from 1 to 5.", stream=True)

# model fallbacks (first request will fail, second will work)
response = get_response(model=["mistral:nonexistent-model", model], messages="Hi")

# structured output -> response.parsed will have a Recipe instance
class Ingredient(BaseModel):
    name: str
    quantity: int
    unit: str = ""

class Recipe(BaseModel):
    ingredients: list[Ingredient]
response = get_response(model=model, messages="How do I make cheescake?", output_schema=Recipe)

# parallel calls
results = get_response_batch(model=model, messages_list=["Greet in english", "Saluda en espanyol."])
```

## Development

### Structure
```text
src/lmtk/
├── core.py         # Entry points: get_response, get_response_batch
├── datatypes.py    # Common message and response schemas
├── provider.py     # Base Provider class and registry
├── providers/      # Concrete implementations (Mistral, Vertex, etc.)
├── errors.py       # Unified HTTP and API error handling
└── utils.py        # Shared helper functions
```

### Tooling
We use `just` for development tasks. Use:
- `just sync`: Updates lockfile and syncs environment.
- `just format`: Lints and formats with `ruff`.
- `just check-types`: Static analysis with `ty`.
- `just analyze-complexity`: Cyclomatic complexity checks with `complexipy`.
- `just test`: Runs pytest with 90% coverage threshold.

### Contribute
1. **Hooks**: Install pre-commit hooks via `just install-hooks`. PRs will fail CI if linting/formatting is not applied.
2. **Issues**: Open an issue first using the default template.
3. **PRs**: Link your PR to the relevant issue using the PR template.

You can use `just validate <model>` (runs `example.py`) to verify which features run properly and which do not for a new provider / model.
Not all of them have to pass to open a PR: some providers do not even support native structured output.

## License
MIT
