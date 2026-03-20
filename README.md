# lmtk

What it offers:
- **Simple interface to call different Language Model APIs**
- Minimal dependencies: HTTP requests only, no third party packages
- Streaming
- Comfy structured outputs, **only if the provider / model supports it natively**
- Parallel calls
- Unified HTTP error handling
- Locations (for AWS Bedrock, GCP Vertex and Azure)
- Fallbacks

What it does **NOT** offer:
- Tools / function calling
- Agents or agentic workflows
- Multimodality
- Shady under-the-hood prompt modification (e.g. to force structured output)

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
Here add a directory tree with the src/lmtk folder and a brief explanation of what each module does

### Tooling
Briefly explain the justfile, pre-commit hook and .github/workflows/verify.yaml

### Contribute
Brierfly explain the .github/ISSUE_TEMPLATE/default-issue.yaml and the .github/pull_request_template.md. Also explain that you can use example.py as a checklist to see if what features for a provider/model work.
