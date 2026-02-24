set dotenv-load

sync:
    uv lock --upgrade
    uv sync --all-extras

reset-env:
    rm -rf .venv
    uv sync

install-hooks:
    prek install
    prek install --hook-type commit-msg

format:
    uv run ruff check --select I --fix .
    uv run ruff format .

test target="":
    uv run pytest --cov --cov-fail-under=90 {{target}}

check-types:
    uv run pyrefly check src

ipython:
    uv run ipython

analyze-complexity:
    uv run complexipy src

run file:
    uv run --env-file .env {{file}}
