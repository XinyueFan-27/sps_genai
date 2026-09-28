# sps_genai

A FastAPI project with a bigram text generator and a word embedding endpoint (spaCy).

## Setup
uv sync
uv run python -m spacy download en_core_web_lg

## Run
uv run fastapi dev app/main.py

## Endpoints
- POST /generate  {"start_word": "bigram", "length": 5}
- POST /embedding {"word": "apple"}