# LinkedIn Post Generator

Generate voice-cloned LinkedIn posts from a topic, niche, and tone. Optionally provide past posts to match the author's exact writing style.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # add your OpenRouter API key
python app.py
```

API runs at `http://localhost:8000`. Docs at `/docs`.

## Usage

### Generate 5 posts (no voice cloning)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Why most SaaS founders fail at content marketing",
    "niche": "B2B SaaS founders",
    "tone": "direct and slightly contrarian"
  }'
```

### Generate with voice cloning

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Building a personal brand on LinkedIn",
    "niche": "tech professionals",
    "tone": "casual and motivational",
    "voice_samples": [
      "I quit my 6-figure job last year. Everyone thought I was crazy. Here'\''s what actually happened...",
      "3 things I learned from failing my first startup:\n\n1. Nobody cares about your idea\n2. Execution > vision\n3. Revenue fixes everything"
    ],
    "count": 5
  }'
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate LinkedIn posts |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

### POST `/generate`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `topic` | string | ✅ | What to write about |
| `niche` | string | ✅ | Target audience |
| `tone` | string | — | Writing tone (default: "professional but approachable") |
| `voice_samples` | string[] | — | 2+ past posts for voice cloning |
| `count` | int | — | Posts to generate (1-10, default 5) |

### Response

```json
{
  "posts": [
    {
      "hook": "The opening 1-2 lines",
      "body": "Full post with \\n line breaks",
      "format_type": "story | listicle | framework | contrarian | observation",
      "hook_type": "which hook pattern was used",
      "estimated_chars": 1200,
      "engagement_prediction": "why this post will get comments"
    }
  ],
  "voice_analyzed": true,
  "generation_time_ms": 8500,
  "model_used": "google/gemini-flash-1.5"
}
```

## Architecture

- **`prompts.py`** — Prompt engineering (the core product)
- **`app.py`** — FastAPI endpoint + LLM client + rate limiter
- Voice analysis extracts a "voice DNA" profile, then generation uses it as a style constraint
- Two-phase pipeline: analyze voice → generate posts (only if 2+ samples provided)
- OpenRouter for model access (swap models via env vars)
- In-memory per-IP rate limiting (20 req/min default, configurable via env)

## Deployment

Designed for RapidAPI or any hosting. Single process, no database, no dependencies beyond an API key.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIMAX_API_KEY` | — | **Required.** MiniMax API key |
| `MINIMAX_BASE_URL` | `https://api.minimax.io/anthropic` | MiniMax Anthropic-compatible endpoint |
| `GENERATION_MODEL` | `MiniMax-M2.5` | Model for post generation |
| `VOICE_MODEL` | `MiniMax-M2.5` | Model for voice analysis |
| `PUBLIC_URL` | `https://hetzner-vps.tail80b7e1.ts.net` | Public URL for OpenAPI spec |
| `RATE_LIMIT_REQUESTS` | `20` | Max requests per IP per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |
