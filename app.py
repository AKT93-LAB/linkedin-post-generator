"""
LinkedIn Post Generator API
Single endpoint, no database, prompt-engineered for quality.
"""

import json
import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from anthropic import AsyncAnthropic

from prompts import build_voice_analysis_prompt, build_generation_prompt, POST_GENERATION_SYSTEM

app = FastAPI(
    title="LinkedIn Post Generator",
    description=(
        "Generate high-quality, voice-cloned LinkedIn posts from a topic, target niche, and desired tone. "
        "Optionally provide 2+ past posts to clone the author's exact writing style. "
        "Returns 5 posts with varied hooks (contrarian, story, listicle, framework, observation) "
        "and engagement predictions. Built for LinkedIn creators, ghostwriters, and agencies."
    ),
    version="0.1.0",
    servers=[
        {"url": os.getenv("PUBLIC_URL", "https://hetzner-vps.tail80b7e1.ts.net"), "description": "Production"},
    ],
)

# --- Config ---

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
DEFAULT_MODEL = os.getenv("GENERATION_MODEL", "MiniMax-M2.5")
VOICE_MODEL = os.getenv("VOICE_MODEL", "MiniMax-M2.5")

_client = AsyncAnthropic(api_key=MINIMAX_API_KEY, base_url=MINIMAX_BASE_URL) if MINIMAX_API_KEY else None

# Rate limiting: max requests per IP per window
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# In-memory rate limiter
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> None:
    """Check and enforce per-IP rate limit. Raises HTTP 429 if exceeded."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    _rate_limit_store[client_ip] = [
        ts for ts in _rate_limit_store[client_ip] if ts > window_start
    ]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        retry_after = int(_rate_limit_store[client_ip][0] + RATE_LIMIT_WINDOW - now) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s.",
            headers={"Retry-After": str(retry_after)},
        )

    _rate_limit_store[client_ip].append(now)


# --- Request/Response Models ---

class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500, description="What to write about")
    niche: str = Field(..., min_length=2, max_length=200, description="Target audience/niche (e.g. 'B2B SaaS founders')")
    tone: str = Field(
        default="professional but approachable",
        max_length=200,
        description="Desired tone (e.g. 'casual and funny', 'authoritative', 'vulnerable and honest')"
    )
    voice_samples: Optional[list[str]] = Field(
        default=None,
        description="2+ of the user's past LinkedIn posts for voice cloning"
    )
    count: int = Field(default=5, ge=1, le=10, description="Number of posts to generate")


class Post(BaseModel):
    """A single generated LinkedIn post."""
    hook: str = Field(..., description="The opening 1-2 lines that stop the scroll")
    body: str = Field(..., description="Full post content with line breaks for paragraph separation")
    format_type: str = Field(..., description="Post structure: story, listicle, framework, contrarian, or observation")
    hook_type: str = Field(..., description="Hook pattern used: contrarian, bold_statement, story_opener, list_opener, etc.")
    estimated_chars: int = Field(..., description="Approximate character count of the body")
    engagement_prediction: str = Field(..., description="Why this post will drive comments and engagement")


class GenerateResponse(BaseModel):
    """Response containing generated LinkedIn posts."""
    posts: list[Post] = Field(..., description="Array of generated posts")
    voice_analyzed: bool = Field(..., description="Whether voice samples were analyzed for style cloning")
    generation_time_ms: int = Field(..., description="Total generation time in milliseconds")
    model_used: str = Field(..., description="LLM model used for generation")


# --- LLM Client ---

async def call_llm(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> str:
    """Call MiniMax via Anthropic-compatible API."""
    if not _client:
        raise HTTPException(status_code=500, detail="MINIMAX_API_KEY not configured")

    resp = await _client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # MiniMax-M2.5 returns ThinkingBlock + TextBlock; grab the text one
    for block in resp.content:
        if hasattr(block, "text"):
            return block.text
    return resp.content[-1].text


# --- Voice Analysis ---

async def analyze_voice(samples: list[str]) -> dict:
    """Analyze sample posts to extract voice DNA."""
    prompt = build_voice_analysis_prompt(samples)
    raw = await call_llm(
        system="You are a precise writing style analyst. Return only valid JSON.",
        user=prompt,
        model=VOICE_MODEL,
        temperature=0.2,  # Low temp for consistent analysis
        max_tokens=4096,
    )

    # Parse JSON from response (handle markdown code blocks)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: return empty DNA, posts will use defaults
        return {}


# --- Post Generation ---

async def generate_posts(
    topic: str,
    niche: str,
    tone: str,
    voice_dna: Optional[dict] = None,
    count: int = 5,
) -> list[Post]:
    """Generate LinkedIn posts."""
    user_prompt = build_generation_prompt(
        topic=topic,
        niche=niche,
        tone=tone,
        voice_dna=voice_dna,
        count=count,
    )

    raw = await call_llm(
        system=POST_GENERATION_SYSTEM,
        user=user_prompt,
        model=DEFAULT_MODEL,
        temperature=0.85,  # Higher temp for creative variety
        max_tokens=8192,
    )

    # Parse JSON from response
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Handle ```json ... ``` wrapping
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        posts_data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to recover truncated JSON — close open brackets/braces/quotes
        fixed = cleaned
        # Count open vs close brackets
        for open_c, close_c in [('{', '}'), ('[', ']')]:
            diff = fixed.count(open_c) - fixed.count(close_c)
            if diff > 0:
                fixed = fixed + (close_c * diff)
        # Close any trailing unclosed string
        if fixed.count('"') % 2 != 0:
            fixed = fixed + '"'
        try:
            posts_data = json.loads(fixed)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to parse LLM response as JSON. Raw response: {raw[:1000]}"
            )

    posts = []
    for item in posts_data[:count]:
        posts.append(Post(
            hook=item.get("hook", ""),
            body=item.get("body", ""),
            format_type=item.get("format_type", "observation"),
            hook_type=item.get("hook_type", "unknown"),
            estimated_chars=item.get("estimated_chars", len(item.get("body", ""))),
            engagement_prediction=item.get("engagement_prediction", ""),
        ))

    return posts


# --- Endpoints ---

@app.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate LinkedIn posts",
    description=(
        "Generate 1-10 LinkedIn posts about a given topic for a specific audience niche. "
        "Optionally provide 2+ past posts as voice_samples to clone the author's writing style. "
        "Returns posts with hooks, full body text, format type, and engagement predictions."
    ),
    tags=["Posts"],
)
async def generate(req: GenerateRequest, request: Request):
    """
    Generate LinkedIn posts.

    - **topic**: What to write about (3-500 chars)
    - **niche**: Target audience (e.g. "B2B SaaS founders", "marketers", "devs")
    - **tone**: Writing tone (default: professional but approachable)
    - **voice_samples**: Optional array of 2+ past posts for voice cloning
    - **count**: Number of posts to generate (1-10, default 5)
    """
    # Rate limit by IP
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    start = time.time()

    # Step 1: Analyze voice if samples provided
    voice_dna = None
    voice_analyzed = False

    if req.voice_samples and len(req.voice_samples) >= 2:
        try:
            voice_dna = await analyze_voice(req.voice_samples)
            voice_analyzed = bool(voice_dna)
        except Exception:
            voice_dna = None  # Graceful fallback — generate without voice cloning

    # Step 2: Generate posts
    posts = await generate_posts(
        topic=req.topic,
        niche=req.niche,
        tone=req.tone,
        voice_dna=voice_dna,
        count=req.count,
    )

    elapsed_ms = int((time.time() - start) * 1000)

    return GenerateResponse(
        posts=posts,
        voice_analyzed=voice_analyzed,
        generation_time_ms=elapsed_ms,
        model_used=DEFAULT_MODEL,
    )


@app.get("/health", summary="Health check", tags=["System"])
async def health():
    """Check if the API is running and which model is configured."""
    return {"status": "ok", "model": DEFAULT_MODEL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
