"""
LinkedIn Post Generator API
Single endpoint, no database, prompt-engineered for quality.
"""

import json
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx

from prompts import build_voice_analysis_prompt, build_generation_prompt, POST_GENERATION_SYSTEM

app = FastAPI(
    title="LinkedIn Post Generator",
    description="Generate voice-cloned LinkedIn posts from topic + niche + tone",
    version="0.1.0",
)

# --- Config ---

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("GENERATION_MODEL", "anthropic/claude-sonnet-4")
VOICE_MODEL = os.getenv("VOICE_MODEL", "anthropic/claude-sonnet-4")


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
        description="3-5 of the user's past LinkedIn posts for voice cloning"
    )
    count: int = Field(default=5, ge=1, le=10, description="Number of posts to generate")


class Post(BaseModel):
    hook: str
    body: str
    format_type: str
    hook_type: str
    estimated_chars: int
    engagement_prediction: str


class GenerateResponse(BaseModel):
    posts: list[Post]
    voice_analyzed: bool
    generation_time_ms: int
    model_used: str


# --- LLM Client ---

async def call_llm(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> str:
    """Call OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"LLM API error: {resp.status_code} — {resp.text[:500]}"
        )

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# --- Voice Analysis ---

async def analyze_voice(samples: list[str]) -> dict:
    """Analyze sample posts to extract voice DNA."""
    prompt = build_voice_analysis_prompt(samples)
    raw = await call_llm(
        system="You are a precise writing style analyst. Return only valid JSON.",
        user=prompt,
        model=VOICE_MODEL,
        temperature=0.2,  # Low temp for consistent analysis
        max_tokens=2048,
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
        max_tokens=4096,
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

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Generate LinkedIn posts.

    - **topic**: What to write about
    - **niche**: Target audience (e.g. "B2B SaaS founders", "marketers", "devs")
    - **tone**: Writing tone (default: professional but approachable)
    - **voice_samples**: Optional array of past posts for voice cloning
    - **count**: Number of posts (1-10, default 5)
    """
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


@app.get("/health")
async def health():
    return {"status": "ok", "model": DEFAULT_MODEL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
