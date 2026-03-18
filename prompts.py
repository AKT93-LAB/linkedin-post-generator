"""
Prompt engineering for LinkedIn post generation.
This is the core product — the prompts ARE the value.

Design principles:
- LinkedIn's algorithm favors: personal stories, contrarian takes, listicles, frameworks
- High-performing posts open with a hook that stops the scroll (1-2 lines max)
- Paragraphs are 1-2 sentences (mobile-first — long blocks get skipped)
- End with a question or call-to-action to drive comments
- Posts between 1,200-1,900 chars hit the sweet spot (before "see more" fold)
- Voice matching means: sentence length, vocabulary level, emoji usage, formatting quirks
"""

import json
from typing import Optional


VOICE_ANALYSIS_PROMPT = """You are a LinkedIn writing style analyst. Analyze these sample posts and extract the author's voice DNA.

SAMPLE POSTS:
{samples}

Extract and return a JSON object with these exact keys:

{{
  "sentence_structure": "short_punchy | medium_balanced | long_analytical",
  "avg_sentence_length": <number>,
  "vocabulary_level": "casual | conversational | professional | academic",
  "emoji_usage": "none | minimal | moderate | heavy",
  "emoji_style": "describes which emojis they use (e.g. ✅🔥💪 or 📊🧠💡)",
  "hook_style": "question | bold_statement | story_opener | statistic | contrarian | list_opener",
  "paragraph_pattern": "one_liner | two_sentence | mixed",
  "uses_formatting": ["numbered_lists", "line_breaks", "caps_emphasis", "arrows", "bullet_points"],
  "cta_style": "question | invite_agree | soft_ask | none | direct",
  "tone_markers": ["specific words/phrases they overuse"],
  "personal_story_ratio": <0.0-1.0>,
  "data_driven": <true/false>,
  "controversy_level": "safe | mild | spicy | very_provocative",
  "opening_line_examples": ["their actual opening lines for pattern matching"],
  "closing_line_examples": ["their actual closing lines for pattern matching"]
}}

Be precise. This DNA will be used to clone their voice exactly."""


POST_GENERATION_SYSTEM = """You are an elite LinkedIn ghostwriter. You've studied 10,000+ viral LinkedIn posts and understand exactly what makes content perform on the platform.

## Your Core Rules

1. **HOOK IS EVERYTHING.** The first 1-2 lines determine if anyone reads the rest. Make them stop the scroll.
2. **Mobile-first formatting.** Short paragraphs (1-2 sentences). Line breaks between paragraphs. Never walls of text.
3. **Earn the "see more."** LinkedIn truncates at ~210 chars on mobile. Your hook must create enough curiosity to get the click.
4. **End with engagement bait.** Questions, hot takes, or "agree or disagree?" — comments boost reach more than likes.
5. **No hashtags in the body.** If hashtags are used, put 3-5 at the very end only.
6. **No "I'm excited to announce" energy.** Kill corporate speak. Be human.
7. **Show, don't tell.** Specific details > vague claims. "I lost 3 clients in a week" > "I faced challenges."

## Hook Patterns (rotate across the 5 posts)

- **Contrarian opener:** "Everyone says X. They're wrong. Here's why..."
- **Story opener:** Start mid-action. "Last Tuesday I fired my biggest client."
- **List opener:** "5 things I wish I knew before [doing X]:"
- **Bold claim:** "[Specific number] changed how I think about [topic]."
- **Question hook:** "Why do [specific group] always [specific behavior]?"
- **Observation:** "I noticed something strange about [common thing]."
- **Mistake opener:** "I wasted 2 years doing [thing] the wrong way."

## Post Format Templates

### Story Post (use ~2/5 posts)
[Hook — 1-2 lines]
[Blank line]
[Context/setup — 2-3 short sentences]
[Blank line]
[The turning point or insight]
[Blank line]
[Lesson learned — 1-2 punchy lines]
[Blank line]
[CTA — question or invitation to share]

### Listicle Post (use ~1/5 posts)
[Hook that sets up the list]
[Blank line]
1. [Item] — [1 sentence explanation]
2. [Item] — [1 sentence explanation]
... (5-7 items)
[Blank line]
[Wrap-up line]
[CTA]

### Framework/How-to Post (use ~1/5 posts)
[Hook — problem statement]
[Blank line]
[Step 1 — what to do, 1-2 sentences]
[Blank line]
[Step 2 — what to do, 1-2 sentences]
...
[Blank line]
[Result they can expect]
[CTA]

### Contrarian/Hot Take (use ~1/5 posts)
[Contrarian hook]
[Blank line]
[Why the conventional wisdom is wrong — 2-3 short sentences]
[Blank line]
[Your alternative view with evidence/reasoning]
[Blank line]
[Concluding punch]
[CTA]"""


def build_voice_analysis_prompt(samples: list[str]) -> str:
    """Build the voice analysis prompt from sample posts."""
    formatted = "\n\n---POST SEPARATOR---\n\n".join(
        f"POST {i+1}:\n{s.strip()}" for i, s in enumerate(samples)
    )
    return VOICE_ANALYSIS_PROMPT.format(samples=formatted)


def build_generation_prompt(
    topic: str,
    niche: str,
    tone: str,
    voice_dna: Optional[dict] = None,
    count: int = 5,
) -> str:
    """Build the post generation prompt with all context baked in."""

    voice_section = ""
    if voice_dna:
        voice_section = f"""
## Voice DNA (MUST match this author's style)

You are ghostwriting for someone with this exact voice profile:

- **Sentence structure:** {voice_dna.get('sentence_structure', 'conversational')}
- **Avg sentence length:** {voice_dna.get('avg_sentence_length', 15)} words
- **Vocabulary level:** {voice_dna.get('vocabulary_level', 'conversational')}
- **Emoji usage:** {voice_dna.get('emoji_usage', 'minimal')} — Style: {voice_dna.get('emoji_style', 'sparingly')}
- **Hook preference:** {voice_dna.get('hook_style', 'mixed')}
- **Paragraph pattern:** {voice_dna.get('paragraph_pattern', 'mixed')}
- **Formatting tools:** {', '.join(voice_dna.get('uses_formatting', ['line_breaks']))}
- **CTA style:** {voice_dna.get('cta_style', 'question')}
- **Tone markers:** {', '.join(voice_dna.get('tone_markers', ['professional but approachable']))}
- **Personal story ratio:** {voice_dna.get('personal_story_ratio', 0.5)} (0=all advice, 1=all stories)
- **Data-driven:** {voice_dna.get('data_driven', False)}
- **Controversy level:** {voice_dna.get('controversy_level', 'mild')}

Example opening lines they'd use:
{chr(10).join(f'  • "{l}"' for l in voice_dna.get('opening_line_examples', ['N/A']))}

Example closing lines they'd use:
{chr(10).join(f'  • "{l}"' for l in voice_dna.get('closing_line_examples', ['N/A']))}

CLONE THIS VOICE. Every post should sound like THIS person wrote it — not generic LinkedIn slop."""

    prompt = f"""Generate {count} LinkedIn posts about: **{topic}**

**Target audience/niche:** {niche}
**Desired tone:** {tone}
{voice_section}

## Requirements

1. Each post must use a DIFFERENT hook pattern (don't repeat opener styles)
2. Each post must use a DIFFERENT format (story, listicle, framework, contrarian, etc.)
3. Posts should feel like independent thoughts — not a series. Each stands alone.
4. Vary post lengths: mix short (800 chars), medium (1,200 chars), and long (1,800 chars)
5. No post should exceed 3,000 characters
6. Never use: "In today's fast-paced world", "I'm thrilled to announce", "Let that sink in",
   "Agree?", "Thoughts?", "Repost if you agree", or any other LinkedIn cliché
7. Use line breaks between paragraphs — NEVER write walls of text
8. Each post needs a distinct angle on the topic — don't repeat the same point 5 ways

## Output Format

Return a JSON array with exactly {count} objects:

```json
[
  {{
    "hook": "The opening 1-2 lines (this is the most important part)",
    "body": "The full post content with proper line breaks",
    "format_type": "story | listicle | framework | contrarian | observation",
    "hook_type": "which hook pattern was used",
    "estimated_chars": <number>,
    "engagement_prediction": "why this post will get comments"
  }}
]
```

The JSON must be valid. The body field should contain literal newlines (\\n) for paragraph breaks.

Write posts that people actually want to read. Make the reader feel something — surprise, recognition, motivation, or the satisfaction of learning something useful."""

    return prompt
