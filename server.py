"""
PlaceSnap Backend Server
========================
Receives page content from the Chrome extension,
calls Claude to extract locations, returns JSON.
POST /extract
  body: { "text": "...", "url": "...", "platform": "youtube|instagram|tiktok" }
  returns: { "locations": [...] }
"""
import os
import re
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

app = FastAPI(title="PlaceSnap API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

EXTRACTION_PROMPT = """You are an information extraction system.

INPUT TEXT:
{text}

CRITICAL RULES:
1. Extract ONLY locations that appear as explicit text in the INPUT TEXT above.
2. NEVER infer locations from landmarks, activities, travel context, hashtags, account names, brands, or your world knowledge.
3. If a location is not explicitly written in the input, DO NOT output it.
4. Every extracted location MUST include the exact source text that supports it as "evidence".
5. If no explicit location text exists, return {{"locations":[]}}
6. Do NOT normalize or expand — if input says "Barcelona", do not output "Barcelona, Spain".
7. Do NOT convert partial names — if input says "Etna", do not output "Mount Etna, Italy".
8. If evidence cannot be quoted verbatim from the input, exclude the location.

Return ONLY valid JSON, no markdown, no explanation:
{{
  "locations": [
    {{
      "name": "exact location text found in input",
      "evidence": "exact quoted words from input that prove this location",
      "type": "restaurant|bar|museum|archaeological|nature|beach|island|village|city|hotel|library|market|transport|culture|region|country|landmark",
      "context": "one sentence: why/how it appears in this content"
    }}
  ]
}}

Max 25 locations, deduplicate.
"""

class ExtractRequest(BaseModel):
    text: str
    url: str = ""
    platform: str = "unknown"

@app.get("/")
def root():
    return {"status": "PlaceSnap API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/extract")
async def extract(req: ExtractRequest):
    if not req.text or len(req.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text content too short or empty")

    text = req.text[:8000]

    prompt = EXTRACTION_PROMPT.format(
        platform=req.platform,
        url=req.url,
        text=text
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text
        clean = raw.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{[\s\S]*\}", clean)
        if not match:
            raise ValueError("No JSON found in response")

        data = json.loads(match.group(0))
        locations = data.get("locations", [])

        # Evidence filter: reject any location whose evidence is not in the source text
        source_lower = text.lower()
        verified = [
            loc for loc in locations
            if loc.get("evidence") and loc["evidence"].lower() in source_lower
        ]

        return {"locations": verified, "count": len(verified)}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Claude response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
