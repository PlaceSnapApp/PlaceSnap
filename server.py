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

# Allow requests from Chrome extension and any origin (needed for browser extensions)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Anthropic client — key comes from environment variable set in Railway
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

EXTRACTION_PROMPT = """You are a geographic location extractor for travel videos and posts.

Platform: {platform}
URL: {url}
Content:
---
{text}
---

Extract ALL geographic locations mentioned or visited. Include:
- Countries, islands, cities, towns, villages
- Beaches, bays, coves (use local names when given)
- Mountains, volcanoes, lakes, waterfalls, forests
- Landmarks, temples, ports, markets, viewpoints
- Regions, neighborhoods, districts
- Restaurants, bars, hotels, museums (with their city/country)

Return ONLY valid JSON, no markdown, no explanation:
{{
  "locations": [
    {{
      "name": "Place Name, City, Country",
      "type": "restaurant|bar|museum|archaeological|nature|beach|island|village|city|hotel|library|market|transport|culture|region|country|landmark",
      "context": "one sentence: why/how it appears in this content",
      "lat": 12.345,
      "lng": 67.890
    }}
  ]
}}

Rules:
- Max 25 locations, deduplicate
- Always append city and country to name for disambiguation (e.g. "Trattoria da Mario, Florence, Italy")
- Use best-guess coordinates — do not use 0.0 unless truly unknown
- context must be specific to this content, not a generic description
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

    # Trim to avoid massive token usage
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
        # Strip markdown fences if present
        clean = raw.replace("```json", "").replace("```", "").strip()

        # Extract JSON object
        match = re.search(r"\{[\s\S]*\}", clean)
        if not match:
            raise ValueError("No JSON found in response")

        data = json.loads(match.group(0))
        locations = data.get("locations", [])

        return {"locations": locations, "count": len(locations)}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Claude response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
