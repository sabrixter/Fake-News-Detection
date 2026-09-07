import os
import json
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
import requests
from openai import OpenAI
from services.semantic import classify_tavily_results

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# openai_client = OpenAI(api_key=OPENAI_API_KEY)

async def google_fact_check(claim: str):
    url =  "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Google Fact Check API key is not configured",
        )

    params = {
        "key": GOOGLE_API_KEY,
        "query": claim,
        "languageCode": "en",
        "pageSize": 10,
    }

    response = requests.get(url, params=params, timeout=30)
    return response.json()


def extract_search_queries(passage: str) -> list[dict]:
    prompt = f"""
You are a fact-checking assistant.

Given the following passage, identify the factual claims
that could be verified using news articles.

For each claim, generate a concise search query containing
the important keywords.

Do not include opinions, questions, or vague statements.

Return ONLY valid JSON in this format:

{{
    "claims": [
        {{
            "claim": "...",
            "search_query": "..."
        }}
    ]
}}

Passage:
{passage}
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    result = json.loads(response.text)

    return result["claims"]





reputation_df = pd.read_csv("./news_source_reputation_all_categories.csv")
def get_reputation_score(domain: str, genre: int) -> float:

    matches = reputation_df[
        (reputation_df["domain"] == domain) &
        (reputation_df["category"] == genre)
    ]

    if len(matches) == 0:
        return 0.5

    return float(matches.iloc[0]["reputation_score"])



def fake_or_not(claim:str, search_results: list[dict], genre: int) -> dict:
    classified_results = classify_tavily_results(search_results, claim)
    supporting_score = 0.0
    contradicting_score = 0.0

    for item in classified_results:
        
        result = item["result"]

        domain = urlparse(result.get("url", "")).netloc.lower()
        domain = domain.removeprefix("www.")

        reputation_score = get_reputation_score(
            domain=domain,
            genre=genre
        )

        nli_score = item["score"]
        relationship = item["relationship"]

        weighted_score = nli_score * reputation_score

        if relationship == "entailment":
            supporting_score += weighted_score

        elif relationship == "contradiction":
            contradicting_score += weighted_score

        item["reputation_score"] = reputation_score
        item["weighted_score"] = weighted_score

    gap = supporting_score - contradicting_score

    if gap >= 2.0:
        verdict = "real news"

    elif gap >= 0.5:
        verdict = "likely true"

    elif gap <= -2.0:
        verdict = "fake"

    elif gap <= -0.5:
        verdict = "false rumours"

    else:
        verdict = "uncertain"

    return {
        "claim": claim,
        "genre": genre,
        "verdict": verdict,
        "supporting_score": supporting_score,
        "contradicting_score": contradicting_score,
        "gap": gap,
        "results": classified_results
    }