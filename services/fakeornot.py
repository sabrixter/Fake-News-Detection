import os
import json
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
import requests
from services.semantic import classify_tavily_results

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

#checks for user verfied facts from google fact check api to filter out the obvious fake news. for example - "Earth is flat"
#that claim has been verified by general public as false.
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

#Essentially extracts the main claims and the atomic verification points from the user inputted claim. 
#This would have become complicated for deployment, therefore I used the gemini-3.5-flash-lite LLM.
def extract_search_queries(passage: str) -> list[dict]:
    prompt = f""" You are a fact-checking assistant. Given the following news article, identify the distinct factual claims that could be verified using reliable news articles or other credible sources. For each main claim, generate: 1. "claim" — a concise, self-contained version of the factual claim. 2. "search_query" — a concise search query containing the most important keywords needed to find evidence for that claim. In addition, examine the ENTIRE article as a whole and generate a single list of "verification_points". The verification_points are NOT divided under individual claims. They should collectively represent the important factual information in the entire article that needs to be verified. The purpose of verification_points is to provide simple, precise hypotheses that can be directly compared with retrieved source content using an NLI model. Rules for verification_points: - Extract only meaningful factual information that requires verification. - A verification point should represent ONE meaningful factual relationship, event, action, or assertion. - It is okay to break a complex statement into multiple verification points when a single point would require the evidence to establish several distinct facts simultaneously. - Do NOT mechanically split sentences or claims. - Do NOT create verification points merely because a sentence contains multiple words, entities, or grammatical components. - Keep each verification point SHORT and precise. - However, do not make it so short that it loses its factual meaning or necessary context. - Include the minimum context required for the statement to be independently understood. - Remove unnecessary wording, explanations, background information, qualifiers, repetition, and other noise. - Avoid redundancy. Two verification points should not express essentially the same fact. - Preserve important entities, actions, dates, and relationships when they are necessary to establish the fact. - Avoid vague references such as "it", "they", "this incident", "the company", or "the agent". Use the actual entity when possible. - Do not introduce facts that are not stated or clearly implied by the article. - Do not turn verification points into search queries. - Do not include opinions, speculation, questions, or predictions. - Do not create a fixed number of verification points. Create as many as necessary to cover the important factual information in the article without unnecessary fragmentation. - Prefer wording that a news article could naturally state in one sentence or a short passage. - The verification point should be simple enough that a relevant sentence or short passage from a source can directly entail or contradict it. - Optimize for semantic clarity and NLI recognition, not merely for grammatical simplicity. - If two facts are tightly connected and can naturally be established by the same sentence or short passage, they may remain together. - If verifying a point would require an NLI model to combine multiple independent facts from different parts of a source, split it. - A verification point should be neither a full restatement of a long claim nor an extremely small fragment of a fact. Important distinction: "claim" = the complete factual claim represented in the article. "search_query" = keywords used to retrieve relevant evidence. "verification_points" = short, precise, non-redundant factual propositions extracted from the ENTIRE article for NLI-based verification. Do not replace the main claims with verification_points. The verification_points are an additional representation of the article's factual content. Return ONLY valid JSON in exactly this format: {{ "claims": [ {{ "claim": "...", "search_query": "..." }} ], "verification_points": [ "...", "..." ] }} News article: {passage} """
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    result = json.loads(response.text)

    return result




#With the help of the csv - news_source_reputation_all_categories.csv, this function returns the reputation score of a news source based on its domain and the genre of the claim.
reputation_df = pd.read_csv("./news_source_reputation_all_categories.csv")
def get_reputation_score(domain: str, genre: int) -> float:

    matches = reputation_df[
        (reputation_df["domain"] == domain) &
        (reputation_df["category"] == genre)
    ]

    if len(matches) == 0:
        return 0.5

    return float(matches.iloc[0]["reputation_score"])



#Does the final calculation of the supporting score, contradicting score, neutral score and the final verdict.
# filters the domain name and gets the reputation score. and then => weighted score = nli score * reputation score. 
def fake_or_not(claim:str, search_results: list[dict], genre: int, queries: list[str]) -> dict:
    classified_results = classify_tavily_results(search_results, claim, queries)
    supporting_score = 0.0
    contradicting_score = 0.0
    neutral_score = 0.0

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
            supporting_score = weighted_score

        elif relationship == "contradiction":
            contradicting_score = weighted_score
        else:
            neutral_score = weighted_score
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
        "neutral_score": neutral_score,
        "contradicting_score": contradicting_score,
        "gap": gap,
        "results": classified_results
    }