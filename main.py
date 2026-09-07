from fastapi import FastAPI

from services.fakeornot import extract_search_queries, google_fact_check, fake_or_not
from services.search import search
from services.search import predict
from services.semantic import classify_claim

app = FastAPI()

@app.post("/check")
async def check_news(claim: str):
    result = await google_fact_check(claim)
    queries = []
    matches = []
    search_results = []

    matches = classify_claim(claim, result.get("claims", []))

    queries = extract_search_queries(claim)
    genre = predict(claim)
    search_results = search(queries)
    score = fake_or_not(claim, search_results, genre)

    # if result is None:
    #     queries = extract_search_queries(claim)
    #     search_results = search(queries)
    # else:
    #     matches = classify_claim(claim, result.get("claims", []))

    return {"result": result, "queries": queries, "google fact check matches": matches, "score": score}