print(">>> main.py started", flush=True)
from fastapi import FastAPI

from services.fakeornot import extract_search_queries, google_fact_check, fake_or_not
from services.search import search
from services.search import predict
from services.semantic import classify_claim, justcheck
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class ClaimRequest(BaseModel):
    claim: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/check")
async def check_news(request: ClaimRequest):
    claim = request.claim
    result = await google_fact_check(claim)
    queries = []
    matches = []
    search_results = []
    score = 0
    matches = classify_claim(claim, result.get("claims", []))

    queries = extract_search_queries(claim)
    print("queries",queries)
    genre = predict(claim)
    search_results = search(queries["claims"])
    
    score = fake_or_not(claim, search_results, genre, queries)

    return {"result": result, "queries": queries, "google fact check matches": matches, "score": score}


@app.get("/health") #THIS IS A TEST FUNCTION TO CHECK IF THE NLI MODEL IS WORKING PROPERLY. IT IS NOT USED IN THE MAIN APPLICATION.
async def health_check():
    justcheck()