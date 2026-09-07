import os
from tavily import TavilyClient
import joblib

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


def search(queries: list[str]) -> list[dict]:
    results = []

    for query in queries:

        response = tavily_client.search(
            query=query.get("search_query"),
            search_depth="basic",
            max_results=10
        )

        for result in response["results"]:
            results.append({
                "query": query,
                "title": result["title"],
                "url": result["url"],
                "content": result["content"]
            })

    return results


model = joblib.load("./ml/logistic_regression_model.pkl")
vectorizer = joblib.load("./ml/tfidf_vectorizer.pkl")


def predict(claim: str):
    X = vectorizer.transform([claim])
    prediction = model.predict(X)[0]
    print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",prediction)
    return int(prediction)