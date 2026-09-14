import os
from tavily import TavilyClient
import joblib

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

#Uses the tavily search api to get the most relevant articles with advanced search dept feature.
def search(queries: list[str]) -> list[dict]:
    results = []

    for query in queries:

        response = tavily_client.search(
            query=query.get("search_query"),
            search_depth="advanced",
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

#This model is created to determine the genre of the claim. It is created by the .ipynb file in the ml folder. 
model = joblib.load("./ml/logistic_regression_model.pkl")
#Also trained in the .ipynb file in the ml folder is a tfidf vectorizer which is used to transform the claim into a vector that can be used by the logistic regression model.
vectorizer = joblib.load("./ml/tfidf_vectorizer.pkl")

#predicts the genre with the logistic regression model.
def predict(claim: str):
    X = vectorizer.transform([claim])
    prediction = model.predict(X)[0]
    return int(prediction)