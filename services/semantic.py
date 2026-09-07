from sentence_transformers import CrossEncoder

# nli_model = CrossEncoder(
#     "cross-encoder/nli-MiniLM2-L6-H768",
#     backend="onnx"
# )
nli_model = CrossEncoder(
    "cross-encoder/nli-MiniLM2-L6-H768",
    backend="onnx",
    model_kwargs={
        "file_name": "onnx/model_quint8_avx2.onnx"
    }
)

LABELS = [
    "contradiction",
    "entailment",
    "neutral"
]


def classify_claim(claim: str, results: list[dict]) -> dict:
    matches = []

    for result in results:
        title = result["claimReview"][0]["title"]

        scores = nli_model.predict([
            (title, claim)
        ])

        label_index = scores[0].argmax()
        label = LABELS[label_index]
        score = float(scores[0][label_index])

        if label in ["entailment", "contradiction"]:
            matches.append({
                "result": result,
                "relationship": label,
                "score": score
            })

    return {
        "matches": matches
    }

def classify_tavily_results(search_results: list[dict], claim: str) -> list[dict]:
    classified_results = []

    for result in search_results:
        content = result["content"] + " " + result["title"]

        scores = nli_model.predict([
            (content, claim)
        ])

        label_index = scores[0].argmax()
        label = LABELS[label_index]
        score = float(scores[0][label_index])

        
        classified_results.append({
            "result": result,
            "relationship": label,
            "score": score
        })

    return classified_results