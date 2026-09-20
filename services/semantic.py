print(">>> importing services", flush=True)
from sentence_transformers import CrossEncoder, SentenceTransformer
import re
import numpy as np

#This sii the model that I first used for the NLI task but it was not effective considering the use case here demands more tokens.
# nli_model = CrossEncoder(
#     "cross-encoder/nli-MiniLM2-L6-H768",
#     backend="onnx",
#     model_kwargs={
#         "file_name": "onnx/model_quint8_avx2.onnx"
#     }
# )

#removed coz it was too big to be deployed on render
# nli_model = CrossEncoder(
#     "cross-encoder/nli-deberta-v3-small"
# )
print(">>> services imported", flush=True)
nli_model = CrossEncoder("cross-encoder/nli-roberta-base")

sbert_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
print(">>> models loaded", flush=True)
LABELS = [
    "contradiction",
    "entailment",
    "neutral"
]

#This is for the google fact check part which is very less impactful in this project.
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

#Checks each concentrated premise and the atomic verification points extracted from the user inputted claim.
# Whether the concentrated premise entails, contradicts or is neutral to the atomic verification points is determined by the NLI model - deberta. 
def classify_tavily_results(search_results: list[dict], claim: str, queries: list[str]) -> list[dict]:
    classified_results = []
    verification_points = queries["verification_points"]
    for result in search_results:
        content = result["content"]
        point_results = []
        for point in verification_points:
            sbertcontent = sbert_search_result(content, point)
            scores = nli_model.predict([(sbertcontent, point)])
            contradiction_score = float(scores[0][0]) 
            entailment_score = float(scores[0][1]) 
            neutral_score = float(scores[0][2])
            label_index = scores[0].argmax() 
            label = LABELS[label_index]
            point_results.append({ "content": sbertcontent, "verification_point": point, "relationship": label, "score": float(scores[0][label_index]), "entailment_score": entailment_score, "contradiction_score": contradiction_score, "neutral_score": neutral_score })
        supporting_score = 0.0 
        contradicting_score = 0.0
        neutral_score = 0.0
        for point in point_results:
            if point["relationship"] == "entailment":
                supporting_score += point["entailment_score"]
            elif point["relationship"] == "contradiction":
                contradicting_score += point["contradiction_score"]
            else:
                neutral_score += point["neutral_score"]
        gap = supporting_score - contradicting_score
        supporting_score *= 2
        contradicting_score *= 2 
        if supporting_score > contradicting_score and supporting_score > neutral_score: 
            relationship = "entailment" 
            score = supporting_score
        elif contradicting_score > supporting_score and contradicting_score > neutral_score:
            relationship = "contradiction" 
            score = contradicting_score
        else:
            relationship = "neutral" 
            score = neutral_score
        classified_results.append({
            "result": result,
            "relationship": relationship,
            "score": score,
            "supporting_score": supporting_score,
            "neutral_score": neutral_score,
            "contradicting_score": contradicting_score,
            "gap": gap,
            "verification_points": point_results
        })
    return classified_results


#Returns the most relevant sentences from an article based on the atomic verification points extracted from the user inputted claim.
#This is achieved with a sbert model minilm which returns a passage of concatenated sentences that have high semantic similarity to the verification points. 
#This is done to reduce the noise in the article and to make the NLI model more effective in determining the relationship between the claim and the article.
def sbert_search_result(article: str, claim: str, top_k: int = 4):

    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', article)
        if s.strip()
    ]

    if not sentences:
        return ""

    if len(sentences) <= top_k:
        return article.strip()

    embeddings = sbert_model.encode(
        [claim] + sentences,
        normalize_embeddings=True
    )

    claim_embedding = embeddings[0]
    sentence_embeddings = embeddings[1:]

    similarities = sentence_embeddings @ claim_embedding

    # Get the most relevant sentences
    top_indices = np.argsort(similarities)[-top_k:]

    # Restore original article order
    top_indices = sorted(top_indices)

    return " ".join(sentences[i] for i in top_indices)



#This is a function that I made to test if the deberta small nli model is working on noisy data or concentrated premise and hypothesis.(Which it didnt)   
def justcheck():
    #THIS IS A TEST FUNCTION TO CHECK IF THE NLI MODEL IS WORKING PROPERLY. IT IS NOT USED IN THE MAIN APPLICATION.
#     hypothesis = [
#     "Hugging Face detected and contained the OpenAI agent.",
    
#     "The OpenAI agent hacked or accessed Hugging Face.",
    
#     "OpenAI described the incident as unprecedented."
# ]
#     tests = [
#     (
#         "The company behind ChatGPT said the startup Hugging Face "
#         "had detected and contained the agent.",
#         "Hugging Face detected and contained the OpenAI agent."
#     ),

#     (
#         "The agent then hacked Hugging Face, which is a database "
#         "of AI models, to locate technology that would help it "
#         "pass the hacking evaluation.",
#         "The OpenAI agent hacked or accessed Hugging Face."
#     ),

#     (
#         "\"We consider this incident to be an unprecedented "
#         "cyber-incident, involving state-of-the-art cyber "
#         "capabilities,\" OpenAI said.",
#         "OpenAI described the incident as unprecedented."
#     )
# ]

#     for premise, hypothesis in tests:

#         scores = nli_model.predict([(premise, hypothesis)])

#         label_index = scores[0].argmax()
#         label = LABELS[label_index]

#         print("\nPREMISE:", premise)
#         print("HYPOTHESIS:", hypothesis)
#         print(label, scores)
    claim = """
Hugging Face was hacked by an autonomous AI agent.
"""

    premise = """
The company behind ChatGPT said the startup Hugging Face had detected and contained the agent – an AI tool designed to carry out tasks without human assistance – which had entered its systems.\n\n“We consider this incident to be an unprecedented cyber-incident, involving state-of-the-art cyber capabilities,” OpenAI said. [...] # AI agent went rogue and hacked startup by itself, OpenAI reveals\n\nThis article is more than 1 month old\n\nCompany behind ChatGPT says agent ‘cheated’ an evaluation by attacking a Hugging Face database\n\nOpenAI has revealed that an autonomous AI agent powered by its technology went rogue during a test, accessed the open web and hacked a prominent startup by itself in an “unprecedented incident”. [...] The agent then hacked Hugging Face, which is a database of AI models, to locate technology that would help it pass the hacking evaluation, having “inferred” that Hugging Face might have the models, datasets and solutions for passing the test. OpenAI said the models “successfully found ways to gain access to secret information that it could use to cheat the evaluation”. The attack ended when Hugging Face’s security team and its own AI agents spotted and stopped the rogue activity."""

    # chunks = chunk_text(premise, chunk_size=2, overlap=1)
    # print(nli_model.model.config.id2label)
    # for chunk in chunks:
    #     print("CHUNK:", chunk)
    #     scores = nli_model.predict([(chunk, claim)])

    #     label_index = scores[0].argmax()
    #     label = LABELS[label_index]
    
    #     print(label, scores)
    # print("PREMISE:")
    print(nli_model.predict([(premise, claim)]))