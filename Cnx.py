from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://pfa-lovat-omega.vercel.app"}})
MONGO_URI = "mongodb+srv://said:Lamine10@cluster0.ismu8ej.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tls=true&tlsAllowInvalidCertificates=false"

client = MongoClient(MONGO_URI)
db = client["LinkUp"]
users_collection = db["users"]
events_collection = db["events"]

@app.get("/recommend")
def recommend_events():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify([])

    user_doc = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user_doc:
        return jsonify([])

    favorites = user_doc.get('favorites', [])
    joined_events = user_doc.get('eventsJoined', [])
    posted_events = user_doc.get('eventsPosted', [])

    interacted_ids = [ObjectId(e) for e in (favorites + joined_events + posted_events)]
    interacted_events = list(events_collection.find({"_id": {"$in": interacted_ids}}))

    events = list(events_collection.find({}))
    events_to_recommend = [e for e in events if e["_id"] not in interacted_ids]

    if not interacted_events or not events_to_recommend:
        return jsonify([])

    interacted_docs = [f"{e.get('title','')} {e.get('description','')}" for e in interacted_events]
    candidate_docs = [f"{e.get('title','')} {e.get('description','')}" for e in events_to_recommend]

    all_docs = interacted_docs + candidate_docs
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(all_docs)

    interacted_vec = tfidf[:len(interacted_docs)]
    candidate_vec = tfidf[len(interacted_docs):]

    scores = cosine_similarity(candidate_vec, interacted_vec).mean(axis=1)

    for i, e in enumerate(events_to_recommend):
        e["score"] = float(scores[i])

    recommended = sorted(events_to_recommend, key=lambda x: x["score"], reverse=True)
    recommended = recommended[:4]  # take only top 4
    result = [{"id": str(e["_id"]), "title": e["title"], "description": e["description"], "score": round(e["score"], 3)}
            for e in recommended]


    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
