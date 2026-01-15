from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return "Fake News AI Backend is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    if "text" not in data:
        return jsonify({"error": "No text field found"}), 400

    text = data["text"]

    vect = vectorizer.transform([text])
    pred = model.predict(vect)[0]

    result = "Fake" if pred == 1 else "Real"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
