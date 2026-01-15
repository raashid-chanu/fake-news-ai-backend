from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return "Fake News AI Backend is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    vect = vectorizer.transform([data])
    prediction = model.predict(vect)[0]

    return jsonify({
        "prediction": "Fake" if prediction == 0 else "Real"
    })

if __name__ == "__main__":
    app.run()
