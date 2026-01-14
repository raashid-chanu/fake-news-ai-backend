from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    text_vec = vectorizer.transform([data])
    result = model.predict(text_vec)[0]

    if result == 1:
        return jsonify({"result": "REAL NEWS"})
    else:
        return jsonify({"result": "FAKE NEWS"})

if __name__ == "__main__":
    app.run(port=5000)
