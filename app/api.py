from flask import Flask, request, jsonify
from models_loader import model_white, model_red

app = Flask(__name__)

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

@app.route("/predict/white", methods=["POST"])
def predict_white():
    data = request.json
    try:
        X = [[data[f] for f in FEATURES]]
        pred = model_white.predict(X)[0]
        return jsonify({"prediction": int(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/red", methods=["POST"])
def predict_red():
    data = request.json
    try:
        X = [[data[f] for f in FEATURES]]
        pred = model_red.predict(X)[0]
        return jsonify({"prediction": int(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
