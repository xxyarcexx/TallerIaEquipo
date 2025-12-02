from flask import Flask, request, jsonify, render_template  
from app.models_loader import model_white, model_red


app = Flask(__name__)

FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=FEATURES)

@app.route("/predict/white", methods=["POST"])
def predict_white():
    data = request.json
    try:
        # Construir X en el orden correcto de FEATURES
        X = [[data[f] for f in FEATURES]]
        
        # Predicción de clase (0 o 1)
        pred = model_white.predict(X)[0]
        
        # Probabilidad de que sea vino "bueno" (clase 1)
        if hasattr(model_white, "predict_proba"):
            proba = model_white.predict_proba(X)[0][1]
        else:
            proba = None

        result = {
            "prediction": int(pred)
        }
        if proba is not None:
            result["prob_good_wine"] = float(proba)

        return jsonify({
            "wine_type": "white",
            "results": [result]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route("/predict/red", methods=["POST"])
def predict_red():
    data = request.json
    try:
        # Construir X en el orden correcto de FEATURES
        X = [[data[f] for f in FEATURES]]
        
        # Predicción de clase (0 o 1)
        pred = model_red.predict(X)[0]
        
        # Probabilidad de que sea vino "bueno" (clase 1)
        if hasattr(model_red, "predict_proba"):
            proba = model_red.predict_proba(X)[0][1]
        else:
            proba = None

        result = {
            "prediction": int(pred)
        }
        if proba is not None:
            result["prob_good_wine"] = float(proba)

        return jsonify({
            "wine_type": "red",
            "results": [result]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True)
