import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Paths a los modelos
MODEL_WHITE = os.path.join(MODELS_DIR, "best_model_white_rf.pkl")
MODEL_RED = os.path.join(MODELS_DIR, "best_model_red_gb.pkl")

# Cargar modelos
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Error cargando el modelo {path}: {e}")

model_white = load_model(MODEL_WHITE)
model_red = load_model(MODEL_RED)

def predict_white(data: dict):
    try:
        X = [list(data.values())]
        return model_white.predict(X)[0]
    except Exception as e:
        raise Exception(f"Error en predicción white: {e}")

def predict_red(data: dict):
    try:
        X = [list(data.values())]
        return model_red.predict(X)[0]
    except Exception as e:
        raise Exception(f"Error en predicción red: {e}")
    
    
print("WHITE FEATURES:", model_white.feature_names_in_)
print("RED FEATURES:", model_red.feature_names_in_)
import numpy as np

print("WHITE CLASSES:", model_white.classes_)
print("RED CLASSES:", model_red.classes_)

# Probemos con datos random
random_test = np.random.rand(1, len(model_white.feature_names_in_))
print("WHITE RANDOM PRED:", model_white.predict(random_test))
print("RED RANDOM PRED:", model_red.predict(random_test))
