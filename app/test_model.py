import pickle

with open("models/best_model_white_rf.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))
print(model.classes_)
