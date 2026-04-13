import joblib
import numpy as np

model = joblib.load("models/model.pkl")


def predict_game(features):
    X = np.array([features])
    return model.predict_proba(X)[0][1]