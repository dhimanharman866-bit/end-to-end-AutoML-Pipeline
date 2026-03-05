import joblib
def make_prediction(data):
    model=joblib.load("models/best_model.pkl")
    prediction=model.predict([list(data.values())])
    return {"prediction": prediction.tolist()}