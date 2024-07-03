import os
import mlflow
import logging
import numpy as np
from flask.wrappers import Response
from flask import Flask, request, jsonify


logging.basicConfig(level=logging.INFO)

model_run_id = os.getenv("MLFLOW_MODEL_RUN_ID")
model = mlflow.sklearn.load_model("./model_for_prod/")
app = Flask(__name__)


def prepare_features(ride: dict) -> dict:
    features = {}
    features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features["trip_distance"] = ride["trip_distance"]
    return features


def get_prediction(features: dict) -> float:
    preds = model.predict(features)
    return float(preds[0])


@app.route("/predict", methods=["POST"])
def predict_endpoint() -> Response:
    logging.info("NY Taxi trip duration prediction app")
    ride = request.get_json()

    logging.info(f"Ride info: {ride}")
    features = prepare_features(ride)
    logging.info(f"Ride features: {features}")
    pred = get_prediction(features)

    dict_pred = {
        "duration": pred,
        "model_version": model_run_id,
    }
    logging.info(f"Response json: {dict_pred}")

    try:
        json_pred = jsonify(dict_pred)
    except TypeError as e:
        json_pred = jsonify({"error": str(e)})
    return json_pred


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=7860)
