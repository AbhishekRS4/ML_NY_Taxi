import os
import mlflow
import logging
import numpy as np
from flask.wrappers import Response
from flask import Flask, request, jsonify


from ny_taxi.config.config import ProductionConfig
from ny_taxi.utils.production import get_latest_registered_model


logging.basicConfig(level=logging.INFO)

"""
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
config_production = ProductionConfig(mlflow_tracking_uri=mlflow_tracking_uri)
logging.info(f"config_production: {config_production}")
mlflow.set_tracking_uri(config_production.mlflow_tracking_uri)
model_name, model_run_id, model_version = get_latest_registered_model(config_production)
model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
"""

model = mlflow.sklearn.load_model("./model_for_prod/")
# model_run_id = "temp"
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
        # 'model_version': model_run_id
    }
    logging.info(f"Response json: {dict_pred}")

    try:
        json_pred = jsonify(dict_pred)
    except TypeError as e:
        json_pred = jsonify({"error": str(e)})
    return json_pred


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=7860)
