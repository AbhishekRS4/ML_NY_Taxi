import os
import mlflow

model_run_id = os.getenv("MLFLOW_MODEL_RUN_ID")
model = mlflow.sklearn.load_model("./model_for_prod/")


def prepare_features(ride: dict) -> dict:
    features = {}
    features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features["trip_distance"] = ride["trip_distance"]
    return features


def get_prediction(features: dict) -> float:
    preds = model.predict(features)
    return float(preds[0])


def lambda_handler(event: dict, context) -> dict:
    ride_feats = prepare_features(event)
    pred = get_prediction(ride_feats)
    dict_pred = {
        "duration": pred,
        "model_version": model_run_id,
    }
    return dict_pred
