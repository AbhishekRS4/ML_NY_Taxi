import os
import json
import boto3
import base64
import mlflow

# create a client for kinesis
kinesis_client = boto3.client("kinesis")

# get env variables
PRED_STREAM_NAME = os.getenv("PRED_STREAM_NAME", "ny-taxi-ride-duration-pred-event")
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID", "6cf8127aa523476692898b11f4e5d6d0")
TEST_RUN = os.getenv("TEST_RUN", "False") == "True"

# load the model from s3 bucket
logged_model_file = f"s3://mlops-mlflow-tracking/1/{MLFLOW_RUN_ID}/artifacts/xgboost/"
model = mlflow.pyfunc.load_model(logged_model_file)


def prepare_features(ride: dict) -> dict:
    features = {}
    features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features["trip_distance"] = ride["trip_distance"]
    return features


def get_prediction(features: dict) -> float:
    preds = model.predict(features)
    return float(preds[0])


def lambda_handler(event: dict, context) -> dict:
    prediction_events = []

    for record in event["Records"]:
        # load the encoded data from the stream
        encoded_data = record["kinesis"]["data"]

        # decode the encoded data
        decoded_data = base64.b64decode(encoded_data).decode("utf-8")

        # load the dict from the json
        ride_event = json.loads(decoded_data)

        ride = ride_event["ride"]
        ride_id = ride_event["ride_id"]

        # prepare features
        features = prepare_features(ride)

        # get prediction
        prediction = get_prediction(features)

        # create a prediction event
        prediction_event = {
            "model": 'ny_taxi_ride_duration_prediction_model',
            "version": MLFLOW_RUN_ID,
            "prediction": {
                "ride_duration": prediction,
                "ride_id": ride_id   
            }
        }

        # publish the prediction to the stream
        if not TEST_RUN:
            kinesis_client.put_record(
                StreamName=PRED_STREAM_NAME,
                Data=json.dumps(prediction_event),
                PartitionKey=str(ride_id)
            )
        
        # add the prediction event to a list of prediction events
        prediction_events.append(prediction_event)

    dict_pred = {
        "prediction": prediction_events
    }    
    return dict_pred
