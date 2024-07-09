import os
import logging
import lambda_function


def main() -> None:
    os.environ["MLFLOW_MODEL_RUN_ID"] = "8ea4b231f2f04a80a8d0105ab82afb15"
    logging.basicConfig(level=logging.INFO)
    ride = {
        "PULocationID": 25,
        "DOLocationID": 30,
        "trip_distance": 40,
    }

    response = lambda_function.lambda_handler(ride, None)
    logging.info(response)
    return


main()
