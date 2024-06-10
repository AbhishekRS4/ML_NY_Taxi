import os
import logging
import requests
import numpy as np


def send_post_request_to_web_service() -> None:
    logging.basicConfig(level=logging.INFO)

    ride = {
        "PULocationID": 25,
        "DOLocationID": 30,
        "trip_distance": 40,
    }

    # if the deployment is on local machine
    response = requests.post(
        "http://127.0.0.1:7860/predict",
        json=ride,
    )

    logging.info(response.json())
    return


def main() -> None:
    send_post_request_to_web_service()
    return


if __name__ == "__main__":
    main()
