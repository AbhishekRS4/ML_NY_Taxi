import os
import logging
import requests


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    url = "http://localhost:8080/2015-03-31/functions/function/invocations"
    ride = {
        "PULocationID": 25,
        "DOLocationID": 30,
        "trip_distance": 40,
    }

    result = requests.post(url, json=ride).json()
    logging.info(result)
    return


main()
