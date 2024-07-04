import os
import logging
import requests


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    url = "[paste-the-AWS-invoke-API-here]/predict"

    ride = {
        "PULocationID": 25,
        "DOLocationID": 30,
        "trip_distance": 40,
    }

    result = requests.post(url, json=ride).json()
    logging.info(result)
    return


main()
