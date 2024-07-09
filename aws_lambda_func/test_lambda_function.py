from lambda_function import prepare_features


def test_prepare_features() -> None:
    ride = {
        "PULocationID": 25,
        "DOLocationID": 30,
        "trip_distance": 40,
    }
    ride_features = prepare_features(ride)
    expected_ride_features = {
        "PU_DO": f"{ride['PULocationID']}_{ride['DOLocationID']}",
        "trip_distance": ride["trip_distance"],
    }
    assert ride_features == expected_ride_features
    return
