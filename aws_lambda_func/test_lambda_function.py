from lambda_function import prepare_features


def test_prepare_features(tmpdir) -> None:
    # create a temp dir for model
    tmpdir.mkdir("model_for_prod")
    
    ride = {
        "PULocationID": 25,
        "DOLocationID": 30,
        "trip_distance": 40,
    }
    ride_features = prepare_features(ride)

    # expected features
    expected_ride_features = {
        "PU_DO": "25_30",
        "trip_distance": 40,
    }
    assert ride_features == expected_ride_features
    return
