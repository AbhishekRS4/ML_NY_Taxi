import pytest

from web_service_app import prepare_features


@pytest.fixture()
def setup_temp_dir(tmpdir):
    # create a temp dir for model
    tmpdir.mkdir("./model_for_prod")
    return


def test_prepare_features(setup_temp_dir) -> None:
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
