# Flask web service application


## For running the Flask app outside docker container
* For running the Flask app locally, set the environment variable using the following command with the correct value for tracking URI
```
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
```
* Use the following command to run the Flask app
```
cd flask_web_service
gunicorn --bind 0.0.0.0:7860 web_service_app:app
```
* For testing the POST request to the flask web service application [web_service_app.py](web_service_app.py), run the post request to web service script [post_req_web_service.py](post_req_web_service.py)


## For running the Flask app in a docker container
* To build the docker container, run the following command
```
cd flask_web_service
docker build -t flask_ny_taxi .
```
* To run the app, use the following command
```
docker run -p 7860:7860 -t flask_ny_taxi
```

## Running test cases manually
* To run the test cases with a detailed test results for the flask web service app manually, run the following command
```
pytest -vv
```