# Machine Learning with New York Taxi data

## Info
* This is a ML and MLOps project with real-world taxi data
* The following page contains info regarding the [NY_Taxi_Dataset_Info](https://www.nyc.gov/site/tlc/passengers/your-ride.page)
* The dataset can be found in [NY_Taxi_Dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)


## MLFlow tracking UI locally
* To visualize using MLFlow UI, run the following command
```
mlflow ui --backend-store-uri sqlite:///mlruns.db
```


## For training using AWS and Prefect
* For training to use the MLFlow server on the AWS infrastructure and Prefect workpools, refer [AWS_Prefect.md](AWS_Prefect.md)


## For running the Flask app outside docker container
* For running the Flask app locally, set the environment variable using the following command
```
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
```
* Use the following command to run the Flask app
```
gunicorn --bind 0.0.0.0:7860 web_service_app:app
```
* For testing the POST request to the web\_service\_app, run the [test\_web\_service.py](test_web_service.py)