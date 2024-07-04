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


## Flask web service app
* The flask web service app scripts can be found in [flask_web_service](flask_web_service)
* The instructions for running the flask web service app can be found in [flask_web_service/README.md](flask_web_service/README.md)


## For deploying the AWS Lambda function to AWS and creating an API endpoint
* The AWS lambda function can be found in [aws_lambda_func](aws_lambda_func)
* The instructions for running the AWS lambda function can be found in [aws_lambda_func/README.md](aws_lambda_func/README.md)