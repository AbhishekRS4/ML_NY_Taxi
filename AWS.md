# Instructions for running the project on AWS

## Info
* The setup needed for running the entire project on AWS infrastructure is detailed in a blog post
* The blog post is available - [AWS instance setup and usage for deploying MLFlow based Machine Learning pipelines](https://abhishekrs4.github.io/blogs/tech_blogs/tech_blog_2.html)


## Instructions for AWS
* Run the following command to start the mlflow server on an Amazon EC2 instance
```
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USERNAME:DB_PASSWORD@DB_ENDPOINT:DB_PORT/DB_NAME --default-artifact-root s3://S3_BUCKET_NAME
```
* The MLFlow UI can be accessed via the following link (**Note: http and not https**)
```
http://EC2_PUBLIC_DNS:5000
```


## Instructions to run code
* To run the [trainer.py](trainer.py) script on the local machine and to log everything to the AWS MLFlow infrastructure, use the following command (**Note: http and not https**)
```
python3 trainer.py --mlflow_tracking_uri http://EC2_PUBLIC_DNS:5000/
```
