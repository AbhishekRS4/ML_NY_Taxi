# Instructions for running the project on AWS

## AWS Instructions

### MLFlow server setup and running instructions on AWS
* The setup needed for running the entire project on AWS infrastructure is detailed in a blog post - [AWS instance setup and usage for deploying MLFlow based Machine Learning pipelines](https://abhishekrs4.github.io/blogs/tech_blogs/tech_blog_2.html)
* Run the following command to start the MLFlow server on an Amazon EC2 instance
```
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USERNAME:DB_PASSWORD@DB_ENDPOINT:DB_PORT/DB_NAME --default-artifact-root s3://S3_BUCKET_NAME
```
* The MLFlow UI can be accessed via the following link (**Note: http and not https**)
```
http://EC2_PUBLIC_DNS:5000
```


### Instructions to run training code for logging using the MLFlow server running on AWS
* To run the [trainer.py](trainer.py) script on the local machine and to log everything to the AWS MLFlow infrastructure, use the following command (**Note: http and not https**)
```
python3 trainer.py --mlflow_tracking_uri http://EC2_PUBLIC_DNS:5000/
```


## Prefect Instructions

### Prefect setup and running instructions
* The instructions for setting up Prefect, creating deployments and starting work pools can be found in a blog post - [Data Ingestion using Prefect for workflow orchestration](https://abhishekrs4.github.io/blogs/tech_blogs/tech_blog_3.html)
* Although the blog is regarding data ingestion pipeline, the prefect usage applies to ML use case as well with minor changes such as the target scripts and target flows
* To create AWS cred and S3 blocks in prefect, run the following script - [create_s3_bucket_block.py](create_s3_bucket_block.py)
* To run a prefect deployment, run
```
prefect deployment run 'trainer/ml-ny-taxi-dep'
```
* To start a work pool, run
```
prefect worker start --pool 'ml-ny-taxi'
```