# Instructions for running the project on AWS

## Info
* The setup needed for running the entire project on AWS infrastructure is detailed in a blog post
* The blog post is available - [AWS instance setup and usage for deploying MLFlow based Machine Learning pipelines](https://abhishekrs4.github.io/blogs/tech_blogs/tech_blog_2.html)


## Instructions to run code
* To run the [trainer.py](trainer.py) script on the local machine and to log everything to the AWS MLFlow infrastructure, use the following command
```
python3 trainer.py --mlflow_tracking_uri http://EC2_PUBLIC_DNS:5000/
```
