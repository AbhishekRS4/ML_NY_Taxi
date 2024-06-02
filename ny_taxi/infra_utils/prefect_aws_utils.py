import boto3
from prefect_aws import S3Bucket, AwsCredentials


def create_aws_cred_block_block() -> None:
    # create a boto3 session and get the AWS credentials
    # One can retrieve the credentials only after setting up the AWS credentials using the following command
    # `aws configure`
    session = boto3.Session()
    credentials = session.get_credentials()
    credentials = credentials.get_frozen_credentials()

    aws_access_key = credentials.access_key
    aws_secret_key = credentials.secret_key

    # create and save the AWS credentials block in Prefect
    aws_cred_block = AwsCredentials(
        aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
    )
    aws_cred_block.save("my-aws-cred", overwrite=True)
    return


def create_s3_bucket_block() -> None:
    # load the previously created AWS credentials block
    aws_cred_block = AwsCredentials.load("my-aws-cred")

    # create and save AWS S3 bucket block in Prefect
    aws_s3_bucket_block = S3Bucket(
        bucket_name="mlops-prefect", credentials=aws_cred_block
    )
    aws_s3_bucket_block.save(name="nytaxi-aws-s3-prefect-bucket", overwrite=True)
    return
