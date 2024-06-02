import boto3
from prefect_aws import S3Bucket, AwsCredentials


def create_aws_cred_block() -> None:
    session = boto3.Session()
    credentials = session.get_credentials()
    credentials = credentials.get_frozen_credentials()

    aws_access_key = credentials.access_key
    aws_secret_key = credentials.secret_key

    aws_cred = AwsCredentials(
        aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
    )
    aws_cred.save("my-aws-cred", overwrite=True)
    return


def create_s3_bucket_block() -> None:
    aws_cred = AwsCredentials.load("my-aws-cred")
    aws_s3_bucket = S3Bucket(bucket_name="mlops-prefect", credentials=aws_cred)
    aws_s3_bucket.save(name="aws-s3-bucket", overwrite=True)
    return
