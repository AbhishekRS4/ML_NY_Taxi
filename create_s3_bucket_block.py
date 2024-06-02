import os
from time import sleep
from ny_taxi.infra_utils.prefect_aws_utils import (
    create_aws_cred_block,
    create_s3_bucket_block,
)


def main() -> None:
    create_aws_cred_block()
    sleep(5)
    create_s3_bucket_block()
    return


if __name__ == "__main__":
    main()
