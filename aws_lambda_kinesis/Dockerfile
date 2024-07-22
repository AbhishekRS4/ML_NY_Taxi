FROM public.ecr.aws/lambda/python:3.8

RUN pip install -U pip
RUN pip install boto3==1.34.99
RUN pip install scikit-learn==1.3.2
RUN pip install xgboost==2.0.3
RUN pip install mlflow==2.10.0

COPY ./lambda_function.py .

CMD ["lambda_function.lambda_handler"]