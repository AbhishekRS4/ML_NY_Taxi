FROM public.ecr.aws/lambda/python:3.8

RUN pip install scikit-learn==1.3.2
RUN pip install mlflow==2.10.0

COPY ./model_for_prod/ ./model_for_prod/
COPY ./lambda_function.py .

ENV MLFLOW_MODEL_RUN_ID=8ea4b231f2f04a80a8d0105ab82afb15

CMD ["lambda_function.lambda_handler"]