FROM python:3.8.10-slim

WORKDIR /app

# install linux package dependencies
RUN apt-get update -y && \
    # apt-get -y install gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# can copy files only from current working directory where docker builds
# cannot copy files from arbitrary directories

WORKDIR /app

COPY ./requirements_dep.txt .

RUN pip install --no-cache-dir -r requirements_dep.txt

COPY ./model_for_prod/ ./model_for_prod/
COPY ./web_service_app.py .

ENV MLFLOW_MODEL_RUN_ID=8ea4b231f2f04a80a8d0105ab82afb15

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "web_service_app:app"]