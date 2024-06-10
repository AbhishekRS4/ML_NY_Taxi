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

COPY ./ny_taxi/ ./ny_taxi/
COPY ./model_for_prod/ ./model_for_prod/
COPY ./web_service_app.py .

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "web_service_app:app"]