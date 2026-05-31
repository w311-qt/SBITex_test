FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.15.0
ENTRYPOINT ["mlflow"]
