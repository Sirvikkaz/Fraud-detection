#base image
FROM python:3.11-slim

#run
RUN apt update -y && apt install awscli -y

#workdir
WORKDIR /app

#install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

RUN mkdir -p models && aws s3 cp s3://fraud-model-452110701928/models models --recursive

#port
EXPOSE 8000

#comands
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
