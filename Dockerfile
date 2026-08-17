FROM python:3.11-slim

RUN apt update -y && apt install awscli -y

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "dvc[s3]"

COPY . .

# Port
EXPOSE 8000

# Download models at runtime not build time
CMD ["sh", "-c", "aws s3 cp s3://fraud-model-452110701928/models models --recursive && uvicorn src.app:app --host 0.0.0.0 --port $PORT"]