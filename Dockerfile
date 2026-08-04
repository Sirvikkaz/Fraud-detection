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

#port
EXPOSE 8000

#comands
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
