#base image
FROM python:3.11-slim

#run
RUN apt update -y && apt install awscli -y

#workdir
WORKDIR /app

#install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "dvc[s3]"

COPY . .
#pull dvc tracked file from s3
RUN dvc pull models/best_model.pickle models/preprocessor.pkl

#port
EXPOSE 8000

#comands
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
