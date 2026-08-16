# 🔍 Credit Card Fraud Detection System

An end-to-end machine learning system for detecting fraudulent credit card transactions. Built with a full MLOps pipeline including experiment tracking, data versioning, containerization, and CI/CD automation.

---

## 📌 Project Overview

Credit card fraud is a critical problem in the financial industry. This project addresses it using a **Random Forest classifier** trained on real-world transaction data, deployed as a REST API with a complete MLOps workflow.

The system provides **real-time fraud predictions** through a REST API, with a custom decision threshold tuned to maximize recall — ensuring fewer fraudulent transactions go undetected.

---

## 🏗️ Architecture

```text
Raw Data (Kaggle)
      ↓
Data Preprocessing & Feature Engineering
      ↓
Model Training (Random Forest + RandomizedSearchCV)
      ↓
Experiment Tracking (MLflow)
      ↓
Data Versioning (DVC + AWS S3)
      ↓
Model Serialization (Pickle)
      ↓
REST API (FastAPI)
      ↓
Containerization (Docker)
      ↓
CI/CD Pipeline (GitHub Actions)
      ↓
Cloud Deployment (Render)
```

---

## 📊 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 99.95% |
| Precision | 95.89% |
| Recall    | 73.68% |
| F1 Score  | 83.33% |
| ROC-AUC   | 93.35% |

> **Note:** Accuracy is intentionally de-emphasized due to class imbalance (0.166% fraud rate). ROC-AUC and F1 Score are the primary evaluation metrics.
> A custom decision threshold of **0.4** was applied to optimize the precision-recall tradeoff for fraud detection.

---

## 🗂️ Project Structure

```text
Fraud-detection/
├── .dvc/                     # DVC configuration
├── .github/                  # GitHub Actions workflows
│   └── workflows/
│       └── ci.yaml           # GitHub Actions CI/CD workflow
├── data/
│   └── creditcard.csv.dvc    # DVC tracked dataset pointer
├── mlartifacts/              # MLflow artifacts
├── models/
│   ├── best_model.pickle     # Trained Random Forest model
│   └── preprocessor.pkl      # Fitted preprocessor (StandardScaler)
├── notebooks/
│   └── creditcard.ipynb      # EDA and experimentation notebook
├── reports/
│   ├── drift_report.html     # Check if our data is drifting compared to training input
│   ├── prediction_log.csv    # Where we store the data from our API
│   ├── prediction_outcome.csv # Stores the outcomes of our predictions
│   └── metrics.json          # Evaluation metrics
├── src/
│   ├── app.py                # FastAPI application
│   ├── data_preprocessing.py # Data cleaning and feature engineering
│   ├── evaluation.py         # Model evaluation
│   ├── model_training.py     # Model training with MLflow tracking
│   └── prediction.py         # Inference logic
├── .dockerignore
├── docker-compose.yaml
├── Dockerfile
├── dvc.yaml                  # DVC pipeline definition
├── requirements.txt          # Full dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Category            | Tools                                           |
| ------------------- | ----------------------------------------------- |
| Machine Learning    | Scikit-learn, Random Forest, RandomizedSearchCV |
| Data Processing     | Pandas, NumPy, StandardScaler                   |
| Experiment Tracking | MLflow                                          |
| Data Versioning     | DVC + AWS S3                                    |
| API                 | FastAPI, Uvicorn                                |
| Containerization    | Docker, Docker Compose                          |
| Monitoring          | Evidently                                       |
| CI/CD               | GitHub Actions                                  |
| Version Control     | Git, GitHub                                     |

---

## 📦 Dataset

* **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Size:** 284,807 transactions over 2 days
* **Fraud Rate:** 0.166% (highly imbalanced)
* **Features:** 28 PCA-transformed features (V1-V28) + Time + Amount
* **Target:** Class (0 = Normal, 1 = Fraud)

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11+
* Docker
* AWS CLI configured
* DVC

### 1. Clone the repository

```bash
git clone https://github.com/Sirvikkaz/Fraud-detection.git
cd Fraud-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull data with DVC

```bash
dvc pull
```

### 4. Run the training pipeline

```bash
dvc repro
```

### 5. Start the API locally

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### 6. Run with Docker

```bash
docker-compose up -d
```

---

## 🔌 API Reference

### Health Check

```text
GET /
```

**Response:**

```json
{"message": "Model ready!"}
```

### Predict Transaction

```text
POST /predict
```

**Request Body:**

```json
{
    "Time": 406.0,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
}
```

**Response:**

```json
{
    "prediction": 1,
    "label": "Fraud"
}
```

---

## ⚙️ MLOps Pipeline

### Experiment Tracking (MLflow)

All training runs are tracked with MLflow including:

* Hyperparameters (n_estimators, max_depth, min_samples_split etc)
* Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
* Model artifacts and confusion matrix

### Data Versioning (DVC)

Raw data and processed datasets are versioned with DVC and stored remotely on AWS S3:

```text
s3://fraud-detection-dvc-olanrewaju-victor
```

---

## 👨‍💻 Author

**Olanrewaju Victor Bayode**

* GitHub: [@Sirvikkaz](https://github.com/Sirvikkaz)
* LinkedIn: [victor-olanrewaju](https://linkedin.com/in/victor-olanrewaju-165634238)
* Email: [olanrewajuvictor2018@gmail.com](mailto:olanrewajuvictor2018@gmail.com)

---

## 📄 License

This project is licensed under the MIT License.
