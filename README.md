# Heart Disease Prediction MLOps Project

This repository contains an end-to-end machine learning application for predicting heart disease risk using a Flask web app, ML training pipeline, Docker containerization, and CI/CD deployment setup.

## Overview

The project includes:
- a training pipeline for data ingestion, preprocessing, model training, and artifact generation
- a Flask-based web application for single-record and batch prediction
- Docker support for containerized deployment
- Kubernetes manifests for deployment orchestration
- GitHub Actions workflows for linting, testing, building, and deployment

## Project Goals

- Build a reliable heart disease prediction model
- Serve predictions through a simple web interface
- Package the solution for deployment in cloud/container environments
- Implement a CI/CD workflow for repeatable releases

## Tech Stack

- Python 3.11
- Flask
- scikit-learn
- pandas / numpy
- Docker
- Kubernetes
- GitHub Actions
- pytest and flake8

## Project Structure

```text
app.py                  # Flask application entry point
Dockerfile              # Container definition
requirements.txt       # Python dependencies
setup.py                # Package metadata
src/
  components/
    data_ingestion.py
    data_transformation.py
    model_trainer.py
  pipeline/
    predict_pipeline.py
    train_pipeline.py
  exception.py
  logger.py
  utils.py
templates/             # HTML templates for the web UI
tests/                 # Unit and integration tests
artifacts/             # Trained model and preprocessing artifacts
k8s/                   # Kubernetes deployment manifests
.github/workflows/     # CI/CD automation
screenshots/           # Folder for report screenshots
```

## Model and Prediction Flow

The ML pipeline performs the following steps:
1. Load heart disease dataset from the notebook data folder
2. Split data into train and test sets
3. Apply preprocessing and scaling
4. Train classification models
5. Save the best model and preprocessing object into artifacts
6. Use those artifacts during prediction requests from the Flask app

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd heart_disease_mlops
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

Run the training pipeline:

```bash
python src/components/data_ingestion.py
```

This generates the model and preprocessing artifacts in the artifacts directory.

## Run the Web App Locally

```bash
python app.py
```

Then open:

```text
http://localhost:8000
```

## API Endpoints

- `/` - Home page
- `/health` - Health check endpoint
- `/predictdata` - Single-record prediction form
- `/bulk_predict` - Upload JSON file for batch prediction

## Testing

Run unit tests:

```bash
pytest -q
```

Run lint checks:

```bash
python -m flake8 . --exclude=.git,venv,venv312,.venv,__pycache__
```

## Docker

Build the image:

```bash
docker build -t heart-disease .
```

Run the container:

```bash
docker run -p 8000:8000 heart-disease
```

## Kubernetes Deployment

The deployment manifests are available in the k8s folder.

Apply them with:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
```

Replace the image name in the deployment manifest if needed before applying.

## CI/CD

The repository includes GitHub Actions workflows for:
- linting with flake8
- running pytest
- building a Docker image
- deploying to Azure Web App

Workflow files are located under:
- `.github/workflows/ci-cd.yml`
- `.github/workflows/main_heartdiseaseanalysis.yml`

## Deliverables and Reporting

Use the screenshots folder to store images for project reporting, such as:
- application homepage
- deployment screenshots
- workflow run screenshots
- monitoring/logging screenshots

## Notes

- The model artifacts are expected in the artifacts folder before prediction requests are served.
- The app uses the default port 8000 unless overridden with the PORT environment variable.
