![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)

![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)

![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white)

![Railway](https://img.shields.io/badge/Railway-Deployed-black?logo=railway)

![AWS S3](https://img.shields.io/badge/AWS_S3-Storage-FF9900?logo=amazonaws&logoColor=white)

![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)

# 🏡 Housing Price Prediction Dashboard

A production-ready end-to-end Machine Learning application that predicts housing prices using an XGBoost regression model. The project demonstrates how to take an ML model from experimentation to deployment with a modern MLOps workflow.

Unlike a typical notebook-based project, this application is designed with production practices in mind, including a FastAPI backend, an interactive Streamlit dashboard, Docker containers, CI/CD automation, cloud deployment, and model management.

---

## 🚀 Live Demo

**Dashboard:** *(precious-perfection-production-2dab.up.railway.app/dashboard)*

---

## Project Overview

This project predicts housing prices on unseen housing data and provides an interactive dashboard for exploring predictions, comparing results, and monitoring application performance.

The application separates the user interface from the prediction API, making it easier to scale, maintain, and deploy.

---

## Features

- Interactive Streamlit Dashboard
- FastAPI Prediction API
- XGBoost Regression Model
- Batch Prediction Support
- Dashboard Monitoring
- Prediction History
- Model Performance Metrics
- Cloud Deployment using Railway
- Dockerized Services
- CI/CD with GitHub Actions
- MLflow Integration
- AWS S3 Model Storage

---

## Tech Stack

### Machine Learning
- Python
- XGBoost
- Pandas
- NumPy
- Scikit-learn

### Backend
- FastAPI
- Uvicorn

### Frontend
- Streamlit

### MLOps
- Docker
- MLflow
- GitHub Actions
- Railway
- AWS S3

---

## Project Architecture

```
User
   │
   ▼
Streamlit Dashboard
   │
   ▼
FastAPI REST API
   │
   ▼
XGBoost Model
   │
   ▼
Prediction Results
```

---

## Folder Structure

```
housing-price-prediction-mlops/

├── app.py
├── main.py
├── src/
├── models/
├── notebooks/
├── tests/
├── Dockerfile
├── Dockerfile.streamlit
├── pyproject.toml
├── README.md
└── .github/
```

---

## Dashboard Highlights

The dashboard includes:

- Housing price prediction
- Dynamic filtering
- Prediction analytics
- Prediction latency monitoring
- Records processed statistics
- Dashboard uptime
- Prediction history
- Model information panel

---

## MLOps Highlights

This project follows a production-style workflow:

- Separate API and UI services
- Docker containers
- Cloud deployment
- Automated CI pipeline
- Version-controlled code
- MLflow model tracking
- AWS S3 integration
- Modular project structure

---

## Running Locally

Clone the repository

```bash
git clone https://github.com/manthanvyas23/housing-price-prediction-mlops.git
```

Move into the project

```bash
cd housing-price-prediction-mlops
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run FastAPI

```bash
python main.py
```

Run Streamlit

```bash
streamlit run app.py
```

---

## Deployment

The application is deployed using:

- Railway
- GitHub Actions
- Docker

Every push to the `main` branch automatically triggers the CI workflow and deploys the latest version.

---

## What I Learned

Building this project helped me gain hands-on experience with:

- Designing production-ready ML applications
- REST API development using FastAPI
- Interactive dashboard development with Streamlit
- Docker containerization
- CI/CD automation
- Cloud deployment
- MLflow model management
- AWS S3 integration
- Writing modular and maintainable Python code

---

## Future Improvements

- User authentication
- Model version comparison
- Real-time monitoring dashboard
- Automated model retraining
- Database integration
- Unit and integration test expansion
- Kubernetes deployment
- Infrastructure as Code

---

## About Me

Hi, I'm **Manthan Vyas**.

I'm an aspiring Data Scientist with a strong interest in Machine Learning, MLOps, and production-ready AI systems. I enjoy building projects that solve real-world problems while following software engineering best practices.

If you'd like to connect or discuss opportunities, feel free to reach out.

**GitHub:** https://github.com/manthanvyas23

---

⭐ If you found this project interesting, consider giving it a star.