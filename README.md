# 🚀 MLflow & FastAPI Integration Project

This project demonstrates how to:

- ✅ Track and log machine learning experiments with **MLflow**  
- 🌐 Serve a REST API with **FastAPI**  
- 🤖 Integrate **Hugging Face Transformers Pipelines**  
- 🔁 Handle multiple GET operations in a modular API  
- 🧪 Work within a **conda environment** for reproducibility

---

## 📁 Repository Structure

MLflow_FastAPI/
│
├── FastAPI/
│ ├── screenshots/ # UI and results screenshots
│ ├── fastapi_main.py # Main FastAPI app with 5 GET endpoints
│ └── requests_script.py # Example client script to test endpoints
│
├── MLflow_tracking/
│ ├── MLflow_notebook.ipynb # Notebook for ML training and MLflow tracking
│ ├── functions.py # Helper functions for training and logging
│ └── main.py # Script for model training with MLflow tracking
│
└── environment: conda


---

## 🧠 MLflow Tracking

The `MLflow_tracking` folder contains:

- 📒 **Jupyter notebook** for step-by-step training and experiment tracking  
- 🛠️ **Reusable functions** in `functions.py` to simplify training logic  
- ▶️ **Executable script** `main.py` for training a model and logging metrics, parameters, and artifacts with MLflow  

📌 MLflow was used to track:
- Model accuracy
- Training parameters
- Artifact versioning
- Experiment comparison

---

## ⚡ FastAPI Application

The `FastAPI` folder includes a modular REST API (`fastapi_main.py`) with **five GET endpoints**, including:

- 🧮 Basic data processing endpoints
- 🤗 Two endpoints using **Hugging Face Transformers Pipelines** for:
  - Text classification
  - Sentiment analysis

### 🔗 Access

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
- Test with: `requests_script.py`

---

## ▶️ Running the Project

### 1. ✅ Clone the repository
```bash
git clone https://github.com/your-username/MLflow_FastAPI.git
cd MLflow_FastAPI
2. 🐍 Create the conda environment
conda create -n mlflow_fastapi_env python=3.10
conda activate mlflow_fastapi_env
3. 📦 Install dependencies
pip install -r requirements.txt
4. 📊 Run MLflow training
cd MLflow_tracking
python main.py
5. 🌐 Launch FastAPI server
cd ../FastAPI
uvicorn fastapi_main:app --reload
📸 Screenshots

Screenshots of Swagger UI and example responses are available in the FastAPI/screenshots/ folder.

💡 Technologies Used

MLflow
FastAPI
Hugging Face Transformers
Anaconda
Uvicorn
