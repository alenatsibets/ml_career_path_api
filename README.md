Career Path Prediction System
================================

A Machine Learning–Powered RIASEC Career Recommender
----------------------------------------------------

### Overview

This project implements an end-to-end career path prediction system, built on structured psychometric (RIASEC) questionnaire responses. It includes:

*   Data cleaning & preprocessing pipeline (48-item RIASEC questionnaire)
    
*   Machine Learning model (Logistic Regression, multinomial)
    
*   REST API using FastAPI
    
*   Interactive Web App using Streamlit
    
*   Testing suite with ≥ 70% coverage (pytest + pytest-cov)
    
*   Continuous Integration using GitHub Actions
    
*   Deployment to Streamlit Cloud
    

The system predicts top-5 most suitable career categories, provides probabilities, and presents an intuitive UI where users rate 48 activities on a 1–5 scale.

### 🚀 Quick Start (Local Deployment)

**1. Create virtual environment**

`   python3 -m venv .venv
    
    source .venv/bin/activate   `

**2. Install dependencies**

`   pip install -r requirements.txt   `

**3. Prepare the dataset**

`   python prepare_data_48.py       # generates data/final_data_48.csv   `

**4. Train the model**

`   python train_model.py   `

**5. Run API**

`   uvicorn app:app --reload   `

Visit: http://localhost:8000/docs

**6. Run Streamlit UI**

`   streamlit run streamlit_app.py   `

### Streamlit App (Cloud Deployment)

https://mlcareerpathapi.streamlit.app/
        

### Testing & Coverage

Run full test suite:
`   pytest -v   `

With coverage:

`   pytest --cov=. --cov-report=term-missing   `

A CI step enforces minimum 70% coverage:

`   - name: Enforce minimum test coverage    

run: |      coverage_total=$(coverage report | awk 'END{print $4}' | sed 's/%//')      

if (( $(echo "$coverage_total < 70" | bc -l) )); 

then        

exit 1      

fi   `

### CI/CD Pipeline (GitHub Actions)

CI runs on every:

*   push to main
    
*   pull request targeting main
    

It performs:

*   ✔ Install dependencies
    
*   ✔ Run tests
    
*   ✔ Compute coverage
    
*   ✔ Enforce coverage threshold
        
*   ✔ Trigger Streamlit auto-deploy
    

Workflow file: .github/workflows/ci.yml

### Data Preparation Summary

Both prepare\_data.py and prepare\_data\_48.py perform:

*   Duplicate removal
    
*   Text normalization
    
*   Major name standardization
    
*   Dictionary mapping
    
*   Fuzzy matching (RapidFuzz)
    
*   Removal of rare classes
    
*   Creation of clean datasets
    

Test mode (TEST\_MODE=1) disables fuzzy logic for deterministic testing.

### API Usage (FastAPI)

POST /predict

**Request:**

`   {    "features": [0.12, 0.52, 0.33, ... 48 values]  }   `

**Response:**

`   {    "top5_predictions": [      

{"label": "Nursing", "probability": 0.62},     

{"label": "Biology", "probability": 0.21},      ...    ]  }   `

Interactive docs available at: http://localhost:8000/docs

### Streamlit UI Features

*   48 sliders (default value = 1)
    
*   Instruction text explaining rating scale
    
*   Top-5 predicted career paths with probabilities
    
*   Clean, user-friendly layout
  
