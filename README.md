🎓 Career Path Prediction System
================================

A Machine Learning–Powered RIASEC Career Recommender
----------------------------------------------------

### 📌 Overview

This project implements an end-to-end career path prediction system, built on structured psychometric (RIASEC) questionnaire responses.It includes:

*   Data cleaning & preprocessing pipeline (48-item RIASEC questionnaire)
    
*   Machine Learning model (Logistic Regression, multinomial)
    
*   REST API using FastAPI
    
*   Interactive Web App using Streamlit
    
*   Testing suite with ≥ 70% coverage (pytest + pytest-cov)
    
*   Continuous Integration using GitHub Actions
    
*   Deployment to Streamlit Cloud
    

The system predicts top-5 most suitable career categories, provides probabilities, and presents an intuitive UI where users rate 48 activities on a 1–5 scale.

### 🧠 Model Summary

The final model was selected after multiple MLflow-logged experiments:

Experiment

Undersampling

Hyperparameters

Accuracy

Top-5 Accuracy

Logistic Regression (baseline)

❌

Default

0.24

0.51

Logistic Regression (undersampling)

✔

Default

0.14

0.39

Logistic Regression (undersampling + tuning)

✔

GridSearch

0.14

0.39

**Logistic Regression (no undersampling + tuning)**

❌

GridSearch

**0.04**

**0.19**

*   **Final chosen model:** Logistic Regression (48 features, no undersampling, default hyperparameters)
    
*   **🎯 Reason:** Highest top-5 accuracy while preserving class distribution integrity.
    

### 📁 Project Structure

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ml_career_path_api/  │  ├── app.py                      # FastAPI backend  ├── streamlit_app.py            # Streamlit user interface  ├── train_model.py              # ML model training script  │  ├── prepare_data.py             # Data cleaning (aggregated RIASEC scores)  ├── prepare_data_48.py          # Data cleaning (48 original RIASEC items)  │  ├── model/  │   ├── logreg_model.pkl        # Trained model  │   ├── label_encoder.pkl       # Encodes/decodes majors  │   └── feature_list.json       # Feature metadata  │  ├── data/  │   ├── data.csv                # Raw questionnaire data  │   ├── final_data.csv          # Cleaned aggregated dataset (6 features)  │   └── final_data_48.csv       # Cleaned expanded dataset (48 features)  │  ├── tests/  │   ├── test_api.py  │   ├── test_model.py  │   ├── test_prepare_data.py  │   ├── test_prepare_data_48.py  │   └── test_streamlit_app.py  │  ├── requirements.txt  ├── README.md                   # (this file)  └── .github/workflows/ci.yml    # GitHub Actions CI/CD   `

### 🚀 Quick Start (Local Deployment)

1️⃣ **Create virtual environment**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python3 -m venv .venv  source .venv/bin/activate   `

2️⃣ **Install dependencies**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

3️⃣ **Prepare the dataset**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python prepare_data_48.py       # generates data/final_data_48.csv   `

4️⃣ **Train the model**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python train_model.py   `

5️⃣ **Run API**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   uvicorn app:app --reload   `

Visit: http://localhost:8000/docs

6️⃣ **Run Streamlit UI**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit run streamlit_app.py   `

### 🌐 Streamlit App (Cloud Deployment)

Your Streamlit app automatically redeploys whenever you push to main.

Example entry in .streamlit/config.toml is optional.

To deploy manually:

1.  Go to https://streamlit.io/cloud
    
2.  Connect your GitHub repo
    
3.  Select:
    
    *   Main file: streamlit\_app.py
        
    *   Python version: 3.11 or 3.12
        

### 🧪 Testing & Coverage

Run full test suite:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pytest -v   `

With coverage:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pytest --cov=. --cov-report=term-missing   `

A CI step enforces minimum 70% coverage:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   - name: Enforce minimum test coverage    run: |      coverage_total=$(coverage report | awk 'END{print $4}' | sed 's/%//')      if (( $(echo "$coverage_total < 70" | bc -l) )); then        exit 1      fi   `

### ⚙️ CI/CD Pipeline (GitHub Actions)

CI runs on every:

*   push to main
    
*   pull request targeting main
    

It performs:

*   ✔ Install dependencies
    
*   ✔ Run tests
    
*   ✔ Compute coverage
    
*   ✔ Enforce coverage threshold
    
*   ✔ Build Docker image (optional)
    
*   ✔ Trigger Streamlit auto-deploy
    

Workflow file: .github/workflows/ci.yml

### 📊 Data Preparation Summary

Both prepare\_data.py and prepare\_data\_48.py perform:

*   Duplicate removal
    
*   Text normalization
    
*   Major name standardization
    
*   Dictionary mapping
    
*   Fuzzy matching (RapidFuzz)
    
*   Removal of rare classes
    
*   Creation of clean datasets
    

Test mode (TEST\_MODE=1) disables fuzzy logic for deterministic testing.

### 🖥 API Usage (FastAPI)

POST /predict

**Request:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   {    "features": [0.12, 0.52, 0.33, ... 48 values]  }   `

**Response:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   {    "top5_predictions": [      {"label": "Nursing", "probability": 0.62},      {"label": "Biology", "probability": 0.21},      ...    ]  }   `

Interactive docs available at:👉 http://localhost:8000/docs

### 🎨 Streamlit UI Features

*   48 sliders (default value = 1)
    
*   Instruction text explaining rating scale
    
*   Top-5 predicted career paths with probabilities
    
*   Clean, user-friendly layout
    

### 🔐 Ethical Considerations

*   The model predicts career categories, not abilities or personal worth.
    
*   Data originates from self-reported questionnaire responses.
    
*   Predictions should not be used as the sole basis for academic or career decisions.
    
*   Model should be retrained periodically due to data drift, evolving majors, and new educational trends.
