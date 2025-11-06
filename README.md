# 🚀 Trace Transformer – Network Traffic Analysis & Intrusion Detection
## 🧩 Overview

Trace Transformer is an advanced **network traffic analysis and intrusion detection system** leveraging the **FT-Transformer architecture** for high-dimensional tabular and trace data. The project integrates powerful preprocessing, deep learning, optimization, and explainable AI components — built collaboratively by four team members — to ensure accurate, interpretable, and secure network monitoring.


### 👥 Team Contributions
## 🧮 Member 1 – Data Engineering & Feature Processing

**Role:** Data Preprocessing and Feature Engineering Specialist
**Focus:** Preparing clean, structured, and scalable datasets for downstream machine learning.

#### Key Contributions

- Cleaned and preprocessed large-scale network traffic datasets.

- Engineered and transformed 80+ traffic-related features.

- Implemented missing-value imputation, normalization, and outlier handling.

- Conducted exploratory data analysis (EDA) to identify network anomalies.

- Designed a modular data-cleaning pipeline for reproducibility.

#### Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

#### Setup
 ``` pip install pandas numpy scikit-learn matplotlib seaborn jupyter ```

 ***Run: Member1_FeatureEngineering.ipynb after placing CSV datasets in the working directory.***

## 🤖 Member 2 – Deep Learning Architecture & Ensembling

**Role:** Deep Learning Architect
**Focus:** FT-Transformer implementation and ensemble stacking for robust classification.

#### Key Contributions

- Designed and implemented an FT-Transformer for trace/log data using PyTorch.

- Built data loaders and custom dataset pipelines for sequence modeling.

- Developed ensemble and stacking frameworks combining multiple deep models.

- Performed extensive hyperparameter tuning and experiment tracking.

- Created reusable training, validation, and inference modules.

#### Tech Stack
PyTorch, Scikit-learn, NumPy, Pandas, Matplotlib, Jupyter

#### Setup
```
git clone https://github.com/aadyanair/Trace_Transformer.git
cd Trace_Transformer
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

```

***Run: Member2_work(1).ipynb to train the FT-Transformer and stacking ensemble.***


## ⚙️ Member 3 – Model Optimization & Evaluation

**Role:** Optimization & Evaluation Specialist
**Focus:** Hyperparameter tuning, performance evaluation, and model validation.

#### Key Contributions

- Used Optuna for automated hyperparameter optimization (learning rate, dropout, batch size).

- Implemented early stopping and learning rate scheduling to prevent overfitting.

- Achieved validation accuracy ≈ 92.1% and weighted F1 ≈ 0.91.

- Built **AdvancedModelEvaluator** for detailed metrics and visualizations (Confusion Matrix, ROC, PR Curves).

- Automated model saving and configuration management with JSON and PyTorch checkpoints.

#### Files

```Member3_work.ipynb ```– main training and evaluation notebook

#### Key Results

- Validation Accuracy: 92.1%

- Weighted F1: 0.91+

- Stable training by epoch 36


## 🔐 Member 4 – Explainability & Security Integration

**Role:** Explainable AI & Security Specialist
**Focus:** Interpretability of model predictions and integration with Zero Trust security.

#### Key Contributions

- Applied SHAP and Integrated Gradients for feature-level interpretability.

- Generated counterfactual explanations to understand decision boundaries.

- Implemented a Zero Trust Security layer with token-based authentication and confidence-based prediction flags.

- Built an Explainable AI wrapper for per-request interpretability.

- Ensured reliable and transparent model deployment.

#### Key Insights

- SHAP and Integrated Gradients identified consistent top features.

- Counterfactuals showed stable and logical decision boundaries.

- Zero Trust pipeline effectively filtered low-confidence predictions.

#### Files

```Member4_work_final.ipynb``` – SHAP, counterfactuals, and Zero Trust modules

## 📊 Project Highlights

| Component          | Description                                         | Tools Used           |
| ------------------ | --------------------------------------------------- | -------------------- |
| Data Preprocessing | Cleaning, feature scaling, missing value imputation | Pandas, NumPy        |
| Deep Learning      | FT-Transformer, model stacking                      | PyTorch              |
| Optimization       | Optuna-based hyperparameter tuning                  | Optuna, Scikit-learn |
| Evaluation         | Metrics, confusion matrix, ROC, PR curves           | Matplotlib           |
| Explainability     | SHAP, Integrated Gradients, Counterfactuals         | SHAP, Captum         |
| Security           | Zero Trust authentication and logging               | Python, PyTorch      |


# 🧠 Results Summary

**Validation Accuracy:** ≈ 92.1%

**Weighted F1 Score:** ≈ 0.91

**Early Stopping Epoch:** 36

**Architecture:** FT-Transformer (256 dim, 7 layers, 16 heads)

**Ensemble Accuracy:** 92.52% across 7 classes

## 🔮 Future Scope

- Integrate SHAP-based feature importance in real-time dashboards.

- Combine FT-Transformer with TabNet or XGBoost for hybrid ensembles.

- Extend Zero Trust to federated learning environments.

- Deploy using FastAPI or Docker for scalable production inference.



## 💡 Summary

Trace Transformer unites data preprocessing, deep learning, optimization, and explainable AI to deliver a **secure and interpretable network intrusion detection system**.
Each member’s contribution forms an essential layer of the pipeline — from raw data to explainable predictions — making it robust, scalable, and production-ready.


