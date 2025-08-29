Feature Engineering & Class Balancing with SMOTE
📌 Project Overview

This project demonstrates a complete data preprocessing and feature engineering pipeline applied to a dataset of ~500K+ records.
The focus is on ensuring clean inputs, handling class imbalance, and preparing data for machine learning models.

⚙️ Pipeline Steps & Rationale
1. Data Loading

Used: Pandas CSV loader (pd.read_csv)

Why: The dataset was originally in CSV format; Pandas offers the most convenient functions for exploration and cleaning.

Not Used: Polars / Parquet

Although tested for performance, they were not adopted in the final version since the pipeline already ran efficiently with Pandas.

2. Data Cleaning

Handled missing values & infinite values

Replaced NaN/Inf with column means to avoid training failures.

Why: Machine learning algorithms cannot handle NaN values directly.

Optimized datatypes

Converted columns to smaller numeric types (e.g., int32, float32) to save memory.

Why: Dataset had 500K+ rows; memory optimization improved efficiency.

3. Feature Engineering

Converted categorical variables into numeric form using Label Encoding.

Why: ML models require numerical input.

Ensured all features were numeric and compatible for downstream algorithms.

4. Handling Class Imbalance

Problem: The dataset was highly imbalanced → majority class overwhelmed the minority.

Tried:

SMOTE (Used ✅)

Generates synthetic samples of the minority class by interpolating between neighbors.

Balanced the dataset successfully and improved minority representation by ~300%.

ADASYN (Not Used ❌)

Adaptive version of SMOTE that focuses on harder-to-learn minority samples.

Issue: Failed on this dataset → “No neighbors in majority class” error.

Decision: Dropped in favor of SMOTE since it produced valid results.

5. Visualization & Validation

Before SMOTE: Severe class imbalance (majority class dominated).

After SMOTE: Nearly balanced dataset (green bars).

Added both bar charts and class count summaries to clearly show the improvement.

📊 Results

Original dataset: ~90% majority class vs ~10% minority class.

After SMOTE: ~50-50 balance between classes.

Minority class representation increased by 300%.

Enabled fairer training of machine learning models by reducing bias.

🚀 Tech Stack

Python 3.11

Pandas, NumPy → Data manipulation

Matplotlib → Visualization

Scikit-learn → Preprocessing & encoding

Imbalanced-learn → SMOTE & ADASYN

📂 Repository Structure
├── Member1_FeatureEngineering.ipynb   # Main notebook with pipeline
├── README.md                          # Project documentation
└── requirements.txt                   # Dependencies

✅ How to Run

Clone the repository

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name


Install dependencies

pip install -r requirements.txt


Run the notebook

jupyter notebook Member1_FeatureEngineering.ipynb

📌 Applications

This pipeline can be reused in:

Fraud Detection – oversampling rare fraudulent transactions.

Healthcare – predicting rare diseases.

Predictive Maintenance – detecting rare failure events.

Any Imbalanced Dataset Problem where fairness is critical.

✨ Key Learnings & Choices

SMOTE was chosen because it successfully balanced the dataset without errors.

ADASYN was rejected because the dataset’s structure made it unsuitable (neighbors issue).

Polars/Parquet were skipped as Pandas already offered sufficient performance.

Data cleaning (NaN/Inf replacement) was essential since SMOTE cannot handle missing values.

🔥 This project highlights how thoughtful preprocessing and the right choice of resampling technique can drastically improve the fairness and reliability of ML models.
