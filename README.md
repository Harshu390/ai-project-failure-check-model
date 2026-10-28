# AI / AIOps project failure-risk model + project selection tool

Use Python 3.12 (not 3.15).

## Analysis (produces the charts and numbers for the report)
    py -3.12 -m venv venv
    venv\Scripts\activate
    pip install --only-binary=:all: -r requirements-full.txt
    python 01_build_dataset.py
    python 02_train_evaluate.py     # 3 models compared, charts/, SHAP
    python 03_leakage_check.py      # honest check on the 17 real rows

## Tool (deployed app)
    streamlit run app.py

The app trains a Logistic Regression on dataset.csv at start-up (no pickle needed), so deployment needs only:
app.py, dataset.csv, requirements.txt  (requirements.txt is the light file; requirements-full.txt is for the analysis).

Why Logistic Regression in the app: near-best CV accuracy, best result on the honest real-case check,
smoother probabilities than XGBoost (which saturates at 1% / 99%), and coefficients are explainable.