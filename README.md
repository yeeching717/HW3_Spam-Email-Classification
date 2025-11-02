# IoTHW3 — Spam Classification Subproject

This workspace contains an IoT homework project with an added spam classification subproject (SMS/email). The spam classifier is implemented in Python and uses a TF-IDF + Logistic Regression pipeline. The repository includes preprocessing utilities, training/tuning scripts, and a small test harness.

Quick summary
- Baseline: Logistic Regression (tuned via GridSearchCV)
- Data: Packt Publishing SMS spam dataset (headerless CSV)
- Artifacts: `models/spam_classifier.joblib`, `models/tfidf_vectorizer.joblib`

Reproducible steps (Windows / PowerShell)

1. Create environment and install dependencies

```powershell
conda create -p .conda python=3.9 pandas numpy scikit-learn requests joblib -y
conda activate ./.conda
pip install -r requirements.txt
```

2. Run training (downloads dataset, preprocesses, trains tuned LogisticRegression, and saves artifacts)

```powershell
python src/train.py
```

3. Run quick tests

```powershell
python src/test.py
```

Recommended training flags (used for final artifacts)
- class_weight: balanced
- ngram_range: 1,2
- min_df: 2
- sublinear_tf: True
- LogisticRegression: penalty='l1', C=10.0, solver='liblinear', class_weight='balanced'

Artifacts
- `models/spam_classifier.joblib` — trained LogisticRegression model
- `models/tfidf_vectorizer.joblib` — fitted TF-IDF vectorizer
- `datasets/processed/sms_spam_clean.csv` — cleaned dataset used for training

Tuning
- `src/tune.py` runs GridSearchCV (scoring='recall') to find hyperparameters that improve recall. Modify `param_grid` to expand search.

Next steps (suggested)
- Add pytest unit tests and a GitHub Actions workflow to run tests automatically.
- Update the OpenSpec proposal and project docs (done) with final metrics and parameters.
- Optionally deploy the Streamlit demo at `app/streamlit_app.py`.

If you want, I can add a minimal test suite + CI workflow next.
