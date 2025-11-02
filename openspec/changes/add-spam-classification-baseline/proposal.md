# Add Spam Classification System (Phase 1 - Baseline)

## Overview
This change proposal introduces a baseline spam classification system using machine learning. The baseline has been implemented using Logistic Regression (replacing an earlier SVM prototype). Future phases will add enhancements such as richer preprocessing, model ensembling, and a user-facing demo.

## Motivation
Spam detection is a critical component of email security. Building a machine learning-based classification system will help filter unwanted messages effectively. Starting with a baseline SVM implementation will provide a foundation for future improvements.

## Scope
Phase 1 (Baseline):
- Data acquisition and preprocessing from provided dataset
- Implementation of Logistic Regression classifier (baseline)
- Basic model evaluation metrics and artifacts
- Initial API for classification

Future Phases (Placeholder):
- Phase 2: [To be determined]
- Phase 3: [To be determined]
- Phase 4: [To be determined]

## Impact
- New machine learning model implementation
- Data processing pipeline
- Model evaluation system
- Classification API
- Project structure for future extensions

## Technical Details
- Dataset Source: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- Baseline Algorithm: Logistic Regression (tuned)
- Language: Python
- Key Libraries: scikit-learn, pandas, numpy

### Tuned Baseline (summary)
- Tuning method: GridSearchCV (scoring='recall', cv=5)
- Parameters searched: penalty in [l1,l2], C in [0.01,0.1,1.0,10.0], solver=liblinear
- Best parameters found: {"C": 10.0, "penalty": "l1", "solver": "liblinear"}
- Test set performance (held-out):
	- Accuracy: 0.9785
	- Precision: 0.9496
	- Recall: 0.8859
	- F1 Score: 0.9167

Artifacts saved to `models/`:
- `spam_classifier.joblib` (LogisticRegression model with best params)
- `tfidf_vectorizer.joblib` (fitted TF-IDF vectorizer)

Recommended training flags (to reproduce or improve):
- class-weight: balanced
- ngram-range: 1,2
- min_df: 2
- sublinear_tf: true
- C: 10.0 (or tune further)

Notes: the tuning focused on improving recall (reduce false negatives). If you prefer a different trade-off (e.g., higher precision), re-run tuning with a different scoring metric or constrain precision via custom scoring.

## Risks
- Data quality and preprocessing challenges
- Model performance considerations
- Scalability concerns
- Memory usage for large datasets
- Processing time constraints

## Testing Requirements
- Data preprocessing validation
- Model accuracy metrics
- Cross-validation testing
- Performance benchmarking
- API functionality testing

## Documentation Requirements
- Data preprocessing steps
- Model training procedure
- API usage guide
- Performance metrics documentation
- Future phase considerations