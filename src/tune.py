"""
Hyperparameter tuning for Logistic Regression using GridSearchCV.
Optimizes for recall to improve spam detection recall.
"""
import os
from data_processor import DataProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

def main():
    dp = DataProcessor()

    DATASET_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
    DATA_PATH = os.path.join('data', 'spam_dataset.csv')
    MODEL_PATH = os.path.join('models', 'spam_classifier.joblib')
    VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.joblib')

    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Ensure dataset exists
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        if not dp.download_dataset(DATASET_URL, DATA_PATH):
            print("Failed to download dataset.")
            return

    print("Loading and preparing data...")
    df = dp.load_data(DATA_PATH)
    if df is None:
        print("Failed to load data.")
        return

    X_train, X_test, y_train, y_test = dp.prepare_data(df)
    if X_train is None:
        print("Failed to prepare data.")
        return

    # Grid search: tune penalty and C. Use solver 'liblinear' which supports l1 and l2.
    param_grid = {
        'penalty': ['l2', 'l1'],
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['liblinear']
    }

    print("Starting GridSearchCV (scoring='recall')...")
    base = LogisticRegression(max_iter=1000)
    grid = GridSearchCV(base, param_grid, scoring='recall', cv=5, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best = grid.best_estimator_

    # Evaluate on test set
    y_pred = best.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\nTuned Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save best model and vectorizer
    print("\nSaving tuned model and vectorizer...")
    joblib.dump(best, MODEL_PATH)
    dp.save_vectorizer(VECTORIZER_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")

if __name__ == '__main__':
    main()
