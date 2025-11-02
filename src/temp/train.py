"""
Main script to train and evaluate the spam classifier.
"""
import os
from data_processor import DataProcessor
from spam_classifier import SpamClassifier

def main():
    # Initialize data processor and classifier
    data_processor = DataProcessor()
    classifier = SpamClassifier()
    
    # Dataset URL and paths
    DATASET_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
    DATA_PATH = os.path.join('data', 'spam_dataset.csv')
    MODEL_PATH = os.path.join('models', 'spam_classifier.joblib')
    VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.joblib')
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Download dataset
    print("Downloading dataset...")
    if not os.path.exists(DATA_PATH):
        success = data_processor.download_dataset(DATASET_URL, DATA_PATH)
        if not success:
            print("Failed to download dataset.")
            return
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = data_processor.load_data(DATA_PATH)
    if df is None:
        print("Failed to load data.")
        return
        
    X_train, X_test, y_train, y_test = data_processor.prepare_data(df)
    if X_train is None:
        print("Failed to prepare data.")
        return
    
    # Train the model
    print("Training model...")
    success = classifier.train(X_train, y_train)
    if not success:
        print("Failed to train model.")
        return
    
    # Evaluate the model
    print("\nEvaluating model...")
    metrics = classifier.evaluate(X_test, y_test)
    if metrics is None:
        print("Failed to evaluate model.")
        return
    
    # Print metrics
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save the model
    print("\nSaving model...")
    success = classifier.save_model(MODEL_PATH)
    if success:
        print(f"Model saved to {MODEL_PATH}")
    else:
        print("Failed to save model.")
        
    # Save the vectorizer
    print("\nSaving vectorizer...")
    success = data_processor.save_vectorizer(VECTORIZER_PATH)
    if success:
        print(f"Vectorizer saved to {VECTORIZER_PATH}")
    else:
        print("Failed to save vectorizer.")

if __name__ == "__main__":
    main()