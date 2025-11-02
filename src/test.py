"""
Test script for the spam classification system.
"""
import os
from data_processor import DataProcessor
from spam_classifier import SpamClassifier

def test_classification():
    # Test messages
    test_messages = [
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
        "I'll be there in 10 minutes. See you soon!",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
        "Hey, can you send me the meeting notes from yesterday?"
    ]
    
    # Initialize components
    data_processor = DataProcessor()
    classifier = SpamClassifier()
    
    # Paths
    DATA_PATH = os.path.join('data', 'spam_dataset.csv')
    MODEL_PATH = os.path.join('models', 'spam_classifier.joblib')
    VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.joblib')
    
    # Train model if not exists
    if not os.path.exists(MODEL_PATH):
        print("Training new model...")
        
        # Load and prepare data
        df = data_processor.load_data(DATA_PATH)
        if df is None:
            print("Failed to load data.")
            return
            
        X_train, X_test, y_train, y_test = data_processor.prepare_data(df)
        if X_train is None:
            print("Failed to prepare data.")
            return
            
        # Train and save model
        classifier.train(X_train, y_train)
        classifier.save_model(MODEL_PATH)
        data_processor.save_vectorizer(VECTORIZER_PATH)
    else:
        print("Loading existing model...")
        classifier.load_model(MODEL_PATH)
        # Load the fitted vectorizer so preprocessing works
        data_processor.load_vectorizer(VECTORIZER_PATH)
    
    # Test classification
    print("\nTesting classification:")
    for message in test_messages:
        # Preprocess message
        X = data_processor.preprocess_text(message)
        if X is None:
            continue
            
        # Get prediction and probability
        prediction = classifier.predict(X)
        probabilities = classifier.predict_proba(X)
        
        # Print results
        print("\nMessage:", message[:100], "..." if len(message) > 100 else "")
        print("Classification:", "SPAM" if prediction[0] == 1 else "HAM")
        print(f"Confidence: {max(probabilities[0]):.2%}")
        print("-" * 80)

if __name__ == "__main__":
    test_classification()