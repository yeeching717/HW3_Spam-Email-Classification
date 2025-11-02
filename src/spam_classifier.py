"""
Logistic Regression classifier for spam detection.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib


class SpamClassifier:
    def __init__(self, penalty='l2', C=1.0, random_state=42, max_iter=1000, solver='lbfgs', class_weight=None):
        """Initialize a LogisticRegression model with sensible defaults.

        Args:
            penalty: Regularization penalty ('l1' or 'l2').
            C: Inverse of regularization strength.
            random_state: RNG seed for reproducibility.
            max_iter: Maximum number of solver iterations.
            solver: Solver to use ('lbfgs','liblinear','saga', etc.).
        """
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            random_state=random_state,
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight
        )
        
    def train(self, X_train, y_train):
        """Train the SVM classifier."""
        try:
            self.model.fit(X_train, y_train)
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
            
    def predict(self, X):
        """Make predictions on new data."""
        try:
            return self.model.predict(X)
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
            
    def predict_proba(self, X):
        """Get prediction probabilities."""
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            print(f"Error getting prediction probabilities: {e}")
            return None
            
    def evaluate(self, X_test, y_test):
        """Evaluate the model's performance."""
        try:
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test,
                y_pred,
                average='binary'
            )
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix
            }
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return None
            
    def save_model(self, model_path):
        """Save the trained model to disk."""
        try:
            joblib.dump(self.model, model_path)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
            
    def load_model(self, model_path):
        """Load a trained model from disk."""
        try:
            self.model = joblib.load(model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False