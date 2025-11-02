"""
Data processing utilities for spam classification.
"""
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import requests
import joblib

class DataProcessor:
    def __init__(self, random_state=42, max_features=5000, ngram_range=(1,1), min_df=1, sublinear_tf=False, stop_words='english'):
        """DataProcessor with configurable TF-IDF settings.

        Args:
            random_state: seed for train/test split
            max_features: max features for TF-IDF
            ngram_range: tuple for ngram range
            min_df: minimum document frequency for TF-IDF
            sublinear_tf: whether to apply sublinear tf scaling
            stop_words: stop words setting for TF-IDF
        """
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            stop_words=stop_words
        )
        self.is_fitted = False
        
    def download_dataset(self, url, save_path):
        """Download the dataset from the given URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
            
    def load_data(self, file_path):
        """Load and preprocess the spam dataset."""
        try:
            # Load the data (assuming CSV with two columns: label and text)
            df = pd.read_csv(file_path, names=['label', 'text'])

            # Minimal cleaning: lowercase and strip
            df['text_clean'] = df['text'].astype(str).str.lower().str.strip()

            # Convert labels to binary (0 for ham, 1 for spam)
            df['label'] = (df['label'] == 'spam').astype(int)

            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
    def prepare_data(self, df, test_size=0.2):
        """Prepare data for training and testing."""
        try:
            # Split features and target
            # Use cleaned text when available
            text_col = 'text_clean' if 'text_clean' in df.columns else 'text'
            X = self.vectorizer.fit_transform(df[text_col])
            self.is_fitted = True
            y = df['label'].values
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y
            )
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None, None, None, None
            
    def preprocess_text(self, text):
        """Preprocess a single text input for prediction."""
        try:
            if not self.is_fitted:
                raise ValueError("The TF-IDF vectorizer must be fitted before use")
            return self.vectorizer.transform([text])
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return None

    def save_vectorizer(self, path):
        """Save the fitted vectorizer to disk."""
        try:
            joblib.dump(self.vectorizer, path)
            return True
        except Exception as e:
            print(f"Error saving vectorizer: {e}")
            return False

    def load_vectorizer(self, path):
        """Load a fitted vectorizer from disk."""
        try:
            self.vectorizer = joblib.load(path)
            self.is_fitted = True
            return True
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            return False

    def save_processed(self, df, out_path):
        """Save processed dataframe to CSV for reproducibility."""
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # keep label and text_clean (or text)
            text_col = 'text_clean' if 'text_clean' in df.columns else 'text'
            df_to_save = df.copy()
            # Ensure columns exist
            if 'text_clean' not in df_to_save.columns:
                df_to_save['text_clean'] = df_to_save[text_col].astype(str).str.lower().str.strip()
            df_to_save.to_csv(out_path, index=False)
            return True
        except Exception as e:
            print(f"Error saving processed csv: {e}")
            return False