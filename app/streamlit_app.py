from pathlib import Path
import io
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


REPO_ROOT = Path(__file__).resolve().parent.parent




@st.cache_resource
def load_artifacts():
    """Load base model and vectorizer."""
    model_path = REPO_ROOT / "models" / "spam_classifier.joblib"
    vec_path = REPO_ROOT / "models" / "tfidf_vectorizer.joblib"
    model = None
    vec = None
    if model_path.exists() and vec_path.exists():
        try:
            model = joblib.load(model_path)
            vec = joblib.load(vec_path)
        except Exception:
            model = None
            vec = None
    return model, vec


@st.cache_resource
def train_model_with_params(test_size: float, random_seed: int):
    """Train model with user-specified parameters."""
    from src.data_processor import DataProcessor
    from src.spam_classifier import SpamClassifier
    
    # Initialize with user parameters
    data_processor = DataProcessor(
        random_state=random_seed,
        ngram_range=(1,2),
        min_df=2,
        sublinear_tf=True
    )
    
    classifier = SpamClassifier(
        penalty='l1',
        C=10.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=random_seed
    )
    
    # Load and prepare data
    data_path = REPO_ROOT / "data" / "spam_dataset.csv"
    if not data_path.exists():
        st.error("Dataset not found. Please run setup scripts first.")
        return None, None
        
    df = data_processor.load_data(data_path)
    if df is None:
        st.error("Failed to load dataset.")
        return None, None
        
    # Prepare data with user-specified test_size
    X_train, X_test, y_train, y_test = data_processor.prepare_data(df, test_size=test_size)
    if X_train is None:
        st.error("Failed to prepare data.")
        return None, None
        
    # Train model
    success = classifier.train(X_train, y_train)
    if not success:
        st.error("Failed to train model.")
        return None, None
        
    return classifier.model, data_processor.vectorizer


def predict_message(model, vec, text: str):
    if model is None or vec is None:
        return None, None
    X = vec.transform([text])
    
    try:
        probs = model.predict_proba(X)[0]
        prob = float(probs.max())
        # Use the sidebar threshold for prediction
        threshold = st.session_state.get('decision_threshold', 0.5)
        pred = 'spam' if probs[1] >= threshold else 'ham'
        return pred, prob
    except Exception:
        return None, None


def batch_predict_df(model, vec, df: pd.DataFrame, text_col: str):
    texts = df[text_col].astype(str).tolist()
    X = vec.transform(texts)
    
    try:
        # Get probabilities for positive class
        probs = model.predict_proba(X)[:, 1]
        # Use threshold from session state
        threshold = st.session_state.get('decision_threshold', 0.5)
        preds = (probs >= threshold).astype(int)
        out = df.copy()
        out["prediction"] = ["spam" if p else "ham" for p in preds]
        out["probability"] = probs
        return out
    except Exception:
        st.error("Failed to make predictions. Check if model supports predict_proba.")
        return None


def get_top_tokens(model, vectorizer, n_top=15):
    """Get top N tokens for each class based on model coefficients."""
    if not hasattr(model, 'coef_'):
        return None, None
    
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]  # For binary classification
    
    # Get indices of top positive (spam) and negative (non-spam) coefficients
    top_spam_idx = np.argsort(coef)[-n_top:]
    top_ham_idx = np.argsort(coef)[:n_top]
    
    # Get tokens and their coefficients
    spam_tokens = [(feature_names[i], coef[i]) for i in top_spam_idx]
    ham_tokens = [(feature_names[i], coef[i]) for i in top_ham_idx]
    
    return spam_tokens[::-1], ham_tokens[::-1]  # Reverse to get descending order


def show_top_tokens(model, vec):
    """Display top tokens visualization."""
    st.header("Top Tokens by Class")
    
    spam_tokens, ham_tokens = get_top_tokens(model, vec)
    if spam_tokens is None or ham_tokens is None:
        st.warning("Could not extract feature importance from the model.")
        return
    
    # Use columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 'Not Spam' Tokens")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        tokens, coeffs = zip(*ham_tokens)
        colors = ['#2ecc71' if c < 0 else '#e74c3c' for c in coeffs]
        ax1.barh(range(len(tokens)), coeffs, color=colors)
        ax1.set_yticks(range(len(tokens)))
        ax1.set_yticklabels(tokens)
        ax1.set_xlabel("Coefficient Value")
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Top 'Spam' Tokens")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        tokens, coeffs = zip(*spam_tokens)
        colors = ['#2ecc71' if c < 0 else '#e74c3c' for c in coeffs]
        ax2.barh(range(len(tokens)), coeffs, color=colors)
        ax2.set_yticks(range(len(tokens)))
        ax2.set_yticklabels(tokens)
        ax2.set_xlabel("Coefficient Value")
        plt.tight_layout()
        st.pyplot(fig2)


def get_model_predictions(model, X):
    """Get both binary predictions and probabilities."""
    y_pred_proba = model.predict_proba(X)
    return y_pred_proba[:, 1]  # Return probabilities for positive class

def show_metrics_panel(model, vec):
    processed = REPO_ROOT / "datasets" / "processed" / "sms_spam_clean.csv"
    if not processed.exists():
        st.info("Processed dataset not found — cannot show metrics. Run `python src/train.py` to create it.")
        return
    
    df = pd.read_csv(processed)
    if "text_clean" not in df.columns or "label" not in df.columns:
        st.warning("Processed dataset missing expected columns: 'text_clean' and 'label'.")
        return
        
    X = vec.transform(df["text_clean"].astype(str).tolist())
    y_true = df["label"].values
    
    # Get probabilities for positive class
    y_pred_proba = get_model_predictions(model, X)
    
    # Get current threshold from session state
    threshold = st.session_state.get('decision_threshold', 0.5)
    
    # Make predictions using current threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    st.subheader("Model Performance")
    
    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            ax=ax_cm,
            cmap='Blues',
            display_labels=['Ham', 'Spam']
        )
        ax_cm.set_title(f'Confusion Matrix\n(threshold={threshold:.2f})')
        st.pyplot(fig_cm)
        
    with col2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())
    
    # Threshold Sweep Analysis
    st.subheader("Threshold Sweep Analysis")
    thresholds = np.arange(0.1, 1.0, 0.1)
    sweep_results = []
    
    for t in thresholds:
        y_pred_t = (y_pred_proba >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_t, average='binary', zero_division=0
        )
        sweep_results.append({
            'Threshold': f'{t:.1f}',
            'Precision': f'{precision:.3f}',
            'Recall': f'{recall:.3f}',
            'F1-Score': f'{f1:.3f}'
        })
    
    sweep_df = pd.DataFrame(sweep_results)
    st.dataframe(
        sweep_df.style.highlight_max(subset=['F1-Score'], axis=0),
        hide_index=True
    )


def demo_tab():
    st.header("Spam classifier demo")
    model, vec = load_artifacts()
    if model is None or vec is None:
        st.error("Model or vectorizer not found in `models/`. Run `python src/train.py` to produce artifacts.")
        return
    
    # Show top tokens visualization first
    show_top_tokens(model, vec)

    st.subheader("Single message")
    sample_text = st.text_area("Enter a message to classify", value="Free entry text here")
    if st.button("Classify"):
        pred, prob = predict_message(model, vec, sample_text)
        if pred is None:
            st.error("Prediction failed — ensure model supports predict/predict_proba.")
        else:
            st.success(f"Prediction: {pred}")
            if prob is not None:
                st.info(f"Confidence: {prob:.3f}")

    st.subheader("Sample messages")
    samples = [
        "Congratulations! You've won a free ticket. Reply to claim.",
        "Are we still meeting at 10am tomorrow?",
        "URGENT: Your account will be closed unless you verify now.",
        "Free entry in 2 a weekly competition to win FA Cup finals tickets. Text WIN to 80085",
    ]
    cols = st.columns(2)
    for i, s in enumerate(samples):
        with cols[i % 2]:
            st.write(s)
            if st.button(f"Classify sample {i+1}", key=f"sample_{i}"):
                p, pr = predict_message(model, vec, s)
                st.write(f"Prediction: {p}")
                if pr is not None:
                    st.write(f"Confidence: {pr:.3f}")

    st.subheader("Batch upload (CSV)")
    st.markdown("Upload a CSV with columns for text data and labels.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            st.error("Failed to read CSV. Ensure it's a valid CSV file.")
            return
            
        # Add column selectors to sidebar
        st.sidebar.subheader("Data Column Selection")
        text_col = st.sidebar.selectbox("Select Text Column", options=df.columns.tolist(), index=0)
        label_col = st.sidebar.selectbox("Select Label Column", options=df.columns.tolist(), index=min(1, len(df.columns)-1))
        
        if st.button("Run batch prediction"):
            out = batch_predict_df(model, vec, df, text_col)
            st.dataframe(out.head(50))
            towrite = io.BytesIO()
            out.to_csv(towrite, index=False)
            towrite.seek(0)
            st.download_button("Download predictions CSV", data=towrite, file_name="predictions.csv")

    st.subheader("Model performance")
    show_metrics_panel(model, vec)


def main():
    st.set_page_config(
        page_title="Spam Email Classification",
        page_icon="✉️",
        layout="wide"
    )
    
    # Configure sidebar
    st.sidebar.title("Spam Email Classification")
    
    # Add model parameter controls to sidebar
    st.sidebar.subheader("Model Parameters")
    
    # Store parameters in session state
    test_size = st.sidebar.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Proportion of dataset to include in the test split"
    )
    
    random_seed = st.sidebar.number_input(
        "Random Seed",
        value=42,
        min_value=0,
        help="Random seed for reproducibility"
    )
    
    st.session_state['decision_threshold'] = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Probability threshold for classification"
    )
    
    # Train model with current parameters
    model, vec = train_model_with_params(test_size, random_seed)
    if model is None or vec is None:
        st.error("Failed to initialize model with the selected parameters.")
        return(
        "Random Seed",
        value=42,
        min_value=0,
        help="Random seed for reproducibility"
    )
    
    st.session_state['decision_threshold'] = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Probability threshold for classification"
    )
    
    # Main content area
    st.title("Spam Email Classification")
    
    # Show visualizations first
    show_top_tokens(model, vec)
    show_metrics_panel(model, vec)
    
    # Single message classification
    st.header("Single Message Classification")
    text_input = st.text_area(
        "Enter a message to classify",
        value="Enter your message here...",
        height=100
    )
    
    if st.button("Classify Message"):
        if text_input and text_input != "Enter your message here...":
            pred, prob = predict_message(model, vec, text_input)
            if pred:
                st.write(f"Classification: **{pred.upper()}**")
                st.write(f"Confidence: **{prob:.2%}**")
        else:
            st.warning("Please enter a message to classify.")
            
    # Batch classification
    st.header("Batch Classification")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.subheader("Data Column Selection")
            text_col = st.sidebar.selectbox(
                "Select Text Column",
                options=df.columns.tolist(),
                index=0
            )
            label_col = st.sidebar.selectbox(
                "Select Label Column",
                options=df.columns.tolist(),
                index=min(1, len(df.columns)-1)
            )
            
            if st.button("Run Batch Classification"):
                results = batch_predict_df(model, vec, df, text_col)
                if results is not None:
                    st.write("Preview of classification results:")
                    st.dataframe(results.head())
                    
                    # Download button for results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "spam_classification_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    main()
