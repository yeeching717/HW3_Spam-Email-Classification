from pathlib import Path
import io
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parent.parent




@st.cache_resource
def load_artifacts():
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


def predict_message(model, vec, text: str):
    if model is None or vec is None:
        return None, None
    X = vec.transform([text])
    pred = model.predict(X)[0]
    prob = None
    try:
        prob = float(model.predict_proba(X)[0].max())
    except Exception:
        prob = None
    return pred, prob


def batch_predict_df(model, vec, df: pd.DataFrame, text_col: str):
    texts = df[text_col].astype(str).tolist()
    X = vec.transform(texts)
    preds = model.predict(X)
    try:
        probs = model.predict_proba(X).max(axis=1)
    except Exception:
        probs = [None] * len(preds)
    out = df.copy()
    out["prediction"] = preds
    out["probability"] = probs
    return out


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
    y_pred = model.predict(X)
    st.subheader("Model metrics on processed dataset")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


def demo_tab():
    st.header("Spam classifier demo")
    model, vec = load_artifacts()
    if model is None or vec is None:
        st.error("Model or vectorizer not found in `models/`. Run `python src/train.py` to produce artifacts.")
        return

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
    st.set_page_config(page_title="Spam classifier demo", layout="wide")
    st.sidebar.title("Spam Email Classification")
    demo_tab()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
