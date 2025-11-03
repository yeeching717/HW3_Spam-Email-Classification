from pathlib import Path
import io
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
from sklearn.model_selection import train_test_split # 匯入 train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parent.parent




@st.cache_resource
def load_artifacts():
    """
    載入快取的模型和向量化器。
    """
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
    """
    預測單一訊息。
    """
    if model is None or vec is None:
        return None, None
    X = vec.transform([text])
    
    try:
        probs = model.predict_proba(X)[0]
        # 使用 session_state 中的閾值
        threshold = st.session_state.get('decision_threshold', 0.5)
        # 假設 probs[1] 是 'spam' 的機率
        pred = 'spam' if probs[1] >= threshold else 'ham'
        prob = float(probs[1]) # 回傳 'spam' 的機率
        return pred, prob
    except Exception as e:
        st.error(f"預測時發生錯誤: {e}")
        return None, None


def batch_predict_df(model, vec, df: pd.DataFrame, text_col: str):
    """
    對整個 DataFrame 進行批次預測。
    """
    texts = df[text_col].astype(str).tolist()
    X = vec.transform(texts)
    
    try:
        # 獲取 'spam' (正面類別) 的機率
        probs = model.predict_proba(X)[:, 1]
        # 使用 session_state 中的閾值
        threshold = st.session_state.get('decision_threshold', 0.5)
        preds = (probs >= threshold).astype(int)
        out = df.copy()
        out["prediction"] = ["spam" if p else "ham" for p in preds]
        out["probability_spam"] = probs
        return out
    except Exception:
        st.error("無法進行批次預測。請檢查模型是否支援 predict_proba。")
        return None


def get_top_tokens(model, vectorizer, n_top=15):
    """根據模型係數獲取每個類別的 Top N 關鍵詞。"""
    if not hasattr(model, 'coef_'):
        return None, None
    
    feature_names = vectorizer.get_feature_names_out()
    # 假設是二元分類，且模型是 Logistic Regression 之類的線性模型
    if model.coef_.ndim > 1:
        coef = model.coef_[0]
    else:
        coef = model.coef_
        
    
    # 獲取 top positive (spam) 和 negative (ham) 係數的索引
    top_spam_idx = np.argsort(coef)[-n_top:]
    top_ham_idx = np.argsort(coef)[:n_top]
    
    # 獲取關鍵詞及其係數
    spam_tokens = [(feature_names[i], coef[i]) for i in top_spam_idx]
    ham_tokens = [(feature_names[i], coef[i]) for i in top_ham_idx]
    
    return spam_tokens[::-1], ham_tokens # 反轉 'ham' 列表以正確顯示
    

def show_top_tokens(model, vec):
    """顯示 Top N 關鍵詞的視覺化圖表。"""
    st.header("Top Tokens by Class (模型關鍵詞)")
    
    spam_tokens, ham_tokens = get_top_tokens(model, vec)
    if spam_tokens is None or ham_tokens is None:
        st.warning("無法從模型中提取特徵重要性 (feature importance)。")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 'Not Spam' Tokens")
        if not ham_tokens:
            st.write("沒有 'Not Spam' 關鍵詞可顯示。")
            return
            
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        tokens, coeffs = zip(*ham_tokens)
        colors = ['#2ecc71' for c in coeffs] # 'Not Spam' 應為負係數
        ax1.barh(range(len(tokens)), coeffs, color=colors)
        ax1.set_yticks(range(len(tokens)))
        ax1.set_yticklabels(tokens)
        ax1.set_xlabel("Coefficient Value")
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Top 'Spam' Tokens")
        if not spam_tokens:
            st.write("沒有 'Spam' 關鍵詞可顯示。")
            return

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        tokens, coeffs = zip(*spam_tokens)
        colors = ['#e74c3c' for c in coeffs] # 'Spam' 應為正係數
        ax2.barh(range(len(tokens)), coeffs, color=colors)
        ax2.set_yticks(range(len(tokens)))
        ax2.set_yticklabels(tokens)
        ax2.set_xlabel("Coefficient Value")
        plt.tight_layout()
        st.pyplot(fig2)


# --- 重大修改開始 ---

@st.cache_data
def get_evaluation_data(df, _vec, _model, text_col, label_col, test_size, random_seed):
    """
    這是一個快取函式。
    只有當 test_size 或 random_seed 改變時，才會重新執行。
    它會執行資料分割和模型預測（最耗時的步驟）。
    
    注意：我們傳入 _vec 和 _model (帶底線) 是為了告訴 Streamlit
    "我們知道這些參數不會變，不用檢查它們的變化"，
    這能提高快取的效率。
    """
    st.write(f"快取失效，正在以 Test Size={test_size}, Seed={random_seed} 重新計算預測結果...")
    
    X = _vec.transform(df[text_col].astype(str).tolist())
    y_true = df[label_col].values
    
    # 根據側邊欄的值來分割資料
    _, X_test, _, y_test = train_test_split(
        X, y_true, 
        test_size=test_size, 
        random_state=random_seed,
        stratify=y_true # 建議進行分層抽樣
    )
    
    # 在 *測試集* 上獲取預測機率
    y_pred_proba = _model.predict_proba(X_test)[:, 1] # 獲取 'spam' 的機率
    
    return y_test, y_pred_proba

def show_metrics_panel(model, vec, test_size, random_seed, decision_threshold):
    """
    顯示所有模型性能指標。
    現在這個函式會接收 test_size, random_seed, 和 decision_threshold。
    """
    processed = REPO_ROOT / "datasets" / "processed" / "sms_spam_clean.csv"
    if not processed.exists():
        st.info("找不到 `datasets/processed/sms_spam_clean.csv` — 無法顯示指標。")
        return
    
    # 載入 *整個* 資料集
    df = pd.read_csv(processed)
    if "text_clean" not in df.columns or "label" not in df.columns:
        st.warning("處理過的資料集缺少 'text_clean' 或 'label' 欄位。")
        return
        
    # ---
    # 呼叫快取函式來獲取測試集和預測機率
    # 只有 test_size 或 random_seed 改變時，這行才會真正重新計算
    # ---
    y_true, y_pred_proba = get_evaluation_data(
        df, vec, model, "text_clean", "label", test_size, random_seed
    )
    
    # ---
    # 以下的所有計算都是基於快取的 y_true 和 y_pred_proba
    # 這些計算 *會* 隨著 decision_threshold 的改變而立即重新計算，
    # 因為它們不在快取函式內部。
    # ---
    
    # 根據 *目前* 的閾值進行二元預測
    y_pred = (y_pred_proba >= decision_threshold).astype(int)
    
    st.subheader("Model Performance (on Test Set)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 混淆矩陣
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            ax=ax_cm,
            cmap='Blues',
            display_labels=['Ham', 'Spam']
        )
        ax_cm.set_title(f'Confusion Matrix\n(Threshold={decision_threshold:.2f})')
        st.pyplot(fig_cm)
        
    with col2:
        # ROC 曲線 (這個不依賴於 decision_threshold)
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
    
    # 分類報告
    st.subheader("Classification Report")
    report_str = classification_report(y_true, y_pred, target_names=['Ham', 'Spam'])
    st.text(report_str)
    # report = classification_report(y_true, y_pred, output_dict=True, target_names=['Ham', 'Spam'])
    # st.write(pd.DataFrame(report).transpose())
    
    # 閾值掃描分析 (Threshold Sweep Analysis)
    # 這裡的 y_true 和 y_pred_proba 都是正確的 (來自測試集)
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

# --- 重大修改結束 ---


def main():
    st.set_page_config(
        page_title="Spam Email Classification",
        page_icon="✉️",
        layout="wide"
    )
    
    st.sidebar.title("Spam Email Classification")
    
    model, vec = load_artifacts()
    if model is None or vec is None:
        st.error("找不到模型或向量化器 `models/`。請執行 `python src/train.py` 來產生。")
        return
        
    st.sidebar.subheader("Model Parameters")
    
    # --- 修改：不再將值存入 session_state (除非其他函式需要) ---
    # 而是將它們當作變數傳遞
    
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
    
    decision_threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Probability threshold for classification"
    )
    
    # 為了讓 predict_message 和 batch_predict_df 能運作，我們 *仍然* 需要將 threshold 存入 session_state
    st.session_state['decision_threshold'] = decision_threshold
    
    
    st.title("Spam Email Classification")
    
    # 顯示 Top Tokens (這和參數無關)
    show_top_tokens(model, vec)
    
    # --- 修改：傳入所有必要的參數 ---
    show_metrics_panel(model, vec, test_size, random_seed, decision_threshold)
    
    # --- 以下的 UI 保持不變 ---
    
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
                # 確保 prob 是 'spam' 的機率
                if pred == 'spam':
                    st.write(f"Spam Probability: **{prob:.2%}**")
                else:
                    st.write(f"Spam Probability: **{prob:.2%}**")
        else:
            st.warning("Please enter a message to classify.")
            
    st.header("Batch Classification")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.subheader("Data Column Selection")
            text_col = st.sidebar.selectbox(
                "Select Text Column",
                options=df.columns.tolist(),
                index=0,
                key="batch_text_col" # 新增 key 避免衝突
            )
            label_col = st.sidebar.selectbox(
                "Select Label Column",
                options=df.columns.tolist(),
                index=min(1, len(df.columns)-1),
                key="batch_label_col" # 新增 key 避免衝突
            )
            
            if st.button("Run Batch Classification"):
                results = batch_predict_df(model, vec, df, text_col)
                if results is not None:
                    st.write("Preview of classification results:")
                    st.dataframe(results.head())
                    
                    csv = results.to_csv(index=False).encode('utf-8')
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
