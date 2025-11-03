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
# HW3 — Spam Email Classification

這個專案包含一個簡單的垃圾郵件 / 簡訊 (spam) 分類基線實作與互動式 Streamlit 展示介面。

Live demo
---------

已部署的 Streamlit 應用：

https://hw3spam-email-classification-lsa7mtewxuzdojg82caung.streamlit.app/

快速上手（Windows / PowerShell）
-------------------------------

1) 建議使用 conda 建立環境並安裝相依套件：

```powershell
conda create -p .conda python=3.9 -y
conda activate ./.conda
pip install -r requirements.txt
```

2) 若尚未產生模型與向量化器，請執行訓練腳本（會下載資料並儲存於 `models/`）：

```powershell
C:/Users/User/anaconda3/Scripts/conda.exe run -p C:\Users\User\Desktop\IoTHW3\.conda --no-capture-output python src/train.py
```

3) 啟動本機 Streamlit 應用：

```powershell
C:/Users/User/anaconda3/Scripts/conda.exe run -p C:\Users\User\Desktop\IoTHW3\.conda --no-capture-output streamlit run app/streamlit_app.py
```

主要功能
--------

- 單筆訊息分類（輸入文字並取得預測與信心水準）
- 批次上傳 CSV 並產生預測、下載結果
- 模型分析：Top tokens by class（依係數排序）、混淆矩陣、ROC 與 AUC、threshold sweep
- 可調整參數：test size、random seed、decision threshold

重要檔案
--------

- `app/streamlit_app.py` — Streamlit 應用程式
- `src/data_processor.py` — 資料處理與 TF-IDF
- `src/spam_classifier.py` — LogisticRegression 包裝類別
- `src/train.py` — 訓練腳本（會輸出 `models/*.joblib`）
- `models/` — 模型與向量化器檔案（`spam_classifier.joblib`, `tfidf_vectorizer.joblib`）
- `datasets/processed/sms_spam_clean.csv` — 處理後用於評估的資料集
- `requirements.txt` — 執行環境套件清單

部署與除錯小技巧
---------------

- 若 Streamlit 顯示「Model or vectorizer not found in `models/`」，請先執行 `src/train.py` 產生 artifacts。
- 若出現 `ModuleNotFoundError: src`，確認 `src/__init__.py` 存在，或啟動腳本將專案根目錄加入 `sys.path`。
- 若出現 StreamlitDuplicateElementId，通常是因為 UI 中使用了相同 label 的多個 widget（例如重複的 `st.sidebar.slider("Test Size")`），請檢查 `app/streamlit_app.py` 並移除重複項目。
- 部署到 Streamlit Cloud 時，請確認 `requirements.txt` 包含所有依賴，並在 Streamlit Cloud 的「Manage app」→「Logs」查看錯誤訊息。

貢獻與後續建議
----------------

- 新增單元測試（pytest）和 GitHub Actions CI
- 增加資料與輸入驗證、更多前處理步驟
- 嘗試其他模型（RandomForest, XGBoost）與更完整的超參數搜尋
