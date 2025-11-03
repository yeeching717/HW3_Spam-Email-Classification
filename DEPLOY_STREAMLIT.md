# Deploying this repo to Streamlit Cloud

This repository already contains a Streamlit app at `app/streamlit_app.py` and a `requirements.txt` with `streamlit` listed.

Follow these steps to deploy the app to Streamlit Cloud (share.streamlit.io):

1. Sign in to Streamlit Cloud
   - Visit https://share.streamlit.io and sign in with your GitHub account.

2. Create a new app
   - Click "New app" → "From a GitHub repository".
   - Select the repository: `yeeching717/HW3_Spam-Email-Classification`.
   - Branch: `main`.
   - Main file path: `app/streamlit_app.py`.

3. Configure the app (optional)
   - Streamlit Cloud will install dependencies from `requirements.txt` automatically.
   - If your app needs secrets (API keys), go to the app -> Settings -> Secrets and add them there (they will be available as `st.secrets`).

4. Deploy
   - Click "Deploy" — the first deploy will take a little longer while packages are installed.
   - After deployment completes, you will receive a public URL for the app.

5. Verify and update README
   - Open the URL and verify the proposal renders correctly.
   - Optionally add the URL to `README.md` under a "Live demo" section.

Troubleshooting
 - If the app fails to start, check the "Logs" tab in Streamlit Cloud for stack traces.
 - Ensure `requirements.txt` contains all runtime dependencies (we added `streamlit>=1.0`).
 - If you prefer pinned versions for deterministic builds, update `requirements.txt` to pin exact package versions.
 - Module import errors:
   - The app requires the `src` directory to be in the Python path
   - We've added `__init__.py` to make `src` a proper Python package
   - The app automatically adds the project root to `sys.path`
   - If still seeing import errors, try restarting the app in Streamlit Cloud

Advanced: automated deploy via GitHub Actions
 - Streamlit Cloud currently deploys via its UI when linked to the repository. For CI-based or alternative hosting, consider:
   - Using GitHub Actions to run tests and checks on push (we can add a workflow for that).
   - Using other hosting providers (Heroku, Render) if you require a different deployment mechanism.
