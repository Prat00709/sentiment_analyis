# Sentiment Analysis — Streamlit Dashboard

This repository contains a simple Streamlit dashboard that integrates your preprocessing and training scripts
for a sentiment analysis project. The dashboard is designed to be easy to run locally, on Streamlit Cloud, or via Docker.

## Repository structure

```
.
├── app.py
├── data_collection_and_processing_.py      # original preprocessing script (uploaded)
├── data_collection_and_processing_patched.py  # adapter exposing run_preprocessing()
├── model.py                               # original model/training script (uploaded)
├── model_patched.py                       # adapter exposing run_training()
├── requirements.txt
├── Dockerfile
├── models/                                # trained models saved here
└── dashboard_outputs/                     # cleaned CSVs written here by preprocessing
```

## What I changed / added

- **Patched adapters**: `data_collection_and_processing_patched.py` and `model_patched.py` — these expose `run_preprocessing(save_folder)` and `run_training(data_path, models_dir)` respectively so the Streamlit `app.py` can import and call them directly without running subprocesses. They do **not** modify your original files; they will import and call functions from them if available, or execute the scripts as a fallback.
- **README**: this file — instructions below show how to run locally, with Docker, or deploy to Streamlit Cloud.

## How to run locally

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. Create virtual environment and install requirements:

```bash
python -m venv venv
# Windows:
venv\\Scripts\\activate
# mac / linux:
source venv/bin/activate

pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The dashboard will show buttons to run preprocessing and training using the patched adapters.

## Deploy to Streamlit Cloud (from GitHub)

1. Push your repo to GitHub.
2. Go to https://share.streamlit.io and connect your GitHub account.
3. Select your repository, branch, and set the main file to `app.py`.
4. Deploy — Streamlit Cloud will install `requirements.txt` and run the app. Note: file storage on Streamlit Cloud is ephemeral.

## Docker (optional)

Build and run with Docker:

```bash
docker build -t sentiment-dashboard .
docker run -p 8501:8501 sentiment-dashboard
```

Then open `http://localhost:8501`.

## Notes & troubleshooting

- If your original scripts require NLTK corpora or downloads, ensure `nltk.download()` calls run or pre-download them in the Docker image / cloud environment.
- If your scripts use interactive prompts (`input()`), remove them or guard behind `if __name__ == '__main__':` so adapters can call the functions programmatically.
- If you want me to *directly modify* the original scripts to add explicit `run_preprocessing()` and `run_training()` functions rather than using adapters, I can patch them in-place — tell me and I will update the original files.

---
