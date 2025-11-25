import streamlit as st
from pathlib import Path
import pandas as pd

# Import your real scripts directly (these files DO exist)
import data_collection_and_processing_ as pp
import model as mdl

# Output folders
OUT_DIR = Path("dashboard_outputs")
MODELS_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("Sentiment Analysis Dashboard")

col1, col2 = st.columns(2)

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
with col1:
    st.header("Preprocessing")
    st.write("Runs run_preprocessing() inside data_collection_and_processing.py")

    if st.button("Run Preprocessing"):
        try:
            generated = pp.run_preprocessing(save_folder=str(OUT_DIR))
            st.success("Preprocessing Completed")
            st.write("Generated files:", generated)
        except Exception as e:
            st.error(f"Error running preprocessing: {e}")

# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------
with col2:
    st.header("Training")
    st.write("Runs run_training() inside model.py")

    if st.button("Run Training"):
        try:
            saved = mdl.run_training(data_path=None, models_dir=str(MODELS_DIR))
            st.success("Training Completed")
            st.write("Saved models:", saved)
        except Exception as e:
            st.error(f"Error running training: {e}")

# ---------------------------------------------------
# VIEW CSV OUTPUTS
# ---------------------------------------------------
st.markdown("---")
st.header("View Processed CSV Files")

csv_files = list(OUT_DIR.glob("*.csv"))

if csv_files:
    selected = st.selectbox("Choose a CSV to view", [c.name for c in csv_files])
    df = pd.read_csv(OUT_DIR / selected)
    st.dataframe(df)
else:
    st.info("No CSV files found. Run preprocessing first.")
