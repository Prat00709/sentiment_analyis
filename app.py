# Streamlit app that uses the patched adapters to run preprocessing & training inside Streamlit.
import streamlit as st
from pathlib import Path
import pandas as pd
import importlib.util, sys

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("Sentiment Dashboard (Streamlit)")

# Paths to adapters (assumes repo root)
PREPROCESS_ADAPTER = Path(__file__).parent / "data_collection_and_processing_patched.py"
TRAIN_ADAPTER = Path(__file__).parent / "model_patched.py"
OUT_DIR = Path(__file__).parent / "dashboard_outputs"
MODELS_DIR = Path(__file__).parent / "models"
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod

col1, col2 = st.columns(2)

with col1:
    st.header("Preprocessing")
    st.write("This calls `run_preprocessing(save_folder)` from the adapter which in turn runs your original script.")
    if st.button("Run preprocessing"):
        try:
            mod = load_adapter(PREPROCESS_ADAPTER)
            if hasattr(mod, "run_preprocessing"):
                new_files = mod.run_preprocessing(save_folder=str(OUT_DIR))
                st.success("Preprocessing finished.")
                st.write("New CSVs:", new_files)
                for f in new_files:
                    st.write(f)
            else:
                st.error("Adapter missing run_preprocessing()")
        except Exception as e:
            st.error(f"Error running preprocessing: {e}")

with col2:
    st.header("Training")
    st.write("This calls `run_training(data_path, models_dir)` from the adapter which in turn runs your original model script.")
    if st.button("Run training"):
        try:
            mod = load_adapter(TRAIN_ADAPTER)
            if hasattr(mod, "run_training"):
                new_models = mod.run_training(data_path=None, models_dir=str(MODELS_DIR))
                st.success("Training finished.")
                st.write("New model artifacts:", new_models)
            else:
                st.error("Adapter missing run_training()")
        except Exception as e:
            st.error(f"Error running training: {e}")

st.markdown('---')
st.header("Inspect outputs")
csvs = sorted(OUT_DIR.glob('*.csv'))
if csvs:
    sel = st.selectbox("Choose CSV to view", [c.name for c in csvs])
    if sel:
        df = pd.read_csv(OUT_DIR / sel)
        st.write(df.head())
else:
    st.info("No CSVs in dashboard_outputs/ yet. Run preprocessing or upload a CSV.")

models = sorted(MODELS_DIR.glob('*'))
if models:
    st.write("Model files available:")
    for m in models:
        st.write(m.name)
