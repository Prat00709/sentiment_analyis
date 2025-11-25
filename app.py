# app.py — robust loader + helpful error display
import streamlit as st
from pathlib import Path
import importlib, importlib.util, sys, traceback
import pandas as pd
import os

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("Sentiment Dashboard (robust loader)")

# Where we expect the scripts to be (relative to repo root)
EXPECTED_PREPROCESS = "data_collection_and_processing.py"
EXPECTED_MODEL = "model.py"

OUT_DIR = Path("dashboard_outputs")
MODELS_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def load_module_fallback(module_name: str, file_path: str):
    """
    Try normal import, otherwise load module from given file path.
    Returns (module, error_text). module is None on failure; error_text contains traceback.
    """
    # 1) try normal import
    try:
        module = importlib.import_module(module_name)
        return module, None
    except Exception:
        # proceed to fallback
        pass

    # 2) fallback: try loading from file path
    try:
        p = Path(file_path)
        if not p.exists():
            return None, f"File not found at path: {file_path}"
        spec = importlib.util.spec_from_file_location(module_name, str(p))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore
        return module, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, tb

# Attempt to load modules (and show diagnostics)
pp_module, pp_err = load_module_fallback("data_collection_and_processing", EXPECTED_PREPROCESS)
mdl_module, mdl_err = load_module_fallback("model", EXPECTED_MODEL)

st.sidebar.header("Diagnostics")
st.sidebar.write("Preprocessing module:")
if pp_module:
    st.sidebar.success("Loaded data_collection_and_processing_.py")
else:
    st.sidebar.error("Preprocessing module NOT loaded")
    st.sidebar.text(pp_err)

st.sidebar.write("Model module:")
if mdl_module:
    st.sidebar.success("Loaded model.py")
else:
    st.sidebar.error("Model module NOT loaded")
    st.sidebar.text(mdl_err)

col1, col2 = st.columns(2)

with col1:
    st.header("Preprocessing")
    st.write("Calls `run_preprocessing(save_folder)` from the preprocessing module.")
    if st.button("Run Preprocessing"):
        if not pp_module:
            st.error("Preprocessing module not loaded. See sidebar for details.")
        else:
            try:
                # Prefer function if exists
                if hasattr(pp_module, "run_preprocessing"):
                    out = pp_module.run_preprocessing(save_folder=str(OUT_DIR))
                elif hasattr(pp_module, "main"):
                    out = pp_module.main()
                else:
                    raise AttributeError("module has no run_preprocessing() or main()")
                st.success("Preprocessing finished")
                st.write("Generated:", out)
            except Exception:
                st.error("Preprocessing failed — see traceback below")
                st.text(traceback.format_exc())

with col2:
    st.header("Training")
    st.write("Calls `run_training(data_path, models_dir)` from the model module.")
    if st.button("Run Training"):
        if not mdl_module:
            st.error("Model module not loaded. See sidebar for details.")
        else:
            try:
                if hasattr(mdl_module, "run_training"):
                    out = mdl_module.run_training(data_path=None, models_dir=str(MODELS_DIR))
                elif hasattr(mdl_module, "train"):
                    out = mdl_module.train()
                else:
                    raise AttributeError("module has no run_training() or train()")
                st.success("Training finished")
                st.write("Saved artifacts:", out)
            except Exception:
                st.error("Training failed — see traceback below")
                st.text(traceback.format_exc())

st.markdown("---")
st.header("View processed CSVs")
csvs = sorted(OUT_DIR.glob("*.csv"))
if csvs:
    sel = st.selectbox("Choose CSV", [c.name for c in csvs])
    df = pd.read_csv(OUT_DIR / sel)
    st.dataframe(df.head(200))
else:
    st.info("No CSVs in dashboard_outputs/. Run preprocessing first or upload a CSV.")
