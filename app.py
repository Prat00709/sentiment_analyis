# app.py — robust runner that prefers direct callable but falls back to wrappers
import streamlit as st
from pathlib import Path
import importlib, importlib.util, sys, traceback, runpy, os
import pandas as pd

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("Sentiment Dashboard (robust)")

# Files
PREPROCESS_FILE = Path("data_collection_and_processing_.py")
PREPROCESS_WRAPPER = Path("data_collection_and_processing_wrapper.py")
MODEL_FILE = Path("model.py")
OUT_DIR = Path("dashboard_outputs")
MODELS_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def load_module_name_or_path(name, path):
    try:
        mod = importlib.import_module(name)
        return mod, None
    except Exception as e:
        pass
    # try loading by path
    if Path(path).exists():
        try:
            spec = importlib.util.spec_from_file_location(name, str(Path(path)))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)  # type: ignore
            return mod, None
        except Exception as e:
            return None, traceback.format_exc()
    return None, f"Neither module {name} nor file {path} found."

# Try to load preprocessing module directly; if fails, try wrapper
pp_mod, pp_err = load_module_name_or_path("data_collection_and_processing_", str(PREPROCESS_FILE))
if not pp_mod and PREPROCESS_WRAPPER.exists():
    pp_mod, pp_err = load_module_name_or_path("data_collection_and_processing_wrapper", str(PREPROCESS_WRAPPER))

# Try to load model module directly
mdl_mod, mdl_err = load_module_name_or_path("model", str(MODEL_FILE))
# Also a simple wrapper for model can be created if needed (not created here)

st.sidebar.header("Diagnostics")
if pp_mod:
    st.sidebar.success("Preprocessing module loaded")
else:
    st.sidebar.error("Preprocessing module not loaded")
    st.sidebar.text(pp_err)

if mdl_mod:
    st.sidebar.success("Model module loaded")
else:
    st.sidebar.error("Model module not loaded")
    st.sidebar.text(mdl_err)

col1, col2 = st.columns(2)

with col1:
    st.header("Preprocessing")
    if st.button("Run Preprocessing"):
        if not pp_mod:
            st.error("Preprocessing module not available. See sidebar for details.")
        else:
            try:
                if hasattr(pp_mod, "run_preprocessing"):
                    out = pp_mod.run_preprocessing(save_folder=str(OUT_DIR))
                elif hasattr(pp_mod, "main"):
                    out = pp_mod.main()
                else:
                    # as last resort, if module is the wrapper it will run the script via runpy
                    out = pp_mod.run_preprocessing(save_folder=str(OUT_DIR))
                st.success("Preprocessing finished")
                st.write("Generated:", out)
            except Exception:
                st.error("Preprocessing failed — see traceback below")
                st.text(traceback.format_exc())

with col2:
    st.header("Training")
    if st.button("Run Training"):
        if not mdl_mod:
            st.error("Model module not available. See sidebar for details.")
        else:
            try:
                if hasattr(mdl_mod, "run_training"):
                    out = mdl_mod.run_training(data_path=None, models_dir=str(MODELS_DIR))
                elif hasattr(mdl_mod, "train"):
                    out = mdl_mod.train()
                else:
                    # fallback: try executing the model script via runpy
                    cwd = os.getcwd()
                    try:
                        os.chdir(Path(MODEL_FILE).parent)
                        runpy.run_path(str(MODEL_FILE), run_name="__main__")
                        out = sorted([str(p) for p in Path(MODELS_DIR).glob("*")])
                    finally:
                        os.chdir(cwd)
                st.success("Training finished")
                st.write("Saved artifacts:", out)
            except Exception:
                st.error("Training failed — see traceback below")
                st.text(traceback.format_exc())

st.markdown("---")
st.header("View outputs")
csvs = sorted(OUT_DIR.glob("*.csv"))
if csvs:
    sel = st.selectbox("Choose CSV", [c.name for c in csvs])
    df = pd.read_csv(OUT_DIR / sel)
    st.dataframe(df.head(200))
else:
    st.info("No CSVs in dashboard_outputs/. Run preprocessing first.")
