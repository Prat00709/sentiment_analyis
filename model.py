"""
Streamlit-ready training module.
This file wraps the original training logic into run_training(data_path=None, models_dir="models").
Original script content derived from the uploaded file.
"""

# Execute original content into namespace
_orig = {}
exec('#!/usr/bin/env python\n# coding: utf-8\n\n# In[1]:\n\n\n# Run this once if packages are missing\nget_ipython().system(\'pip install scikit-learn pandas numpy joblib flask gunicorn nltk matplotlib seaborn\')\nget_ipython().system(\'python -m nltk.downloader punkt stopwords wordnet omw-1.4\')\n\n\n# In[1]:\n\n\nimport os\nimport pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report\nimport joblib\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nsns.set(style="whitegrid")\nRANDOM_STATE = 42\n\n\n# In[2]:\n\n\n# Try possible cleaned filenames\ncandidates = [\n    "samsung_a35_cleaned_step1_2_binary.csv",\n    "iphone_13_cleaned_step1_2_binary.csv",\n    "combined_cleaned_reviews.csv"\n]\n\ndf = None\nfor f in candidates:\n    if os.path.exists(f):\n        print("Loading", f)\n        df = pd.read_csv(f)\n        break\n\nif df is None:\n    raise FileNotFoundError(\n        "No cleaned binary CSV found. Place one of the expected filenames in the notebook folder, "\n        "or update the `candidates` list with your file name."\n    )\n\n# We expect columns: \'text_clean\' OR \'review_text\', and \'sentiment_binary\' or \'sentiment\'\ntext_col = None\nfor c in [\'text_clean\', \'review_text\', \'review_text_clean\', \'review_text_processed\']:\n    if c in df.columns:\n        text_col = c\n        break\nif text_col is None:\n    raise ValueError("No text column found. Expected one of: text_clean, review_text, review_text_clean, review_text_processed")\n\nlabel_col = None\nfor c in [\'sentiment_binary\', \'sentiment\', \'label\']:\n    if c in df.columns:\n        label_col = c\n        break\nif label_col is None:\n    # Try to detect 0/1 column\n    for c in df.columns:\n        if df[c].dropna().isin([0,1]).all():\n            label_col = c\n            break\n\nif label_col is None:\n    raise ValueError("No binary label column found. Expected \'sentiment_binary\' (0/1).")\n\n# Keep only rows with valid text and labels\ndf = df[[text_col, label_col]].dropna().copy()\ndf.columns = [\'text\', \'label\']\ndf[\'label\'] = df[\'label\'].astype(int)\n\nprint("Dataset loaded. Rows:", len(df))\ndf.head()\n\n\n# In[3]:\n\n\n# 80-20 split (change to 70-30 if you prefer)\nX_train, X_test, y_train, y_test = train_test_split(\n    df[\'text\'], df[\'label\'], test_size=0.2, stratify=df[\'label\'], random_state=RANDOM_STATE\n)\nprint("Train:", len(X_train), "Test:", len(X_test))\n\n\n# In[4]:\n\n\n# We\'ll build pipelines for each model using TF-IDF\ntfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words=\'english\')\n\npipelines = {\n    "LogisticRegression": Pipeline([(\'tfidf\', tfidf), (\'clf\', LogisticRegression(max_iter=200, random_state=RANDOM_STATE))]),\n    "RandomForest": Pipeline([(\'tfidf\', tfidf), (\'clf\', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))]),\n    "DecisionTree": Pipeline([(\'tfidf\', tfidf), (\'clf\', DecisionTreeClassifier(random_state=RANDOM_STATE))]),\n    "NaiveBayes": Pipeline([(\'tfidf\', tfidf), (\'clf\', MultinomialNB())])\n}\n\n\n# In[5]:\n\n\nresults = {}\nfor name, pipe in pipelines.items():\n    print("\\nTraining:", name)\n    pipe.fit(X_train, y_train)\n    y_pred = pipe.predict(X_test)\n    # Some classifiers (RF) can give probabilities; handle safely for ROC AUC\n    try:\n        y_prob = pipe.predict_proba(X_test)[:,1]\n        auc = roc_auc_score(y_test, y_prob)\n    except Exception:\n        auc = None\n\n    acc = accuracy_score(y_test, y_pred)\n    prec = precision_score(y_test, y_pred, zero_division=0)\n    rec = recall_score(y_test, y_pred, zero_division=0)\n    f1 = f1_score(y_test, y_pred, zero_division=0)\n    cm = confusion_matrix(y_test, y_pred)\n\n    results[name] = {\n        \'model\': pipe,\n        \'accuracy\': acc,\n        \'precision\': prec,\n        \'recall\': rec,\n        \'f1\': f1,\n        \'roc_auc\': auc,\n        \'confusion_matrix\': cm,\n        \'classification_report\': classification_report(y_test, y_pred, zero_division=0)\n    }\n    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  AUC: {auc}")\n\n\n# In[6]:\n\n\nsummary = []\nfor name, r in results.items():\n    summary.append({\n        \'model\': name,\n        \'accuracy\': r[\'accuracy\'],\n        \'precision\': r[\'precision\'],\n        \'recall\': r[\'recall\'],\n        \'f1\': r[\'f1\'],\n        \'roc_auc\': r[\'roc_auc\']\n    })\nsummary_df = pd.DataFrame(summary).sort_values(\'f1\', ascending=False).reset_index(drop=True)\ndisplay(summary_df)\n\nbest_name = summary_df.loc[0, \'model\']\nprint("Best model by F1:", best_name)\nbest_model = results[best_name][\'model\']\n\n# Save best model\nos.makedirs("models", exist_ok=True)\njoblib.dump(best_model, f"models/{best_name}_tfidf_model.joblib")\nprint("Saved best model to:", f"models/{best_name}_tfidf_model.joblib")\n\n\n# In[7]:\n\n\nimport matplotlib.pyplot as plt\nfrom sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay\n\nr = results[best_name]\ny_pred = r[\'model\'].predict(X_test)\nprint("Classification report:\\n", r[\'classification_report\'])\n\n# confusion matrix\ncm = r[\'confusion_matrix\']\ndisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neg","pos"])\ndisp.plot(cmap=\'Blues\')\nplt.title(f"{best_name} - Confusion Matrix")\nplt.show()\n\n# ROC curve if probability available\ntry:\n    y_prob = r[\'model\'].predict_proba(X_test)[:,1]\n    RocCurveDisplay.from_predictions(y_test, y_prob)\n    plt.title(f"{best_name} - ROC Curve")\n    plt.show()\nexcept Exception:\n    print("Model does not support probability predictions (no ROC).")\n\n\n# In[8]:\n\n\ny_pred = best_model.predict(X_test)\ntry:\n    y_prob = best_model.predict_proba(X_test)[:,1]\nexcept:\n    y_prob = [None]*len(y_pred)\n\nout_df = pd.DataFrame({\n    \'text\': X_test,\n    \'label_true\': y_test,\n    \'label_pred\': y_pred,\n    \'prob_pos\': y_prob\n})\nout_df.to_csv("models/test_predictions.csv", index=False, encoding=\'utf-8\')\n\nmetrics = summary_df\nmetrics.to_csv("models/models_summary.csv", index=False)\nprint("Saved test predictions and summary to models/ folder.")\n\n\n# In[ ]:\n\n\n\n\n\n# In[ ]:\n\n\n\n\n\n# # \n\n# In[ ]:\n\n\n\n\n', _orig)


import os
import pandas as pd
import joblib

def run_training(data_path: str = None, models_dir: str = "models"):
    models_dir = str(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    saved = []

    candidates = _orig.get('candidates', None)
    df = None
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        if candidates:
            for f in candidates:
                if os.path.exists(f):
                    df = pd.read_csv(f); break

    if df is None:
        raise FileNotFoundError("No cleaned binary CSV found. Place one of the expected filenames in the repo or provide data_path.")

    text_col = None
    for c in ['text_clean','review_text','review_text_clean','review_text_processed']:
        if c in df.columns:
            text_col = c; break
    label_col = None
    for c in ['sentiment_binary','sentiment','label']:
        if c in df.columns:
            label_col = c; break
    if label_col is None:
        for c in df.columns:
            try:
                vals = df[c].dropna().unique()
                if set([int(v) for v in vals if str(v).strip()!='']) <= set([0,1]):
                    label_col = c; break
            except Exception:
                continue
    if text_col is None or label_col is None:
        raise ValueError("Required text/label columns not found in dataset.")

    df = df[[text_col,label_col]].dropna().copy()
    df.columns = ['text','label']
    df['label'] = df['label'].astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    pipelines = _orig.get('pipelines', None)
    if pipelines is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
        pipelines = {
            "LogisticRegression": Pipeline([('tfidf', tfidf), ('clf', LogisticRegression(max_iter=200, random_state=42))])
        }

    results = {}
    for name, pipe in pipelines.items():
        print(f"Training: {name}")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
            try:
                y_prob = pipe.predict_proba(X_test)[:,1]
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc = None
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        except Exception:
            acc = prec = rec = f1 = auc = None
        results[name] = {'model': pipe, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc}

    best = None
    best_name = None
    for name, r in results.items():
        if best is None or (r['f1'] is not None and r['f1'] > (best.get('f1') or 0)):
            best = r; best_name = name

    if best is None:
        for name, r in results.items():
            path = os.path.join(models_dir, f"{name}_model.joblib")
            joblib.dump(r['model'], path)
            saved.append(path)
    else:
        path = os.path.join(models_dir, f"{best_name}_tfidf_model.joblib")
        joblib.dump(best['model'], path)
        saved.append(path)

    try:
        import pandas as pd
        summary = []
        for name, r in results.items():
            summary.append({'model': name, 'accuracy': r['accuracy'], 'precision': r['precision'], 'recall': r['recall'], 'f1': r['f1'], 'roc_auc': r['roc_auc']})
        pd.DataFrame(summary).to_csv(os.path.join(models_dir, 'models_summary.csv'), index=False)
        saved.append(os.path.join(models_dir, 'models_summary.csv'))
    except Exception:
        pass

    return saved

if __name__ == "__main__":
    print("Running training (wrapper). Models saved to models/")
    print(run_training(models_dir='models'))
