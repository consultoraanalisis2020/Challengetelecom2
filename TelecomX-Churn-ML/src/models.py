
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMB_OK = True
except Exception:
    IMB_OK = False

def make_preprocessor(cat_cols, num_cols, scale_numeric=False):
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if num_cols:
        if scale_numeric:
            transformers.append(("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols))
        else:
            transformers.append(("num", "passthrough", num_cols))
    return ColumnTransformer(transformers=transformers)

def metrics_dict(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            pass
    return out

def train_evaluate_models(X, y, test_size=0.3, random_state=42, use_smote=False):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Column groups
    from prep import get_cat_num_cols
    cat_cols, num_cols = get_cat_num_cols(X)

    results = {}
    feature_names = {}

    # Helper to get feature names after ColumnTransformer
    def _feat_names(ct):
        try:
            return ct.get_feature_names_out()
        except Exception:
            names = []
            for name, trans, cols in ct.transformers_:
                if name == "remainder" or trans == "drop":
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    base = trans.get_feature_names_out(cols)
                else:
                    base = np.array(cols, dtype=object)
                names.extend(list(base))
            return np.array(names, dtype=object)

    # Logistic Regression (scaled)
    preproc_lr = make_preprocessor(cat_cols, num_cols, scale_numeric=True)
    clf_lr = LogisticRegression(max_iter=1000)
    if use_smote and IMB_OK:
        pipe_lr = ImbPipeline([("prep", preproc_lr), ("smote", SMOTE(random_state=42)), ("clf", clf_lr)])
    else:
        pipe_lr = Pipeline([("prep", preproc_lr), ("clf", clf_lr)])
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    y_proba = getattr(pipe_lr, "predict_proba", lambda X: None)(X_test)
    y_proba = y_proba[:,1] if y_proba is not None else None
    results["LogisticRegression"] = metrics_dict(y_test, y_pred, y_proba)
    feature_names["LogisticRegression"] = _feat_names(pipe_lr.named_steps["prep"])

    # KNN (scaled)
    preproc_knn = make_preprocessor(cat_cols, num_cols, scale_numeric=True)
    clf_knn = KNeighborsClassifier(n_neighbors=7)
    if use_smote and IMB_OK:
        pipe_knn = ImbPipeline([("prep", preproc_knn), ("smote", SMOTE(random_state=42)), ("clf", clf_knn)])
    else:
        pipe_knn = Pipeline([("prep", preproc_knn), ("clf", clf_knn)])
    pipe_knn.fit(X_train, y_train)
    y_pred = pipe_knn.predict(X_test)
    results["KNN"] = metrics_dict(y_test, y_pred)

    # RandomForest (no scaling)
    preproc_rf = make_preprocessor(cat_cols, num_cols, scale_numeric=False)
    clf_rf = RandomForestClassifier(n_estimators=300, random_state=42)
    pipe_rf = Pipeline([("prep", preproc_rf), ("clf", clf_rf)])
    pipe_rf.fit(X_train, y_train)
    y_pred = pipe_rf.predict(X_test)
    y_proba = getattr(pipe_rf, "predict_proba")(X_test)[:,1]
    results["RandomForest"] = metrics_dict(y_test, y_pred, y_proba)
    feature_names["RandomForest"] = _feat_names(pipe_rf.named_steps["prep"])

    models = {
        "LogisticRegression": pipe_lr,
        "KNN": pipe_knn,
        "RandomForest": pipe_rf,
    }

    return {
        "results": results,
        "models": models,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }
