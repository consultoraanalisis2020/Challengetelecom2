
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List

ID_CANDIDATES = ["customerID","CustomerID","customer_id","id","Id","ID"]

def load_clean_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalizar nombres básicos
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().map({"yes":1,"no":0,"sí":1,"si":1})
    # Eliminar filas completamente vacías
    df = df.dropna(how="all")
    return df

def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in df.columns if c in ID_CANDIDATES]
    return df.drop(columns=cols_to_drop, errors="ignore")

def split_features_target(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    assert target_col in df.columns, f"No se encontró la columna objetivo '{target_col}'"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y

def get_cat_num_cols(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype=="object" or X[c].dtype.name=="category"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols
