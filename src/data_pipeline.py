

# Data_pipeline.py  (revised)
# ------------------------------------------------------------
import sqlite3
import pandas as pd
import numpy as np
import requests
import tempfile
from sklearn.model_selection import train_test_split
from custom_messages import CustomMessages  # keep your existing logger


class DataIngestion:
    URL = "https://techassessment.blob.core.windows.net/aiap20-group-exercise-data/call_duration_modified.db"
    TABLE_NAME = "call_duration_modified"   # change if actual differs

    def __init__(self, seed: int = 42, n_bins: int = 5, stratify: bool = True, use_quantile_bins: bool = False):
        self.seed = seed
        self.n_bins = n_bins
        self.stratify = stratify
        self.use_quantile_bins = use_quantile_bins
        self.cm = CustomMessages()

        # holders
        self.df = None
        self.df_train = self.df_val = self.df_test = None
        self.df_train_fe = self.df_val_fe = self.df_test_fe = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

    # ------------------------ IO ------------------------
    def download_db(self) -> str:
        r = requests.get(self.URL, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Download failed: {r.status_code}")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.write(r.content)
        tmp.close()
        self.cm.rtm("Database downloaded.")
        return tmp.name

    def load_data(self):
        db_file = self.download_db()
        with sqlite3.connect(db_file) as conn:
            self.df = pd.read_sql(f'SELECT * FROM "{self.TABLE_NAME}"', conn)
        self.cm.rtm(f"Loaded data shape: {self.df.shape}")

    # ------------------ Cleaning & FE (full df) ------------------
    def clean_feature_eng(self, data_input: pd.DataFrame) -> pd.DataFrame:
        df = data_input.copy()

        # Drop ID
        if "SN" in df.columns:
            df = df.drop(columns=["SN"])

        # Subscription Status numeric
        if "Previous Outcome" not in df.columns:
            raise KeyError("'Previous Outcome' column missing.")
        success_mask = df["Previous Outcome"].astype(str).str.strip().str.lower().eq("success")
        df["Subscription Status"] = success_mask.astype("int8")

        # Duration clean
        dur_col = "Campaign Call Duration"
        if dur_col not in df.columns:
            raise KeyError(f"'{dur_col}' missing")
        s = pd.to_numeric(df[dur_col], errors="coerce").abs()
        s = s.fillna(s.median())
        df[dur_col] = s

        # Bin
        if self.use_quantile_bins:
            df[f"{dur_col}_bin_id"] = pd.qcut(df[dur_col], q=self.n_bins, labels=range(self.n_bins), duplicates="drop").astype("int8")
            # recreate textual bin for reference
            cats = pd.qcut(df[dur_col], q=self.n_bins, duplicates="drop")
            df[f"{dur_col}_bin"] = cats.astype(str)
        else:
            edges = np.linspace(0, df[dur_col].max(), self.n_bins + 1)
            edges[0] = 0.0
            df[f"{dur_col}_bin"] = pd.cut(df[dur_col], bins=edges, include_lowest=True)
            df[f"{dur_col}_bin_id"] = pd.cut(df[dur_col], bins=edges, labels=range(self.n_bins), include_lowest=True).astype("int8")

        # Drop raw to prevent leakage
        df = df.drop(columns=[dur_col])

        # Simple NA handling
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna("missing")

        self.cm.rtm(f"After FE: {df.shape}")
        return df

    def split_features_target(self, df: pd.DataFrame):
        target = "Campaign Call Duration_bin_id"
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found")
        X = df.drop(columns=[target])
        y = df[target]
        return X, y

    def split_data(self, df_fe: pd.DataFrame, test_size=0.1, val_size=0.1):
        """Split already-engineered dataframe so we can stratify on target."""
        X, y = self.split_features_target(df_fe)
        strat_arg = y if self.stratify else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=strat_arg
        )
        rel_val = val_size / (1 - test_size)
        strat_arg2 = y_temp if self.stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=rel_val, random_state=self.seed, stratify=strat_arg2
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    # ---------------------- Orchestrator ----------------------
    def run(self):
        self.load_data()
        df_fe = self.clean_feature_eng(self.df)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.split_data(df_fe)
        self.cm.rtm(f"X_train:{self.X_train.shape}, y_train:{self.y_train.shape}")
        return (self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test)


# Quick sanity check
if __name__ == "__main__":
    pipe = DataIngestion(seed=123, n_bins=10)
    Xtr, ytr, Xv, yv, Xt, yt = pipe.run()
    print("Train:", Xtr.shape, ytr.shape)
    print("Val  :", Xv.shape,  yv.shape)
    print("Test :", Xt.shape,  yt.shape)
    print("y (train) counts:\n", ytr.value_counts())
