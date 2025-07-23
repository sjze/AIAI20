

# Data_pipeline.py  (revised)
# ------------------------------------------------------------
import sqlite3
import pandas as pd
import numpy as np
import requests
import tempfile
from sklearn.model_selection import train_test_split
from Custom_messages import CustomMessages  # keep your existing logger


class DataIngestion:
    URL = "https://techassessment.blob.core.windows.net/aiap20-group-exercise-data/call_duration_modified.db"
    TABLE_NAME = "call_duration_modified"   # change if actual table name differs

    def __init__(self, seed: int = 42, n_bins: int = 10):
        self.seed = seed
        self.n_bins = n_bins
        self.cm = CustomMessages()

        # Raw splits
        self.df = None
        self.df_train = None
        self.df_val = None
        self.df_test = None

        # Cleaned / engineered
        self.df_train_fe = None
        self.df_val_fe = None
        self.df_test_fe = None

        # Final X / y
        self.X_train = self.y_train = None
        self.X_val   = self.y_val   = None
        self.X_test  = self.y_test  = None

    # ------------------------ IO ------------------------
    def download_db(self) -> str:
        r = requests.get(self.URL, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Download failed: {r.status_code}")
        self.cm.rtm("Database downloaded successfully.")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.write(r.content)
        tmp.close()
        return tmp.name

    def load_data(self):
        db_file = self.download_db()
        with sqlite3.connect(db_file) as conn:
            # If unsure of the table name, uncomment to inspect:
            # tbls = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
            # self.cm.rtm(f"Tables: {tbls['name'].tolist()}")
            self.df = pd.read_sql(f'SELECT * FROM "{self.TABLE_NAME}"', conn)
        self.cm.rtm(f"Loaded data shape: {self.df.shape}")

    def split_data(self, test_size=0.1, val_size=0.1):
        """
        Default: 80/10/10 split.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df_temp, self.df_test = train_test_split(
            self.df, test_size=test_size, random_state=self.seed
        )
        rel_val = val_size / (1 - test_size)
        self.df_train, self.df_val = train_test_split(
            df_temp, test_size=rel_val, random_state=self.seed
        )
        self.cm.rtm(f"Split -> train:{self.df_train.shape}, val:{self.df_val.shape}, test:{self.df_test.shape}")

    # --------------- Cleaning & Feature Eng ----------------
    def clean_feature_eng(self, data_input: pd.DataFrame) -> pd.DataFrame:
        """
        Apply EDA-derived cleaning & feature engineering.
        Steps:
          1. Drop 'SN' (ID-like).
          2. Create numeric 'Subscription Status' = 1 if Previous Outcome == 'success', else 0.
          3. Clean 'Campaign Call Duration': numeric, abs(), median impute.
          4. Bin duration into 10 equal-width bins; use *_bin_id (0..9) as target.
          5. Drop raw duration to avoid leakage.
          6. Basic NA handling for remaining columns (median for numeric / 'missing' for categorical).
        """
        df = data_input.copy()

        # 1) Drop ID
        if "SN" in df.columns:
            df = df.drop(columns=["SN"])

        # 2) Numeric Subscription Status (based on Previous Outcome)
        if "Previous Outcome" not in df.columns:
            raise KeyError("'Previous Outcome' column missing.")
        success_mask = df["Previous Outcome"].astype(str).str.strip().str.lower().eq("success")
        df["Subscription Status"] = success_mask.astype("int8")

        # 3) Clean duration
        dur_col = "Campaign Call Duration"
        if dur_col not in df.columns:
            raise KeyError(f"'{dur_col}' column missing.")

        s = pd.to_numeric(df[dur_col], errors="coerce").abs()
        s = s.fillna(s.median())
        df[dur_col] = s

        # 4) Bin duration
        edges = np.linspace(0, df[dur_col].max(), self.n_bins + 1)
        edges[0] = 0.0
        df[f"{dur_col}_bin"] = pd.cut(df[dur_col], bins=edges, include_lowest=True)
        df[f"{dur_col}_bin_id"] = pd.cut(
            df[dur_col], bins=edges, labels=range(self.n_bins), include_lowest=True
        ).astype("int8")

        # 5) Drop raw variable (leakage)
        df = df.drop(columns=[dur_col])

        # 6) Basic NA handling for the rest
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
        """
        Target = 'Campaign Call Duration_bin_id'.
        """
        target = "Campaign Call Duration_bin_id"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")
        if df[target].isna().any():
            raise ValueError("Target contains NaN.")

        X = df.drop(columns=[target])
        y = df[target]
        self.cm.rtm(f"X:{X.shape}, y:{y.shape}")
        return X, y

    # ---------------------- Orchestration -----------------------
    def run(self):
        """Run all steps sequentially."""
        self.load_data()
        self.split_data()

        self.df_train_fe = self.clean_feature_eng(self.df_train)
        self.df_val_fe   = self.clean_feature_eng(self.df_val)
        self.df_test_fe  = self.clean_feature_eng(self.df_test)

        self.X_train, self.y_train = self.split_features_target(self.df_train_fe)
        self.X_val,   self.y_val   = self.split_features_target(self.df_val_fe)
        self.X_test,  self.y_test  = self.split_features_target(self.df_test_fe)

        return (self.X_train, self.y_train,
                self.X_val,   self.y_val,
                self.X_test,  self.y_test)


# Quick sanity check
if __name__ == "__main__":
    pipe = DataIngestion(seed=123, n_bins=10)
    Xtr, ytr, Xv, yv, Xt, yt = pipe.run()
    print("Train:", Xtr.shape, ytr.shape)
    print("Val  :", Xv.shape,  yv.shape)
    print("Test :", Xt.shape,  yt.shape)
    print("y (train) counts:\n", ytr.value_counts())
