

"""
Models pipeline rewritten for classification of 'Campaign Call Duration_bin_id'.
- Drops regression models (no regression target now)
- Works with DataIngestion in Data_pipeline.py
- Prints performance on val & test

Assumptions:
    * Data_pipeline.py exposes class DataIngestion with .run() returning
      (X_train, y_train, X_val, y_val, X_test, y_test)
    * CustomMessages exists for logging (optional). If not, replace with print.

Author: AMBADY
Date  : 2025-07-23
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,)
from sklearn.exceptions import UndefinedMetricWarning
import warnings

try:
    from Custom_messages import CustomMessages  # original name
except Exception:
    try:
        from custom_messages import CustomMessages
    except Exception:
        class CustomMessages:
            def rtm(self, msg: str):
                print(msg)


class ModelsPipeline:
    def __init__(self, drop_cols: list[str] | None = None, random_state: int = 42):
        self.drop_cols = drop_cols or ["Campaign Call Duration_bin"]
        self.random_state = random_state
        self.cm = CustomMessages()
        self.preprocessor: ColumnTransformer | None = None
        self.models: Dict[str, Pipeline] = {}
        self.results_: Dict[str, Dict[str, float]] = {}

    # -------------------------------
    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        self.cm.rtm(f"Preprocessing columns -> num:{len(num_cols)}, cat:{len(cat_cols)}")
        return ColumnTransformer([
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ], remainder="drop", sparse_threshold=0.3)

    # -------------------------------
    def _build_models(self) -> Dict[str, Pipeline]:
        logreg = LogisticRegression(max_iter=1000, random_state=self.random_state)  # multinomial by default
        rf     = RandomForestClassifier(n_estimators=300, random_state=self.random_state, n_jobs=-1, class_weight="balanced")
        gb     = GradientBoostingClassifier(random_state=self.random_state)
        return {"log_reg": logreg, "random_forest": rf, "grad_boost": gb}

    # -------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_train = X_train.drop(columns=[c for c in self.drop_cols if c in X_train.columns], errors="ignore")
        if self.preprocessor is None:
            self.preprocessor = self._build_preprocessor(X_train)
        base_models = self._build_models()
        self.models = {name: Pipeline([("pre", self.preprocessor), ("model", mdl)]) for name, mdl in base_models.items()}
        for name, mdl in self.models.items():
            self.cm.rtm(f"Training model: {name}")
            mdl.fit(X_train, y_train)
        self.cm.rtm("All models trained.")

    # -------------------------------
    def evaluate(self, X: pd.DataFrame, y: pd.Series, split_name: str = "val") -> pd.DataFrame:
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        X_eval = X.drop(columns=[c for c in self.drop_cols if c in X.columns], errors="ignore")
        rows = []
        for name, mdl in self.models.items():
            y_pred = mdl.predict(X_eval)
            acc = accuracy_score(y, y_pred)
            bacc = balanced_accuracy_score(y, y_pred)
            f1m = f1_score(y, y_pred, average="macro", zero_division=0)
            rows.append({"model": name, "split": split_name, "accuracy": acc, "balanced_acc": bacc, "f1_macro": f1m})
            rep = classification_report(y, y_pred, zero_division=0)
            self.cm.rtm(f"[{split_name}] {name} {rep}")
            self.results_.setdefault(name, {})[f"acc_{split_name}"] = acc
            self.results_.setdefault(name, {})[f"bacc_{split_name}"] = bacc
            self.results_.setdefault(name, {})[f"f1_{split_name}"] = f1m
        return pd.DataFrame(rows).sort_values("f1_macro", ascending=False)

    # -------------------------------
    def detailed_report(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> Tuple[str, np.ndarray]:
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not found")
        X_eval = X.drop(columns=[c for c in self.drop_cols if c in X.columns], errors="ignore")
        y_pred = self.models[model_name].predict(X_eval)
        rep = classification_report(y, y_pred, zero_division=0)
        cmx = confusion_matrix(y, y_pred)
        return rep, cmx


