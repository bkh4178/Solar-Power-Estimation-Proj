
# feature_audit.py
# --------------------------------------
# Drop-in utilities to (1) assemble X/y, (2) check multicollinearity (VIF/corr),
# (3) estimate feature importance (RF + permutation), and (4) suggest drops.
#
# Usage (in your notebook):
#   from feature_audit import (
#       prepare_Xy, compute_vif, high_corr_pairs,
#       fit_importance, summarize_importance, suggest_feature_drops
#   )
#   X, y = prepare_Xy(df, target="nins",
#                     id_cols=["time","pv_id","pv_id_code"],
#                     drop_if_missing=True)
#   vif = compute_vif(X)
#   corr_hits = high_corr_pairs(X, thresh=0.98)
#   model, imp_df, perm_df = fit_importance(X, y, random_state=42, n_jobs=-1)
#   summary = summarize_importance(imp_df, perm_df)
#   drops = suggest_feature_drops(X, vif, corr_hits, summary,
#                                 vif_thresh=10.0, corr_thresh=0.98,
#                                 bottom_quantile=0.15)
#
#   print(summary.head(20))
#   print(vif.sort_values("VIF", ascending=False).head(20))
#   print(corr_hits.head(20))
#   print(drops)   # {'by_vif': [...], 'by_corr': [...], 'by_importance': [...], 'union': [...]}
#
# --------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# -------------
# 1) Assemble X/y
# -------------

def prepare_Xy(df: pd.DataFrame,
               target: str = "nins",
               id_cols: Optional[List[str]] = None,
               feature_cols: Optional[List[str]] = None,
               drop_if_missing: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    \"\"\"Assemble X and y from a DataFrame.
    If feature_cols is None: takes numeric columns except id_cols + target.
    If drop_if_missing: drop rows with NA in y (and in X if needed).
    \"\"\"
    id_cols = id_cols or []
    if feature_cols is None:
        numeric = df.select_dtypes(include=[\"number\"]).columns.tolist()
        feature_cols = [c for c in numeric if c not in set(id_cols + [target])]

    X = df.loc[:, feature_cols].copy()
    y = df.loc[:, target].copy()

    if drop_if_missing:
        keep = y.notna()
        if X.isna().any().any():
            keep &= ~(X.isna().any(axis=1))
        X = X.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)

    return X, y

# ---------------------------------------
# 2) Multicollinearity: VIF & high-corr
# ---------------------------------------

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Compute VIFs by regressing each standardized feature against the others.\"\"\"
    cols = X.columns.tolist()
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(X.values.astype(float))
    Z = np.nan_to_num(Z, nan=0.0)

    vif_values = []
    for j in range(Z.shape[1]):
        yj = Z[:, j]
        Xj = np.delete(Z, j, axis=1)
        reg = LinearRegression().fit(Xj, yj)
        r2 = reg.score(Xj, yj)
        vif = np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2)
        vif_values.append(vif)

    out = pd.DataFrame({\"feature\": cols, \"VIF\": vif_values}).sort_values(\"VIF\", ascending=False).reset_index(drop=True)
    return out

def high_corr_pairs(X: pd.DataFrame, thresh: float = 0.98, method: str = \"pearson\") -> pd.DataFrame:
    \"\"\"Return pairs of features with |corr| >= thresh (upper triangle only).\"\"\"
    corr = X.corr(method=method).abs()
    hits = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            v = corr.iat[i, j]
            if np.isfinite(v) and v >= thresh:
                hits.append((cols[i], cols[j], float(v)))
    return pd.DataFrame(hits, columns=[\"feat_a\",\"feat_b\",\"abs_corr\"]).sort_values(\"abs_corr\", ascending=False)

# ---------------------------------------
# 3) Feature importance (model + permutation)
# ---------------------------------------

def fit_importance(X: pd.DataFrame, y: pd.Series,
                   random_state: int = 42, n_jobs: int = -1):
    \"\"\"Fit RandomForest and compute impurity & permutation importance.\"\"\"
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=n_jobs
    )
    rf.fit(X_train, y_train)

    imp_df = pd.DataFrame({
        \"feature\": X.columns,
        \"impurity_importance\": rf.feature_importances_
    }).sort_values(\"impurity_importance\", ascending=False).reset_index(drop=True)

    perm = permutation_importance(rf, X_valid, y_valid,
                                  n_repeats=10, random_state=random_state, n_jobs=n_jobs)
    perm_df = pd.DataFrame({
        \"feature\": X.columns,
        \"perm_importance_mean\": perm.importances_mean,
        \"perm_importance_std\": perm.importances_std
    }).sort_values(\"perm_importance_mean\", ascending=False).reset_index(drop=True)

    return rf, imp_df, perm_df

def summarize_importance(imp_df: pd.DataFrame, perm_df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Merge impurity and permutation importance, add average rank.\"\"\"
    a = imp_df.copy()
    b = perm_df.copy()
    a[\"rank_impurity\"] = a[\"impurity_importance\"].rank(ascending=False, method=\"min\")
    b[\"rank_perm\"]     = b[\"perm_importance_mean\"].rank(ascending=False, method=\"min\")
    out = a.merge(b, on=\"feature\", how=\"outer\")
    out[\"rank_avg\"] = out[[\"rank_impurity\",\"rank_perm\"]].mean(axis=1)
    return out.sort_values(\"rank_avg\", ascending=True).reset_index(drop=True)

# ---------------------------------------
# 4) Suggest drops
# ---------------------------------------

def suggest_feature_drops(X: pd.DataFrame,
                          vif_df: pd.DataFrame,
                          corr_pairs_df: pd.DataFrame,
                          imp_summary_df: pd.DataFrame,
                          vif_thresh: float = 10.0,
                          corr_thresh: float = 0.98,
                          bottom_quantile: float = 0.15) -> Dict[str, List[str]]:
    \"\"\"Heuristic drop suggestions using VIF, high-corr, and low-importance.\"\"\"
    by_vif = vif_df.loc[vif_df[\"VIF\"] >= vif_thresh, \"feature\"].tolist()

    imp = imp_summary_df.copy()
    imp[\"rank_avg\"] = imp[\"rank_avg\"].fillna(1e9)
    cutoff = imp[\"rank_avg\"].quantile(1.0 - bottom_quantile)
    by_imp = imp.loc[imp[\"rank_avg\"] >= cutoff, \"feature\"].tolist()

    ranks = imp.set_index(\"feature\")[\"rank_avg\"].to_dict()
    drops_corr = set()
    for _, row in corr_pairs_df.iterrows():
        a, b = row[\"feat_a\"], row[\"feat_b\"]
        ra = ranks.get(a, 1e9)
        rb = ranks.get(b, 1e9)
        if ra >= rb:
            drops_corr.add(a)
        else:
            drops_corr.add(b)

    union = sorted(set(by_vif) | set(by_imp) | drops_corr)
    return {
        \"by_vif\": sorted(set(by_vif)),
        \"by_corr\": sorted(drops_corr),
        \"by_importance\": sorted(set(by_imp)),
        \"union\": union,
    }
