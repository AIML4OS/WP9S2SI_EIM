
# Imputation evaluation 
# -----------------------------------------------------------------------------

# Core metrics, RMSE, MAE r2 etc.
# Robust spread metrics (IQR/MAD/P95–P5/winsorized variance ratios),
# KS test, distribution comparison plots (hist/ECDF/QQ/box),
# Group-level metrics in `summarize_by_group(...)` -> summary_groups.
# 

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================ Robust spread helpers ===========================

def _winsorize(a: np.ndarray, p: float = 5.0) -> np.ndarray:
    """Winsorize at p and 100-p percentiles (default 5%)."""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return a
    lo, hi = np.percentile(a, [p, 100.0 - p])
    return np.clip(a, lo, hi)

def _mad(a: np.ndarray) -> float:
    """Median Absolute Deviation (unscaled)."""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    med = np.median(a)
    return float(np.median(np.abs(a - med)))

def _iqr(a: np.ndarray) -> float:
    """Interquartile range (Q75 - Q25)."""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    q25, q75 = np.percentile(a, [25, 75])
    return float(q75 - q25)

def _p95p5(a: np.ndarray) -> float:
    """Distance between 95th and 5th percentiles."""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    p5, p95 = np.percentile(a, [5, 95])
    return float(p95 - p5)


# ================================ KS (2-sample) ===============================

def _ks_2samp_stat_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov D statistic + asymptotic p-value (SciPy-free)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    x_sorted = np.sort(x); y_sorted = np.sort(y)
    i = j = 0; d = 0.0
    while i < n1 and j < n2:
        if x_sorted[i] <= y_sorted[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / n1 - j / n2))
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    s = 0.0
    for k in range(1, 101):
        term = 2 * ((-1) ** (k - 1)) * np.exp(-2 * (k * k) * (lam * lam))
        s += term
        if abs(term) < 1e-10:
            break
    p = max(0.0, min(1.0, s))
    return float(d), float(p)


# ============================ Core metrics  ===========================


def _safe_metrics(y: np.ndarray, yhat: np.ndarray, winsor_p: float = 5.0) -> dict:
    """Regression metrics + distributional comparisons, incl. robust spread and bias metrics."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y = y[mask]
    yhat = yhat[mask]
    n = len(y)

    if n == 0:
        return {
            "rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape_%": np.nan,
            "smape_%": np.nan, "rmsle": np.nan, "nrmse_mean": np.nan, "nmae_mean": np.nan,
            "resid_std": np.nan, "resid_var": np.nan, "n": 0,
            # bias block
            "mean_true": np.nan, "mean_imp": np.nan, "mean_diff": np.nan,
            "mean_bias": np.nan, "abs_mean_bias": np.nan,
            "rel_mean_bias_%": np.nan, "abs_rel_mean_bias_%": np.nan,
            "smd": np.nan, "AbsSMD": np.nan,
            # variance & robust block
            "var_true": np.nan, "var_imp": np.nan, "var_ratio": np.nan,
            "IQR_true": np.nan, "IQR_imp": np.nan, "IQR_ratio": np.nan,
            "MAD_true": np.nan, "MAD_imp": np.nan, "MAD_ratio": np.nan,
            "P95P5_true": np.nan, "P95P5_imp": np.nan, "P95P5_ratio": np.nan,
            "Var_winsor_true": np.nan, "Var_winsor_imp": np.nan, "Var_winsor_ratio": np.nan,
            # distributional distance
            "ks_stat": np.nan, "ks_p": np.nan
        }

    # --- core errors
    resid = yhat - y
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae  = float(np.mean(np.abs(resid)))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    ss_res = float(np.sum((y - yhat)**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    eps = 1e-12
    y_safe = np.where(np.abs(y) < eps, eps, y)
    mape  = float(np.mean(np.abs(y - yhat) / np.abs(y_safe)) * 100.0)
    smape = float(100.0 * np.mean(2.0 * np.abs(yhat - y) / (np.abs(y) + np.abs(yhat) + eps)))
    if (y >= 0).all() and (yhat >= 0).all():
        rmsle = float(np.sqrt(np.mean((np.log1p(y) - np.log1p(yhat))**2)))
    else:
        rmsle = np.nan

    mean_abs_y = float(np.mean(np.abs(y)) + eps)
    resid_std  = float(np.std(resid, ddof=1)) if n > 1 else np.nan
    resid_var  = float(np.var(resid, ddof=1)) if n > 1 else np.nan

    # --- bias block
    mean_true = float(np.mean(y))
    mean_imp  = float(np.mean(yhat))
    mean_bias = float(mean_imp - mean_true)              # signed mean bias (MBE)
    abs_mean_bias = float(abs(mean_bias))                # absolute mean bias
    rel_mean_bias_pct = float(100.0 * mean_bias / (abs(mean_true) + eps))
    abs_rel_mean_bias_pct = float(abs(rel_mean_bias_pct))

    # keep backward-compat key as alias
    mean_diff = mean_bias

    # SMD (global pooled)
    if n > 1:
        v_true = float(np.var(y, ddof=1))
        v_imp  = float(np.var(yhat, ddof=1))
    else:
        v_true = v_imp = np.nan
    var_true = v_true if n > 1 else np.nan
    var_imp  = v_imp  if n > 1 else np.nan
    var_ratio = (var_imp/var_true) if (np.isfinite(var_true) and var_true != 0) else np.nan

    pooled_sd = float(np.sqrt(((n - 1) * (v_true if np.isfinite(v_true) else 0.0) +
                               (n - 1) * (v_imp  if np.isfinite(v_imp)  else 0.0)) / max(1, 2*n - 2))) if n > 1 else np.nan
    smd = (mean_imp - mean_true) / pooled_sd if (pooled_sd and pooled_sd > 0) else np.nan
    AbsSMD = float(abs(smd)) if np.isfinite(smd) else np.nan

    # KS
    ks_stat, ks_p = _ks_2samp_stat_p(y, yhat)

    # --- robust spread
    IQR_true = _iqr(y);   IQR_imp = _iqr(yhat)
    IQR_ratio = (IQR_imp / IQR_true) if (np.isfinite(IQR_true) and IQR_true != 0) else np.nan

    MAD_true = _mad(y);   MAD_imp = _mad(yhat)
    MAD_ratio = (MAD_imp / MAD_true) if (np.isfinite(MAD_true) and MAD_true != 0) else np.nan

    P95P5_true = _p95p5(y);  P95P5_imp = _p95p5(yhat)
    P95P5_ratio = (P95P5_imp / P95P5_true) if (np.isfinite(P95P5_true) and P95P5_true != 0) else np.nan

    y_w  = _winsorize(y, winsor_p);  yhat_w = _winsorize(yhat, winsor_p)
    Var_winsor_true = float(np.var(y_w, ddof=1)) if y_w.size > 1 else np.nan
    Var_winsor_imp  = float(np.var(yhat_w, ddof=1)) if yhat_w.size > 1 else np.nan
    Var_winsor_ratio = (Var_winsor_imp/Var_winsor_true) if (np.isfinite(Var_winsor_true) and Var_winsor_true != 0) else np.nan

    return {
        # errors
        "rmse": rmse, "mae": mae, "r2": r2, "mape_%": mape, "smape_%": smape, "rmsle": rmsle,
        "nrmse_mean": float(rmse/mean_abs_y), "nmae_mean": float(mae/mean_abs_y),
        "resid_std": resid_std, "resid_var": resid_var, "n": int(n),
        # bias block
        "mean_true": mean_true, "mean_imp": mean_imp,
        "mean_diff": mean_diff,               # legacy alias
        "mean_bias": mean_bias,               # preferred name
        "abs_mean_bias": abs_mean_bias,
        "rel_mean_bias_%": rel_mean_bias_pct,
        "abs_rel_mean_bias_%": abs_rel_mean_bias_pct,
        "smd": float(smd) if np.isfinite(smd) else np.nan,
        "AbsSMD": AbsSMD,
        # variance & robust
        "var_true": var_true, "var_imp": var_imp, "var_ratio": var_ratio,
        "IQR_true": IQR_true, "IQR_imp": IQR_imp, "IQR_ratio": IQR_ratio,
        "MAD_true": MAD_true, "MAD_imp": MAD_imp, "MAD_ratio": MAD_ratio,
        "P95P5_true": P95P5_true, "P95P5_imp": P95P5_imp, "P95P5_ratio": P95P5_ratio,
        "Var_winsor_true": Var_winsor_true, "Var_winsor_imp": Var_winsor_imp, "Var_winsor_ratio": Var_winsor_ratio,
        # distributional distance
        "ks_stat": ks_stat, "ks_p": ks_p
    }



# ================= rows_long + summary (global, by method/mech/missrate) ======

def evaluate_imputations_from_methods(
    methods_df: pd.DataFrame,
    frames_by_status: Dict[int, pd.DataFrame],
    original_df: pd.DataFrame,
    *,
    id_col: str,
    target_cols: List[str],
    imputed_map: Optional[Dict[str, str]] = None,
    how_join_original: str = "left",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build `rows_long` of (y_true, y_pred) and a `summary` with standard + robust metrics.
    Grouped by: Method, MissMech, MissRate, STATUS_ZAPISA, variable.
    """
    rows_all: List[pd.DataFrame] = []
    for s, df_imp in frames_by_status.items():
        if df_imp is None or df_imp.empty:
            continue
        a = df_imp.copy()
        b = methods_df[["Status_zapisa", "Method", "MissMech", "MissRate"]].copy()
        a = a.merge(b, left_on="STATUS_ZAPISA", right_on="Status_zapisa", how="left") \
             .drop(columns=["Status_zapisa"], errors="ignore")
        c = original_df[[id_col] + list(target_cols)].copy()

        for true_col in target_cols:
            pred_col = (imputed_map or {}).get(true_col, true_col)
            tmp = a[[id_col, pred_col, "STATUS_ZAPISA", "Method", "MissMech", "MissRate"]].copy()
            tmp = tmp.rename(columns={pred_col: "y_pred"})
            merged = tmp.merge(c[[id_col, true_col]].rename(columns={true_col: "y_true"}),
                               on=id_col, how=how_join_original)
            merged["y_pred"] = pd.to_numeric(merged["y_pred"], errors="coerce")
            merged["y_true"] = pd.to_numeric(merged["y_true"], errors="coerce")
            merged["variable"] = true_col
            merged["residual"] = merged["y_pred"] - merged["y_true"]
            rows_all.append(merged)

    if not rows_all:
        return pd.DataFrame(), pd.DataFrame()

    rows_long = pd.concat(rows_all, ignore_index=True)

    grp_keys = ["Method", "MissMech", "MissRate", "STATUS_ZAPISA", "variable"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        met = _safe_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        return pd.Series(met)

    summary = rows_long.groupby(grp_keys, dropna=False).apply(_agg).reset_index()

    # Rename to TitleCase for reporting
    alias = {
        "rmse":"RMSE","mae":"MAE","r2":"R2","mape_%":"MAPE_%","smape_%":"SMAPE_%","rmsle":"RMSLE",
        "nrmse_mean":"NRMSE_mean","nmae_mean":"NMAE_mean",
        "resid_std":"Resid_std","resid_var":"Resid_var","n":"n",
        "mean_true":"Mean_true","mean_imp":"Mean_imp","mean_diff":"Mean_diff","smd":"SMD",
        "var_true":"Var_true","var_imp":"Var_imp","var_ratio":"Var_ratio",
        "ks_stat":"KS_stat","ks_p":"KS_p",
        "IQR_true":"IQR_true","IQR_imp":"IQR_imp","IQR_ratio":"IQR_ratio",
        "MAD_true":"MAD_true","MAD_imp":"MAD_imp","MAD_ratio":"MAD_ratio",
        "P95P5_true":"P95P5_true","P95P5_imp":"P95P5_imp","P95P5_ratio":"P95P5_ratio",
        "Var_winsor_true":"Var_winsor_true","Var_winsor_imp":"Var_winsor_imp","Var_winsor_ratio":"Var_winsor_ratio",
    }
    for k, v in alias.items():
        if k in summary.columns:
            summary[v] = summary[k]

    return rows_long, summary


# ========================= Group-level metrics ==========================

def summarize_by_group(
    rows_long: pd.DataFrame,
    group_col: str,
    *,
    by_keys: Optional[List[str]] = None,
    winsor_p: float = 5.0
) -> pd.DataFrame:
    """
    Return `summary_groups`: metrics computed within each group of `group_col`.
    You can optionally compute them separately per combination of `by_keys`
    (e.g., ["Method","MissMech","MissRate","STATUS_ZAPISA"]).

    Output columns include by_keys, variable, group_col, n, RMSE/MAE/R2,...,
    Mean_true/Mean_imp/Mean_diff, Var_true/Var_imp/Var_ratio,
    robust IQR/MAD/P95P5 ratios, winsorized Var_ratio, KS_stat/KS_p, etc.
    """
    required = {"variable", "y_true", "y_pred", group_col}
    missing = required - set(rows_long.columns)
    if missing:
        raise KeyError(f"rows_long is missing columns: {sorted(missing)}")

    keys = list(by_keys) if by_keys else []
    grp = rows_long.groupby(keys + ["variable", group_col], dropna=False)

    def _agg(g: pd.DataFrame) -> pd.Series:
        m = _safe_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), winsor_p=winsor_p)
        return pd.Series(m)

    summary_groups = grp.apply(_agg).reset_index()

    # TitleCase aliases for reporting
    alias = {
        "rmse":"RMSE","mae":"MAE","r2":"R2","mape_%":"MAPE_%","smape_%":"SMAPE_%","rmsle":"RMSLE",
        "nrmse_mean":"NRMSE_mean","nmae_mean":"NMAE_mean",
        "resid_std":"Resid_std","resid_var":"Resid_var","n":"n",
        "mean_true":"Mean_true","mean_imp":"Mean_imp","mean_diff":"Mean_diff","smd":"SMD",
        "var_true":"Var_true","var_imp":"Var_imp","var_ratio":"Var_ratio",
        "ks_stat":"KS_stat","ks_p":"KS_p",
        "IQR_true":"IQR_true","IQR_imp":"IQR_imp","IQR_ratio":"IQR_ratio",
        "MAD_true":"MAD_true","MAD_imp":"MAD_imp","MAD_ratio":"MAD_ratio",
        "P95P5_true":"P95P5_true","P95P5_imp":"P95P5_imp","P95P5_ratio":"P95P5_ratio",
        "Var_winsor_true":"Var_winsor_true","Var_winsor_imp":"Var_winsor_imp","Var_winsor_ratio":"Var_winsor_ratio",
    }
    for k, v in alias.items():
        if k in summary_groups.columns:
            summary_groups[v] = summary_groups[k]

    # Keep group column explicit and nicely named
    summary_groups = summary_groups.rename(columns={group_col: "group"})
    summary_groups.insert(summary_groups.columns.get_loc("group"), "group_col", group_col)

    return summary_groups


# ============================= SMD by group ============================

def smd_by_group(
    rows_long: pd.DataFrame,
    group_col: str,
    *,
    ref_rule: str = "population",   # "population" | "most_frequent" | "first" | "value"
    ref_value: str | None = None,
    by_keys: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Standardized Mean Difference of group contrasts (ŷ vs y).
    If `by_keys` is provided, SMDs are computed within each combination of by_keys.
    """
    required = {"variable", "y_true", "y_pred", group_col}
    missing = required - set(rows_long.columns)
    if missing:
        raise KeyError(f"rows_long is missing columns: {sorted(missing)}")

    results = [] 

    def _key_view(df: pd.DataFrame, key_vals):
        if key_vals is None:
            return df, {}
        mask = np.ones(len(df), dtype=bool)
        key_map = {}
        for k, v in zip(by_keys, key_vals):
            mask &= (df[k] == v)
            key_map[k] = v
        return df.loc[mask].copy(), key_map

    uniques = [tuple(vals) if by_keys else (None,)
               for vals in rows_long.groupby(by_keys).groups.keys()] if by_keys else [(None,)]

    for key_vals in uniques:
        sub, key_map = _key_view(rows_long, key_vals if by_keys else None)
        if sub.empty:
            continue

        for var, sv in sub.groupby("variable", dropna=False):
            g = sv.groupby(group_col, dropna=False).agg(
                n=("y_true", "size"),
                mean_true=("y_true", "mean"),
                mean_imp=("y_pred", "mean"),
                var_true=("y_true", lambda s: np.var(s, ddof=1)),
                var_imp=("y_pred", lambda s: np.var(s, ddof=1)),
            ).reset_index()

            mean_true_all = sv["y_true"].mean()
            mean_imp_all  = sv["y_pred"].mean()
            var_true_all  = np.var(sv["y_true"], ddof=1)
            var_imp_all   = np.var(sv["y_pred"], ddof=1)

            s_global = np.sqrt((var_true_all + var_imp_all) / 2.0) if np.isfinite(var_true_all) and np.isfinite(var_imp_all) else np.nan
            s_within = np.sqrt((g["var_true"].mean() + g["var_imp"].mean()) / 2.0) if len(g) > 0 else np.nan

            if ref_rule == "population":
                ref_label = "POPULATION"
                ref_true = mean_true_all
                ref_imp  = mean_imp_all
            elif ref_rule == "most_frequent":
                ref_row = g.sort_values("n", ascending=False).iloc[0]
                ref_label = str(ref_row[group_col])
                ref_true = float(ref_row["mean_true"])
                ref_imp  = float(ref_row["mean_imp"])
            elif ref_rule == "first":
                ref_label = sorted(g[group_col].astype(str).unique())[0]
                ref_true = float(g.loc[g[group_col].astype(str) == ref_label, "mean_true"].iloc[0])
                ref_imp  = float(g.loc[g[group_col].astype(str) == ref_label, "mean_imp"].iloc[0])
            elif ref_rule == "value":
                if ref_value is None:
                    raise ValueError("ref_rule='value' requires ref_value.")
                ref_label = str(ref_value)
                if ref_label not in set(g[group_col].astype(str)):
                    raise ValueError(f"Reference '{ref_label}' not found in groups.")
                ref_true = float(g.loc[g[group_col].astype(str) == ref_label, "mean_true"].iloc[0])
                ref_imp  = float(g.loc[g[group_col].astype(str) == ref_label, "mean_imp"].iloc[0])
            else:
                raise ValueError("ref_rule must be one of: 'population' | 'most_frequent' | 'first' | 'value'")

            diffs_true = g["mean_true"] - ref_true
            diffs_imp  = g["mean_imp"]  - ref_imp
            errors     = diffs_imp - diffs_true

            smd_global = errors / s_global if s_global and s_global > 0 else np.nan
            smd_within = errors / s_within if s_within and s_within > 0 else np.nan

            for grp, sg, sw in zip(g[group_col], np.atleast_1d(smd_global), np.atleast_1d(smd_within)):
                row = {"variable": var, "group_col": group_col, "group": grp,
                       "ref": ref_label,
                       "SMD_global": float(sg) if np.isfinite(sg) else np.nan,
                       "SMD_within": float(sw) if np.isfinite(sw) else np.nan}
                for k, v in key_map.items():
                    row[k] = v
                results.append(row)

    return pd.DataFrame(results)


# ============================== Plotting helpers ==============================

def _to_list(x):
    if x is None: return None
    if isinstance(x, (list, tuple, set)): return list(x)
    return [x]

def _to_num(series: pd.Series) -> pd.Series:
    """Coerce strings like '10%' or '0,1' to numeric."""
    s = series.astype(str).str.replace('%','', regex=False).str.replace(',','.', regex=False)
    return pd.to_numeric(s, errors='coerce')

def _apply_filters(df: pd.DataFrame,
                   variables: Optional[Iterable[str]],
                   mechs: Optional[Iterable[str]],
                   methods: Optional[Iterable[str]],
                   missrates: Optional[Iterable[str]]) -> pd.DataFrame:
    """Filter summary by selections."""
    out = df.copy()
    if variables is not None:
        out = out[out["variable"].isin(_to_list(variables))]
    if mechs is not None and "MissMech" in out.columns:
        out = out[out["MissMech"].isin(_to_list(mechs))]
    if methods is not None:
        out = out[out["Method"].isin(_to_list(methods))]
    if missrates is not None and "MissRate" in out.columns:
        out = out[out["MissRate"].isin(_to_list(missrates))]
    return out


# ============================= Metric line plots ==============================

def make_metric_lines(summary: pd.DataFrame,
                      metric: str = "RMSE",
                      variables: Optional[Iterable[str]] = None,
                      mechs: Optional[Iterable[str]] = None,
                      methods: Optional[Iterable[str]] = None,
                      missrates: Optional[Iterable[str]] = None,
                      title_prefix: str = "") -> List[plt.Figure]:
    """Line plots of <metric> vs MissRate per (variable × MissMech)."""
    df = _apply_filters(summary, variables, mechs, methods, missrates)
    if df.empty or metric not in df.columns:
        return []

    if "MissRate" in df.columns:
        df = df.assign(MissRate_num=_to_num(df["MissRate"]))
    else:
        df = df.assign(MissRate_num=np.nan)

    figs: List[plt.Figure] = []
    variables_u = list(df["variable"].dropna().unique()) if "variable" in df.columns else ["_"]
    mechs_u = list(df["MissMech"].dropna().unique()) if "MissMech" in df.columns else ["_"]

    for v in variables_u:
        for m in mechs_u:
            sub = df[(df.get("variable","_") == v) & (df.get("MissMech","_") == m)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("MissRate_num")
            fig = plt.figure()
            for method, g in sub.groupby("Method"):
                plt.plot(g["MissRate_num"], g[metric], marker="o", label=str(method))
            if "MissRate" in sub.columns:
                xlabels = sub.groupby("MissRate_num")["MissRate"].first().sort_index()
                plt.xticks(xlabels.index.values, xlabels.values, rotation=0)
            plt.xlabel("MissRate"); plt.ylabel(metric)
            title = f"{title_prefix}{metric} vs MissRate — {v}"
            if "MissMech" in df.columns:
                title += f" — {m}"
            plt.title(title); plt.legend(loc="best"); plt.tight_layout()
            figs.append(fig)
    return figs


# ================================ Grouped bars ================================

def make_grouped_bars(summary: pd.DataFrame,
                      value_col: str = "Mean_diff",
                      variables: Optional[Iterable[str]] = None,
                      mechs: Optional[Iterable[str]] = None,
                      methods: Optional[Iterable[str]] = None,
                      missrates: Optional[Iterable[str]] = None,
                      group: str = "Method",
                      hue: str = "MissRate",
                      title_prefix: str = "") -> List[plt.Figure]:
    """Grouped bars: x=group (e.g., Method), hue=MissRate; per (variable × MissMech)."""
    df = _apply_filters(summary, variables, mechs, methods, missrates)
    if df.empty or value_col not in df.columns:
        return []

    figs: List[plt.Figure] = []
    variables_u = list(df["variable"].dropna().unique()) if "variable" in df.columns else ["_"]
    mechs_u = list(df["MissMech"].dropna().unique()) if "MissMech" in df.columns else ["_"]

    for v in variables_u:
        for mech in mechs_u:
            sub = df[(df.get("variable","_") == v) & (df.get("MissMech","_") == mech)].copy()
            if sub.empty:
                continue

            groups = list(sorted(sub[group].astype(str).unique()))
            hues   = list(sorted(sub[hue].astype(str).unique())) if hue in sub.columns else []

            n_groups = len(groups)
            n_hues = max(1, len(hues))
            total_width = 0.8
            bar_width = total_width / n_hues

            fig = plt.figure()
            for hi, hv in enumerate(hues if hues else ["_no_hue_"]):
                x_positions = np.arange(n_groups) + (hi - (n_hues-1)/2) * bar_width
                vals = []
                for gname in groups:
                    mask = (sub[group].astype(str) == str(gname)) & ((sub[hue].astype(str) == str(hv)) if hues else True)
                    vals.append(sub.loc[mask, value_col].mean() if mask.any() else np.nan)
                plt.bar(x_positions, vals, width=bar_width, label=str(hv) if hues else None)

            plt.xticks(np.arange(n_groups), groups, rotation=45, ha='right')
            if value_col.lower() == "smd":
                for thr in [0.1, 0.2, -0.1, -0.2]:
                    plt.axhline(thr, linestyle=":", linewidth=1)
            plt.ylabel(value_col)
            title = f"{title_prefix}{value_col} — {v}"
            if "MissMech" in df.columns:
                title += f" — {mech}"
            plt.title(title)
            if hues:
                plt.legend(title=hue, loc="best")
            plt.tight_layout()
            figs.append(fig)
    return figs


# ======================= Distribution comparison plots =======================

def _fd_bins(a: np.ndarray) -> int:
    """Freedman–Diaconis rule for bin count."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    n = a.size
    if n < 2: return 10
    iqr = _iqr(a)
    if iqr <= 0: return int(np.sqrt(n))
    h = 2 * iqr * (n ** (-1/3))
    if h <= 0: return int(np.sqrt(n))
    bins = int(np.ceil((a.max() - a.min()) / h))
    return max(10, min(200, bins))

def _filter_rows(rows_long: pd.DataFrame,
                 variables=None, mechs=None, methods=None, missrates=None) -> pd.DataFrame:
    """Filter rows_long like summary filters."""
    df = rows_long.copy()
    if variables is not None:
        df = df[df["variable"].isin(list(variables))]
    if mechs is not None and "MissMech" in df.columns:
        df = df[df["MissMech"].isin(list(mechs))]
    if methods is not None:
        df = df[df["Method"].isin(list(methods))]
    if missrates is not None and "MissRate" in df.columns:
        df = df[df["MissRate"].isin(list(missrates))]
    return df

def make_hist_overlays(
    rows_long: pd.DataFrame,
    variables=None, mechs=None, methods=None, missrates=None,
    bins="fd", density=True, sample_n: Optional[int] = None,
    colors: Tuple[str, str] = ("tab:blue", "tab:orange"),
    alphas: Tuple[float, float] = (0.5, 0.5),
    edgecolor: str = "white",
    linewidth: float = 0.5,
    x_clip: Optional[Tuple[float, float]] = (0.0, 0.99),  
    x_transform: Optional[str] = None                      
) -> List[plt.Figure]:
    """Overlaid histograms (y vs ŷ) per (variable × MissMech × Method × MissRate)."""
    df = _filter_rows(rows_long, variables, mechs, methods, missrates)
    if df.empty: 
        return []
    figs = []
    by = ["variable"] + (["MissMech"] if "MissMech" in df.columns else []) + ["Method"] + (["MissRate"] if "MissRate" in df.columns else [])
    for keys, sub in df.groupby(by, dropna=False):
        y  = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()

        # optional sampling (if needed)
        if sample_n and len(y) > sample_n:
            idx = np.random.default_rng(0).choice(len(y), size=sample_n, replace=False)
            y, yp = y[idx], yp[idx]

        # optional x transformation (e.g., log1p for right-skewed distributions)
        if x_transform == "log1p":
            y_plot  = np.log1p(y)
            yp_plot = np.log1p(yp)
            x_label = f"log(1 + {sub['variable'].iloc[0]})"
        else:
            y_plot, yp_plot = y, yp
            x_label = sub["variable"].iloc[0]

        # combined data for binning and quantiles
        combo = np.concatenate([y_plot, yp_plot])

        # quantile-based x-axis clipping (truncate sparse tail)
        if x_clip is not None:
            qlo, qhi = np.quantile(combo, x_clip)
            # clip data for plotting and compute bins on clipped range
            in_rng_y  = (y_plot >= qlo) & (y_plot <= qhi)
            in_rng_yp = (yp_plot >= qlo) & (yp_plot <= qhi)
            y_plot, yp_plot = y_plot[in_rng_y], yp_plot[in_rng_yp]
            combo_for_bins = np.concatenate([y_plot, yp_plot])
        else:
            qlo, qhi = np.min(combo), np.max(combo)
            combo_for_bins = combo

        # bins
        b = _fd_bins(combo_for_bins) if bins == "fd" else bins

        # plotting
        fig = plt.figure()
        plt.hist(
            y_plot, bins=b, alpha=alphas[0], density=density, label="Original (y)",
            edgecolor=edgecolor, linewidth=linewidth, color=colors[0] 
        )
        plt.hist(
            yp_plot, bins=b, alpha=alphas[1], density=density, label="Imputed (ŷ)",
            edgecolor=edgecolor, linewidth=linewidth, color=colors[1] 
        )

        # titles and axes
        t = " | ".join([str(k) for k in (keys if isinstance(keys, tuple) else (keys,))])
        plt.title(f"Histogram overlay — {t}")
        plt.xlabel(x_label)
        plt.ylabel("Density" if density else "Count")
        plt.legend(loc="best")
        plt.xlim(qlo, qhi)  # explicitly limit x-axis to selected quantile range
        plt.tight_layout()
        figs.append(fig)
    return figs

def make_ecdf_plots(rows_long: pd.DataFrame,
                    variables=None, mechs=None, methods=None, missrates=None,
                    sample_n: Optional[int] = None) -> List[plt.Figure]:
    """ECDF comparison (y vs ŷ)."""
    df = _filter_rows(rows_long, variables, mechs, methods, missrates)
    if df.empty: return []
    figs = []
    by = ["variable"] + (["MissMech"] if "MissMech" in df.columns else []) + ["Method"] + (["MissRate"] if "MissRate" in df.columns else [])
    for keys, sub in df.groupby(by, dropna=False):
        y  = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        if sample_n and len(y) > sample_n:
            idx = np.random.default_rng(0).choice(len(y), size=sample_n, replace=False)
            y, yp = y[idx], yp[idx]
        def ecdf(a):
            a = a[~np.isnan(a)]
            if a.size == 0: return np.array([]), np.array([])
            a = np.sort(a); x = a; F = np.arange(1, a.size+1)/a.size
            return x, F
        x1, F1 = ecdf(y); x2, F2 = ecdf(yp)
        fig = plt.figure()
        if x1.size: plt.step(x1, F1, where="post", label="Original (y)")
        if x2.size: plt.step(x2, F2, where="post", label="Imputed (ŷ)")
        t = " | ".join([str(k) for k in (keys if isinstance(keys, tuple) else (keys,))])
        plt.title(f"ECDF — {t}")
        plt.xlabel(sub["variable"].iloc[0]); plt.ylabel("F(y)")
        plt.legend(loc="best"); plt.tight_layout()
        figs.append(fig)
    return figs

def make_qq_plots(rows_long: pd.DataFrame,
                  variables=None, mechs=None, methods=None, missrates=None,
                  q_points: int = 101, sample_n: Optional[int] = None) -> List[plt.Figure]:
    """QQ-plot (quantile vs quantile) of ŷ against y."""
    df = _filter_rows(rows_long, variables, mechs, methods, missrates)
    if df.empty: return []
    figs = []
    by = ["variable"] + (["MissMech"] if "MissMech" in df.columns else []) + ["Method"] + (["MissRate"] if "MissRate" in df.columns else [])
    qs = np.linspace(0.01, 0.99, q_points)
    for keys, sub in df.groupby(by, dropna=False):
        y  = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        if sample_n and len(y) > sample_n:
            idx = np.random.default_rng(0).choice(len(y), size=sample_n, replace=False)
            y, yp = y[idx], yp[idx]
        yq  = np.quantile(y[~np.isnan(y)],  qs) if np.isfinite(y).any() else np.array([])
        ypq = np.quantile(yp[~np.isnan(yp)], qs) if np.isfinite(yp).any() else np.array([])
        fig = plt.figure()
        if yq.size and ypq.size:
            plt.plot(yq, ypq, marker="o", linestyle="", label="Quantiles (ŷ vs y)")
            m = np.nanmin([yq.min(), ypq.min()]); M = np.nanmax([yq.max(), ypq.max()])
            plt.plot([m, M], [m, M], linestyle="--", linewidth=1)
        t = " | ".join([str(k) for k in (keys if isinstance(keys, tuple) else (keys,))])
        plt.title(f"QQ-plot — {t}")
        plt.xlabel("Quantiles of y"); plt.ylabel("Quantiles of ŷ")
        plt.tight_layout(); plt.legend(loc="best")
        figs.append(fig)
    return figs

def make_box_plots(rows_long: pd.DataFrame,
                   variables=None, mechs=None, methods=None, missrates=None,
                   showfliers=False, sample_n: Optional[int] = None) -> List[plt.Figure]:
    """Side-by-side boxplots for y and ŷ."""
    df = _filter_rows(rows_long, variables, mechs, methods, missrates)
    if df.empty: return []
    figs = []
    by = ["variable"] + (["MissMech"] if "MissMech" in df.columns else []) + ["Method"] + (["MissRate"] if "MissRate" in df.columns else [])
    for keys, sub in df.groupby(by, dropna=False):
        y  = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        if sample_n and len(y) > sample_n:
            idx = np.random.default_rng(0).choice(len(y), size=sample_n, replace=False)
            y, yp = y[idx], yp[idx]
        fig = plt.figure()
        plt.boxplot([y, yp], labels=["Original (y)", "Imputed (ŷ)"], showfliers=showfliers)
        t = " | ".join([str(k) for k in (keys if isinstance(keys, tuple) else (keys,))])
        plt.title(f"Box-plots — {t}")
        plt.ylabel(sub["variable"].iloc[0])
        plt.tight_layout()
        figs.append(fig)
    return figs


# ================================== Export ===================================

def export_figs(figs: List[plt.Figure], basepath: str, prefix: str) -> list[str]:
    """Save matplotlib figures to PNG files and return their paths."""
    import os
    paths = []
    os.makedirs(basepath, exist_ok=True)
    for i, fig in enumerate(figs, 1):
        p = os.path.join(basepath, f"{prefix}_{i}.png")
        fig.savefig(p, dpi=150, bbox_inches='tight')
        paths.append(p)
    return paths



