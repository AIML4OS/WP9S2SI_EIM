# ============================================================ 
# lgbm_pipeline.py (generic CV-optimized, leakage-safe)
# One-file LightGBM toolkit:
#   - Helpers (categoricals, rare mapping, target transforms, saver)
#   - fast_tune_lightgbm (Optuna on sample; CV-only; leakage-safe; post-hoc min_data_in_bin)
#   - train_lightgbm_final (manual K-fold CV with per-fold rare mapping; picks best_iter; saves)
#   - load_trained_model / predict_with_inverse
# Files saved under: pot_raziskovanja/model/{prefix}_*.* 
# ============================================================

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import os
import json
import time
import tempfile
import pickle
import uuid

import numpy as np
import pandas as pd
import lightgbm as lgb

# --- Robust Optuna imports ---
import importlib

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import SuccessiveHalvingPruner
except Exception as e:
    raise RuntimeError(
        "Failed to import Optuna core. Install/repair with `pip install -U optuna` "
        "in the SAME interpreter you run code in.\n"
        f"Original error: {type(e).__name__}: {e}"
    )

# Trying both integration paths;
LightGBMPruningCallback = None
try:
    _mod = importlib.import_module("optuna.integration")
    LightGBMPruningCallback = getattr(_mod, "LightGBMPruningCallback", None)
except Exception:
    try:
        _mod = importlib.import_module("optuna_integration.lightgbm")
        LightGBMPruningCallback = getattr(_mod, "LightGBMPruningCallback", None)
    except Exception:
        LightGBMPruningCallback = None

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold

# -----------------------------
# Saving helpers
# -----------------------------

def _atomic_write_bytes(path: str, data: bytes) -> None:
    dirn = os.path.dirname(path) or "."
    os.makedirs(dirn, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=dirn, suffix=".tmp") as tmp:
        tmp.write(data)
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def _atomic_write_json(path: str, obj: Any) -> None:
    data = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    _atomic_write_bytes(path, data)

def _safe_save_booster(
    model_or_booster: Any,
    out_path_base: str,
    num_iteration: Optional[int] = None,
    retries: int = 5,
    sleep_s: float = 1.0,
) -> None:
    """Atomically save LightGBM booster as .pkl (fast) and .txt (optional, limited to num_iteration).
    - Writes {out_path_base}.pkl always.
    - Tries to write {out_path_base}.txt with num_iteration if provided; warns on failure but keeps .pkl.
    """
    booster = getattr(model_or_booster, "booster_", None) or model_or_booster

    dirn = os.path.dirname(out_path_base) or "."
    os.makedirs(dirn, exist_ok=True)

    # 1) .pkl (fast, compact)
    pkl_path = out_path_base + ".pkl"
    for i in range(retries):
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=dirn, suffix=".tmp") as tmp:
                pickle.dump(booster, tmp, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_name = tmp.name
            os.replace(tmp_name, pkl_path)
            break
        except Exception:
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass
            if i == retries - 1:
                raise
            time.sleep(sleep_s)

    # 2) .txt (slower; optional)
    txt_path = out_path_base + ".txt"
    try:
        tmp_name = os.path.join(dirn, f".{uuid.uuid4().hex}.tmp")
        booster.save_model(tmp_name, num_iteration=num_iteration)
        os.replace(tmp_name, txt_path)
    except Exception as e:
        try:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass
        print(f"[WARN] TXT save failed (kept .pkl): {e}")

def save_dataframe_csv_atomic(df: pd.DataFrame, path: str, **to_csv_kwargs) -> None:
    """Atomically save a DataFrame to CSV to avoid partial writes on networked or locked disks."""
    dirn = os.path.dirname(path) or "."
    os.makedirs(dirn, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=dirn, suffix=".tmp", mode="w", newline="") as tmp:
        df.to_csv(tmp, **({"index": False} | to_csv_kwargs))
        tmp_name = tmp.name
    os.replace(tmp_name, path)

# -----------------------------
# Target transform helpers
# -----------------------------

def _prepare_target_transform(y: np.ndarray, tt: Optional[Dict[str, Any]]):
    if not tt:
        return y, (lambda zhat, smear=1.0: zhat), {"enabled": False}

    method = tt.get("method", "log")
    eps = float(tt.get("eps", 0.0))

    if method == "log":
        if np.any(y + eps <= 0):
            raise ValueError("For method='log', all y+eps must be > 0.")
        z = np.log(y + eps)
        inv = lambda zhat, smear=1.0: np.exp(zhat) * smear - eps
    elif method == "log1p":
        if np.any(y < 0):
            raise ValueError("For method='log1p', all y must be >= 0.")
        z = np.log1p(y)
        inv = lambda zhat, smear=1.0: np.expm1(zhat) * smear
    else:
        raise ValueError("target_transform.method must be 'log' or 'log1p'.")

    info = {"enabled": True, "method": method, "eps": eps, "smearing": bool(tt.get("smearing", True))}
    return z, inv, info

def _compute_smearing_factor(z_true: np.ndarray, z_hat: np.ndarray) -> float:
    return float(np.mean(np.exp(z_true - z_hat)))

def _inverse_from_info(z_pred: np.ndarray, info: Optional[Dict[str, Any]]) -> np.ndarray:
    if not info or not info.get("enabled", False):
        return z_pred
    smear = float(info.get("smear", 1.0))
    eps = float(info.get("eps", 0.0))
    if info.get("method") == "log":
        return np.exp(z_pred) * smear - eps
    elif info.get("method") == "log1p":
        return np.expm1(z_pred) * smear
    return z_pred

def make_feval_rmse_original_scale(inv_fn):
    """
    Returns a LightGBM feval that computes RMSE on the ORIGINAL target scale.
      - Uses inverse WITHOUT smearing (smear=1.0), so it's comparable to CV/test metrics.
      - Assumes train_data.get_label() are the *transformed* targets (z).
    """
    def _feval(y_pred, train_data):
        z_true = train_data.get_label()
        y_true = inv_fn(z_true, smear=1.0)  # inverse WITHOUT smearing for fair metric
        y_hat  = inv_fn(y_pred, smear=1.0)
        rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
        return ("rmse_orig", rmse, False)  # (name, value, is_higher_better)
    return _feval

# -----------------------------
# Rare-categorical helpers (supposed to be leakage-safe)
# -----------------------------

_RARE_TOKEN = "__RARE__"
_UNSEEN_TOKEN = "__UNSEEN__"

def _prep_categoricals(
    X: pd.DataFrame,
    force_cat_cols: Optional[List[str]],
    rare_threshold: int
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cast object/category columns to categorical dtype.
    If rare_threshold > 0, collapse rare categories into 'Other' (single pass, NOT leakage-safe per-fold).
    If rare_threshold == 0, only cast dtype (no new categories are added).
    """
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    for c in (force_cat_cols or []):
        if c not in X.columns:
            raise ValueError(f"Forced categorical '{c}' not in features.")
        if c not in cat_cols:
            cat_cols.append(c)

    for c in cat_cols:
        X[c] = X[c].astype("category")
        if rare_threshold > 0:
            counts = X[c].value_counts()
            rare = counts[counts < rare_threshold].index
            if len(rare) > 0 and ("Other" not in X[c].cat.categories):
                X[c] = X[c].cat.add_categories(["Other"])
            if len(rare) > 0:
                X[c] = X[c].where(~X[c].isin(rare), "Other").cat.remove_unused_categories()
                new_cats = [k for k in X[c].cat.categories if k != "Other"] + ["Other"]
                X[c] = X[c].cat.set_categories(new_cats, ordered=True)

    return X, cat_cols

def _resolve_rare_threshold(
    n_train: int,
    thr_abs: Optional[int | float] = None,
    frac: Optional[float] = None
) -> int:
    """
    Resolve an absolute per-fold rare threshold from:
      - thr_abs: absolute count (>=1) if >0
      - frac: relative fraction in (0,1] if >0
    Returns 0 if both are None/<=0. Uses max(abs, ceil(frac*n_train)).
    """
    abs_cnt = 0
    if thr_abs is not None and float(thr_abs) > 0:
        abs_cnt = int(np.ceil(float(thr_abs)))
    rel_cnt = 0
    if frac is not None and float(frac) > 0:
        rel_cnt = int(np.ceil(float(frac) * n_train))
    eff = max(abs_cnt, rel_cnt)
    return int(eff) if eff > 0 else 0


def _fit_rare_map(X_trn: pd.DataFrame, cat_cols: List[str], threshold: int) -> Dict[str, Dict[str, Any]]:
    """
    Build per-column rare mapping on TRAIN only.
    Returns dict: col -> {"rare": set(rare_values), "keep": set(kept_values)}
    """
    rare_map: Dict[str, Dict[str, Any]] = {}
    for c in cat_cols:
        s = X_trn[c].astype("category")
        vc = s.value_counts(dropna=False)
        rare_vals = set(vc[vc < threshold].index)
        keep_vals = set(vc.index) - rare_vals
        rare_map[c] = {"rare": rare_vals, "keep": keep_vals}
    return rare_map

def _apply_rare_map_pair(
    X_trn: pd.DataFrame,
    X_val: pd.DataFrame,
    rare_map: Dict[str, Dict[str, Any]],
    use_unseen: bool = True,
    rare_token: str = _RARE_TOKEN,
    unseen_token: str = _UNSEEN_TOKEN,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applying the same mapping to TRAIN and VALID:
      - TRAIN: values in 'rare' -> rare_token
      - VALID: values not in KEEP -> unseen_token (or rare_token if use_unseen=False)
    Ensures both columns have matching categorical dtype categories.
    """
    X_trn = X_trn.copy()
    X_val = X_val.copy()
    for c, info in rare_map.items():
        rare_vals = info["rare"]
        keep_vals = info["keep"]

        s_tr = X_trn[c].astype("category")
        if rare_token not in s_tr.cat.categories:
            s_tr = s_tr.cat.add_categories([rare_token])
        s_tr = s_tr.where(~s_tr.isin(rare_vals), rare_token)

        s_va = X_val[c].astype("category")
        replace_token = unseen_token if use_unseen else rare_token
        if replace_token not in s_va.cat.categories:
            s_va = s_va.cat.add_categories([replace_token])
        s_va = s_va.where(s_va.isin(keep_vals | {rare_token}), replace_token)

        cats = list(pd.Index(s_tr.cat.categories).union(pd.Index(s_va.cat.categories)))
        s_tr = s_tr.cat.set_categories(cats)
        s_va = s_va.cat.set_categories(cats)

        X_trn[c] = s_tr
        X_val[c] = s_va

    return X_trn, X_val

# -----------------------------
# Learning-rate schedule helper
# -----------------------------

def _build_lr_callback(max_boost_round: int, lr_schedule: Optional[Dict[str, Any]]):
    if not lr_schedule:
        return []
    t = lr_schedule.get("type", "linear")
    if t == "linear":
        start = float(lr_schedule.get("start", 0.05))
        end = float(lr_schedule.get("end", 0.005))

        def lr_decay(i: int):
            a = min(max(i / max_boost_round, 0.0), 1.0)
            return start + (end - start) * a

        return [lgb.reset_parameter(learning_rate=lr_decay)]
    if t == "tail":
        base = float(lr_schedule.get("base", 0.05))
        tail = float(lr_schedule.get("tail", 0.01))
        start_frac = float(lr_schedule.get("start_frac", 0.7))

        def lr_tail(i: int):
            return base if (i / max_boost_round) < start_frac else tail

        return [lgb.reset_parameter(learning_rate=lr_tail)]
    return []

# ETA / progress callback 
# -----------------------------
def make_eta_callback(total_rounds: int, period: int = 200):
    start = time.time()
    def _eta(env):
        i = getattr(env, "iteration", 0)  # 1-based
        if i <= 0:
            return
        if (i % period != 0) and (i != total_rounds):
            return

        elapsed = time.time() - start
        rate = i / max(elapsed, 1e-9)
        eta = (total_rounds - i) / rate if rate > 0 else float("inf")

        metric_txt = ""
        try:
            reslist = getattr(env, "evaluation_result_list", None)
            if reslist:
                res0 = reslist[0]
                if len(res0) >= 3:
                    metric_name = str(res0[1])
                    metric_val = float(res0[2])
                elif len(res0) == 2:
                    metric_name = str(res0[0])
                    metric_val = float(res0[1])
                else:
                    metric_name, metric_val = "metric", float("nan")
                metric_txt = f" | {metric_name}={metric_val:.6g}"
        except Exception:
            pass

        print(f"[ETA] iter {i}/{total_rounds} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s{metric_txt}", flush=True)
    return _eta


# Optuna pruner helper 
# -----------------------------
def make_optuna_fold_pruner(trial, metric_name: str, fold_offset: int):
    """
    LightGBM callback that reports metric for valid_0 at each iteration with a
    globally unique, strictly increasing step = fold_offset + iteration.
    """
    last_step = [-1]
    def _cb(env):
        it = getattr(env, "iteration", None)
        if it is None:
            return
        step = int(fold_offset + it)
        # strictly increasing guard
        if step <= last_step[0]:
            return
        last_step[0] = step

        val = None
        # env.evaluation_result_list: list of tuples (data_name, eval_name, result, is_higher_better)
        for data_name, eval_name, result, _ in getattr(env, "evaluation_result_list", []):
            if data_name == "valid_0" and eval_name == metric_name:
                val = float(result)
                break
        if val is None:
            return

        trial.report(val, step=step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return _cb


# -----------------------------
# Regression stratified CV helpers (quantiles / deciles)
# -----------------------------

def _bin_continuous_target(
    y: np.ndarray,
    strategy: str = "quantiles",
    n_bins: int = 10,
) -> np.ndarray:
    """
    Returns integer bin labels for continuous y.
    strategy in {"quantiles","deciles","fixed"}.
    """
    if strategy == "deciles":
        n_bins = 10
        strategy = "quantiles"

    y = np.asarray(y, dtype=float)
    if strategy == "quantiles":
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        return bins.astype(int)
    elif strategy == "fixed":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y, qs)
        edges = np.unique(edges)
        if edges.size <= 2:
            return np.zeros_like(y, dtype=int)
        b = np.digitize(y, edges[1:-1], right=True)
        return b.astype(int)
    else:
        raise ValueError("reg_stratify.strategy must be 'quantiles', 'deciles', or 'fixed'.")

def _make_reg_stratified_folds(
    y: np.ndarray,
    n_splits: int,
    strategy: str = "quantiles",
    n_bins: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
):
    """
    Create folds for regression via stratification on binned targets.
    Returns a list of (train_idx, valid_idx) tuples.
    """
    y_bins = _bin_continuous_target(y, strategy=strategy, n_bins=n_bins)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = [(tr_idx, val_idx) for tr_idx, val_idx in skf.split(np.zeros_like(y_bins), y_bins)]
    return folds

# -----------------------------
# Utilities for Dataset binning params
# -----------------------------

_BINNING_KEYS = (
    "min_data_in_bin",
    "max_bin",
    "max_bin_by_feature",
    "bin_construct_sample_cnt",
    "bin_construct_sample_cnt_per_feature",
    "max_bin_plus_one",
)

def _extract_binning_params(p: Dict[str, Any]) -> Dict[str, Any]:
    """Pick only the binning-related keys that must be fixed at Dataset construction."""
    return {k: p[k] for k in _BINNING_KEYS if k in p}

# -----------------------------
# Post-hoc sweep for min_data_in_bin (after Optuna)
# -----------------------------
"""
# This post-hoc min_data_in_bin sweep uses standard lgb.cv() for simplicity and speed.
# It does not replicate the leakage-safe per-fold rare category mapping used during tuning.
# Therefore, the evaluation procedure is slightly different from the main tuning loop,
# representing a deliberate methodological simplification.
    """

def select_min_data_in_bin_posthoc(
    base_params: dict,
    X: pd.DataFrame,
    label: np.ndarray,
    cat_cols: Optional[List[str]] = None,
    candidates: Tuple[int, ...] = (10, 20, 50, 100),
    best_iter: Optional[int] = None,
    cv_nfold: int = 5,
    seed: int = 42,
    early_stopping_rounds: int = 100,
    metric_key_suffix: str = "-mean",
    reg_lambda_floor_when_small_bin: float = 0.5,
    verbose: bool = True,
) -> Tuple[dict, int, float]:
    """
    Freeze all params from Optuna and pick min_data_in_bin via a tiny CV sweep.
    Enforces a lambda_l2 floor when bins are very fine.
    Dataset per candidate is rebuild, passing the same min_data_in_bin to Dataset.
    """
    frozen = dict(base_params)
    total_rounds = int(frozen.get("num_boost_round") or (best_iter or 6000))

    def _metric_key(cv_dict: Dict[str, List[float]]) -> str:
        return next(k for k in cv_dict.keys() if k.endswith(metric_key_suffix))

    best_score = np.inf
    best_bin   = None
    best_lam2  = None
    best_len   = None

    def _get_lam2(p):
        return float(p.get("lambda_l2", p.get("reg_lambda", 0.0)))

    for mib in candidates:
        p = dict(frozen)
        p["min_data_in_bin"] = int(mib)

        lam2 = _get_lam2(p)
        if mib <= 20 and lam2 < reg_lambda_floor_when_small_bin:
            lam2 = reg_lambda_floor_when_small_bin
        p["lambda_l2"] = lam2
        p["reg_lambda"] = lam2

        # Rebuild Dataset with matching binning params for this candidate
        dtrain = lgb.Dataset(
            X, label,
            categorical_feature=cat_cols,
            params={"min_data_in_bin": int(mib)}
        )

        cv = lgb.cv(
            params=p,
            train_set=dtrain,
            num_boost_round=total_rounds,
            nfold=cv_nfold,
            stratified=False,
            seed=seed,
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.callback.log_evaluation(period=0),
            ],
        )
        key = _metric_key(cv)
        score = float(cv[key][-1])
        if verbose:
            print(f"[posthoc] min_data_in_bin={mib} | {key}={score:.6g} | len={len(cv[key])}")
        if score < best_score:
            best_score = score
            best_bin   = mib
            best_lam2  = lam2
            best_len   = len(cv[key])

    updated = dict(frozen)
    if best_bin is not None:
        updated["min_data_in_bin"] = int(best_bin)
    if best_lam2 is not None:
        updated["lambda_l2"] = float(best_lam2)
        updated["reg_lambda"] = float(best_lam2)
    return updated, int(best_len or total_rounds), float(best_score)

# -----------------------------
def fast_tune_lightgbm(
    df_train: pd.DataFrame,
    target_col: str,
    features: List[str],
    task: str = "regression",
    target_transform: Optional[Dict[str, Any]] = None,
    subsample_n: int = 100_000,
    cv_nfold: int = 3,
    n_trials: int = 100,
    sampler_seed: int = 42,
    early_stopping_rounds: int = 50,
    rare_threshold: int = 0,   # kept for BC, not used globally (avoid leakage)
    pot_raziskovanja: str = ".",
    save_prefix: str = "GBM",
    verbose: bool = True,
    boost_cap_min: int = 1_000,
    boost_cap_max: int = 15_000,
    boost_from_average: bool = True,     # your preferred default
    timeout: Optional[int] = 1800,
    # regression stratification (optional)
    reg_stratify: Optional[Dict[str, Any]] = None,
    # post-hoc sweep for min_data_in_bin
    posthoc_minbin: bool = True,
    posthoc_minbin_candidates: Tuple[int, ...] = (10, 20, 50, 100),
    # leakage-safe rare mapping during tuning (per fold)
    rare_threshold_tune: Optional[int | float] = None,  # absolute count per fold (>=1)
    rare_fraction_tune: Optional[float] = None,         # relative fraction per fold in (0,1]
    # NEW: reproducibility knobs for tuning
    deterministic_tune: str = "off",   # "off" | "light" | "strict"
    id_col: Optional[str] = None,      # stable sort key (optional)
) -> Dict[str, Any]:
    """
    Fast Optuna tuning for LightGBM with leakage-safe option for rare categories during CV.
    - deterministic_tune:
        "off"   : fastest, default (multi-thread, nondeterministic)
        "light" : keep multi-thread, fix seeds + stable sorting (almost reproducible)
        "strict": single-thread + deterministic=True + stable sorting (bitwise reproducible, slower)
    """

    def _strip_conflicting_binning(p: Dict[str, Any], v: bool = False) -> Dict[str, Any]:
        if p is None:
            return {}
        p = dict(p)
        removed = []
        for k in ("max_bin","max_bin_by_feature","bin_construct_sample_cnt",
                  "bin_construct_sample_cnt_per_feature","max_bin_plus_one"):
            if k in p:
                removed.append((k, p.pop(k)))
        if v and removed:
            print(f"[lgbm] Stripped binning params: {removed}")
        return p

    # paths
    model_dir = os.path.join(pot_raziskovanja, "model")
    os.makedirs(model_dir, exist_ok=True)
    params_path = os.path.join(model_dir, f"{save_prefix}_params.json")
    model_path_base = os.path.join(model_dir, f"{save_prefix}_model")

    # folds heuristic
    n_samples = len(df_train)
    cv_nfold_eff = 5 if n_samples < 50_000 else cv_nfold
    if verbose:
        print(f"[fast_tune] Rows={n_samples} → CV folds={cv_nfold_eff}")

    # subsample (with proper stratification)
    df = df_train
    if subsample_n is not None and n_samples > subsample_n:
        if task == "classification":
            parts = []
            for cls, g in df.groupby(target_col):
                k = max(1, int(len(g) * subsample_n / n_samples))
                parts.append(g.sample(n=k, random_state=sampler_seed))
            df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=sampler_seed)
        else:
            if reg_stratify and reg_stratify.get("enabled", False):
                strat = dict(strategy=reg_stratify.get("strategy", "quantiles"),
                             n_bins=int(reg_stratify.get("n_bins", 10)))
                y_bins = _bin_continuous_target(df[target_col].values, **strat)
                parts = []
                tmp = pd.DataFrame({"bin": y_bins}, index=df.index)
                for _, idx in tmp.groupby("bin").groups.items():
                    if len(idx) == 0:
                        continue
                    k = max(1, int(len(idx) * subsample_n / n_samples))
                    parts.append(df.loc[idx].sample(n=k, random_state=sampler_seed))
                df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=sampler_seed)
            else:
                df = df.sample(n=subsample_n, random_state=sampler_seed)
        if verbose:
            print(f"[fast_tune] Subsampled → {len(df)} rows")

    # Optional stable ordering (helps reproducibility in "light"/"strict")
    if id_col is not None and id_col in df.columns and deterministic_tune in ("light", "strict"):
        df = df.sort_values(id_col).reset_index(drop=True)

    # data & categoricals (NO global rare grouping)
    X = df[features].copy()
    y = df[target_col].values
    X, cat_cols = _prep_categoricals(X, force_cat_cols=None, rare_threshold=0)
    if verbose:
        print(f"[fast_tune] Features={len(features)} (categorical={len(cat_cols)})")

    # target transform (optional)
    if task == "regression" and target_transform is not None:
        z, inv, tt_info = _prepare_target_transform(y, target_transform)
        label = z
        feval_orig = make_feval_rmse_original_scale(inv)
        use_feval_orig = True
        if verbose:
            print(f"[fast_tune] Using target transform: {target_transform.get('method', 'log')}")
    else:
        label = y
        feval_orig = None
        use_feval_orig = False

    # objective & metric
    uniq = np.unique(y)
    base_obj = "regression" if task == "regression" else ("binary" if uniq.size == 2 else "multiclass")
    base_metric = "rmse" if task == "regression" else ("binary_logloss" if base_obj == "binary" else "multi_logloss")
    if verbose:
        print(f"[fast_tune] Objective={base_obj}  Metric={base_metric}  Trials={n_trials}")

    # min_data_in_leaf candidates (coarse, good for ~500k)
    def _min_leaf_candidates_by_n(n: int) -> list[int]:
        if n <= 200_000:   return [50, 100, 200, 400, 800]
        if n <= 700_000:    return [100, 200, 400, 800, 1200]
        if n > 700_000:    return [200, 400, 800, 1200, 2000]   # ~500k+

    _mleaf_cands = _min_leaf_candidates_by_n(len(df))
    if verbose:
        print(f"[fast_tune] Dynamic min_data_in_leaf candidates (n={len(df)}): {_mleaf_cands}")

    # boosting cap
    def _cap_from_lr(lr: float) -> int:
        cap = 8000 if lr <= 0.07 else 6000
        cap = max(boost_cap_min, cap)
        cap = min(boost_cap_max, cap)
        return int(cap)

    # search space — coarse steps for speed on ~500k rows
    def _build_space(trial: "optuna.Trial") -> Dict[str, Any]:
        lr = trial.suggest_categorical("learning_rate", [0.05, 0.07, 0.10])
        p = {
            "objective":               base_obj,
            "boost_from_average":      bool(boost_from_average),
            "metric":                  base_metric,
            "num_leaves":              trial.suggest_categorical("num_leaves", [63, 127, 255]),
            "max_depth":               trial.suggest_categorical("max_depth",  [6, 8, 10]),
            "min_data_in_leaf":        trial.suggest_categorical("min_data_in_leaf", _mleaf_cands),
            "min_gain_to_split":       trial.suggest_categorical("min_gain_to_split", [0.0, 0.05, 0.10]),
            "min_sum_hessian_in_leaf": trial.suggest_categorical("min_sum_hessian_in_leaf", [1e-3, 1e-2]),
            # Regularization (coarse)
            "lambda_l1":               trial.suggest_float("lambda_l1", 0.0, 5.0, step=0.5),
            "lambda_l2":               trial.suggest_float("lambda_l2", 0.0, 5.0, step=0.5),
            # (NEW) Feature & bagging subsampling — coarse grid for speed
            "feature_fraction":        trial.suggest_categorical("feature_fraction", [0.6, 0.7, 0.8, 0.9]),
            "bagging_fraction":        trial.suggest_categorical("bagging_fraction", [0.6, 0.7, 0.8, 0.9]),
            "bagging_freq":            trial.suggest_categorical("bagging_freq", [1, 3]),  # >0 to activate bagging
            # LR & boosting cap
            "learning_rate":           lr,
            "num_boost_round":         _cap_from_lr(lr),
            # Stability & speed
            "feature_pre_filter":      False,
            "force_col_wise":          True,
            "num_threads":             -1,
            "deterministic":           False,
            "seed":                    sampler_seed,
            "feature_fraction_seed":   sampler_seed,
            "bagging_seed":            sampler_seed,
            "data_random_seed":        sampler_seed,
            "drop_seed":               sampler_seed,
            "first_metric_only":       True,
        }
        # Determinism modes for tuning
        if deterministic_tune == "strict":
            p.update({
                "deterministic": True,
                "num_threads": 1,
                "seed": sampler_seed,
                "feature_fraction_seed": sampler_seed,
                "bagging_seed": sampler_seed,
                "data_random_seed": sampler_seed,
                "drop_seed": sampler_seed,
            })
        elif deterministic_tune == "light":
            # keep multithread, but fix seeds (already set) — nearly reproducible
            pass

        if base_obj == "multiclass":
            p["num_class"] = int(uniq.size)
        return p

    # Optuna setup
    sampler = TPESampler(seed=sampler_seed)
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=3, min_early_stopping_rate=0)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    # Prebuild folds (used by manual CV path)
    if task == "regression" and reg_stratify and reg_stratify.get("enabled", False):
        folds = _make_reg_stratified_folds(
            y=label,
            n_splits=int(cv_nfold_eff),
            strategy=reg_stratify.get("strategy", "quantiles"),
            n_bins=int(reg_stratify.get("n_bins", 10)),
            shuffle=bool(reg_stratify.get("shuffle", True)),
            random_state=int(reg_stratify.get("random_state", sampler_seed)),
        )
    else:
        folds = list(KFold(n_splits=int(cv_nfold_eff), shuffle=True, random_state=sampler_seed).split(X))

    def _objective(trial: "optuna.Trial") -> float:
        params = _strip_conflicting_binning(_build_space(trial), v=bool(verbose))
        ds_bin_params = _extract_binning_params(params)

        total = int(params.get("num_boost_round") or (8000 if params.get("learning_rate", 0.07) <= 0.07 else 6000))
        params["num_boost_round"] = total

        any_rare = (
            (rare_threshold_tune is not None and float(rare_threshold_tune) > 0) or
            (rare_fraction_tune is not None and float(rare_fraction_tune) > 0)
        )
        metric_name = "rmse_orig" if (task == "regression" and target_transform is not None) \
                      else (params.get("metric") or base_metric)

        # Fast path: lightgbm.cv (no per-fold rare mapping)
        if not any_rare:
            lgb_data = lgb.Dataset(X, label, categorical_feature=cat_cols, params=(ds_bin_params or None))
            callbacks = [
                lgb.callback.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.callback.log_evaluation(period=0),
                make_eta_callback(total_rounds=total, period=max(100, total // 40)),
            ]
            if LightGBMPruningCallback is not None:
                callbacks.insert(1, LightGBMPruningCallback(trial, metric_name))

            params_for_cv = dict(params)
            cv_kwargs = dict(train_set=lgb_data, num_boost_round=total, seed=sampler_seed, callbacks=callbacks)
            if use_feval_orig:
                params_for_cv["metric"] = "None"
                cv_kwargs["feval"] = feval_orig
            cv_kwargs["params"] = params_for_cv

            if task == "regression" and reg_stratify and reg_stratify.get("enabled", False):
                cv = lgb.cv(folds=folds, stratified=False, **cv_kwargs)
            else:
                cv = lgb.cv(nfold=int(cv_nfold_eff), stratified=(task == "classification"), **cv_kwargs)

            key = next(k for k in cv.keys() if k.endswith("-mean"))
            trial.set_user_attr("best_iter", int(len(cv[key])))
            return float(cv[key][-1])

        # Manual path: per-fold rare mapping (leakage-safe)
        scores = []
        best_iters = []

        for step_i, (tr_idx, va_idx) in enumerate(folds, start=0):
            Xtr, Xva = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
            ytr, yva = label[tr_idx], label[va_idx]

            # Stable ordering inside folds when requested ("light"/"strict")
            if id_col is not None and id_col in df.columns and deterministic_tune in ("light", "strict"):
                tr_ids = df.iloc[tr_idx][id_col].to_numpy()
                va_ids = df.iloc[va_idx][id_col].to_numpy()
                order_tr = np.argsort(tr_ids, kind="mergesort")
                order_va = np.argsort(va_ids, kind="mergesort")
                Xtr = Xtr.iloc[order_tr].reset_index(drop=True)
                Xva = Xva.iloc[order_va].reset_index(drop=True)
                ytr = ytr[order_tr]
                yva = yva[order_va]

            # leakage-safe rare mapping (per-fold)
            thr_cnt = _resolve_rare_threshold(len(Xtr), thr_abs=rare_threshold_tune, frac=rare_fraction_tune)
            if thr_cnt > 0 and len(cat_cols) > 0:
                rmap = _fit_rare_map(Xtr, cat_cols, threshold=int(thr_cnt))
                Xtr, Xva = _apply_rare_map_pair(Xtr, Xva, rmap, use_unseen=True)

            dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols, params=(ds_bin_params or None))
            dva = lgb.Dataset(Xva, yva, categorical_feature=cat_cols, params=(ds_bin_params or None), reference=dtr)

            feval = None
            params_cv = dict(params)
            if use_feval_orig:
                params_cv["metric"] = "None"   # so feval name drives callbacks
                feval = feval_orig

       # Unique, strictly increasing step space across folds (avoid duplicate steps)
            fold_offset = (step_i + 1) * (total + 1000)

            callbacks = [
                lgb.callback.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.callback.log_evaluation(period=0),
                make_optuna_fold_pruner(trial, metric_name, fold_offset=fold_offset),
            ]

            bst = lgb.train(
                params_cv,
                dtr,
                num_boost_round=total,
                valid_sets=[dva],
                valid_names=["valid_0"],
                feval=feval,
                callbacks=callbacks,
            )

            # take score and iterations
            try:
                score = float(bst.best_score["valid_0"][metric_name])
            except KeyError:
                vk = list(bst.best_score.get("valid_0", {}).keys())
                score = float(bst.best_score["valid_0"][vk[0]]) if vk else float("nan")

            scores.append(score)
            best_iters.append(int(bst.best_iteration))

        mean_score = float(np.mean(scores)) if scores else float("inf")
        median_iter = int(np.median(best_iters)) if best_iters else total
        trial.set_user_attr("best_iter", median_iter)
        return mean_score

    if verbose:
        print("[fast_tune] Starting Optuna study …")

    study.optimize(_objective, n_trials=int(n_trials), timeout=timeout, show_progress_bar=bool(verbose))

    # collect & freeze
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    fixed_fields = {
        "objective":             base_obj,
        "metric":                base_metric,
        "feature_pre_filter":    False,
        "force_col_wise":        True,
        "num_threads":           -1,
        "deterministic":         False,
        "seed":                  sampler_seed,
        "feature_fraction_seed": sampler_seed,
        "bagging_seed":          sampler_seed,
        "data_random_seed":      sampler_seed,
        "drop_seed":             sampler_seed,
        "first_metric_only":     True,
        "boost_from_average":    bool(boost_from_average),
    }
    if base_obj == "multiclass":
        fixed_fields["num_class"] = int(uniq.size)

    best_params_full = {**fixed_fields, **best_params}
    best_iter = int(best_trial.user_attrs.get("best_iter", best_params_full.get("num_boost_round", 10_000)))

    # post-hoc sweep for min_data_in_bin
    if posthoc_minbin:
        if verbose:
            print("[fast_tune] Post-hoc sweep for min_data_in_bin …")
        best_params_full, best_iter_post, _ = select_min_data_in_bin_posthoc(
            base_params=best_params_full,
            X=X, label=label, cat_cols=cat_cols,
            candidates=posthoc_minbin_candidates,
            best_iter=best_iter,
            cv_nfold=cv_nfold_eff,
            seed=sampler_seed,
            early_stopping_rounds=early_stopping_rounds,
            metric_key_suffix="-mean",
            reg_lambda_floor_when_small_bin=0.5,
            verbose=bool(verbose),
        )
        best_iter = int(best_iter_post)

    # save params
    params_to_save = {**best_params_full, "best_iter": int(best_iter)}
    _atomic_write_json(params_path, params_to_save)
    if verbose:
        print(f"[fast_tune] Best params saved → {params_path}")
        try:
            print(f"[fast_tune] Best value: {study.best_value:.6g}")
            print(f"[fast_tune] Best trial params: {best_params}")
        except Exception:
            pass

    # tiny artifact model (for FI)
    tuned_ds = lgb.Dataset(X, label, categorical_feature=cat_cols, params=_extract_binning_params(best_params_full) or None)
    tuned = lgb.train(best_params_full, tuned_ds, num_boost_round=int(best_iter), callbacks=[lgb.callback.log_evaluation(period=0)])

    try:
        feat_names = tuned.feature_name()
        gain = tuned.feature_importance(importance_type="gain")
        df_imp = pd.DataFrame({"feature": feat_names, "gain": gain}).sort_values("gain", ascending=False)
        save_dataframe_csv_atomic(df_imp, os.path.join(model_dir, f"{save_prefix}_tune_feature_importance.csv"))
        print(f"[fast_tune] Saved tuning feature importances → {model_dir}/{save_prefix}_tune_feature_importance.csv")
    except Exception as e:
        print(f"[fast_tune][WARN] Could not save tuning feature importances: {e}")

    _safe_save_booster(tuned, model_path_base, num_iteration=int(best_iter))
    if verbose:
        print(f"[fast_tune] Tuned model saved → {model_path_base}.pkl (+ .txt)")

    return params_to_save


# -----------------------------
# FINAL TRAIN (manual CV with per-fold rare mapping; fast saves)
# -----------------------------
def train_lightgbm_final(
    df_train: pd.DataFrame,
    target_col: str,
    features: List[str],
    task: str = "regression",
    params: Optional[Dict[str, Any]] = None,
    target_transform: Optional[Dict[str, Any]] = None,
    force_cat_cols: Optional[List[str]] = None,
    rare_threshold: Optional[int | float] = 0,   # can be absolute (>0) or 0/None to disable
    rare_fraction: Optional[float] = None,       # optional fraction in (0,1]
    boost_from_average: bool = True,
    cv_nfold: int = 5,
    early_stopping_rounds: int = 100,
    lr_schedule: Optional[Dict[str, Any]] = None,
    deterministic: bool = True,
    random_state: int = 42,
    pot_raziskovanja: str = ".",
    save_prefix: str = "GBM",
    save_model: bool = True,
    save_params: bool = True,
    verbose: Optional[int] = 100,
    reg_stratify: Optional[Dict[str, Any]] = None,
    id_col: Optional[str] = None,                # NEW: stable sort key for determinism
) -> Tuple[lgb.Booster, np.ndarray, Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any], Dict[str, Any]]:
    """
    Manual K-fold CV to pick best_iter.
    Leakage-safe per-fold rare mapping if rare_threshold/rare_fraction > 0.
    Deterministic training: if id_col is provided, train/valid and final data are
    stably sorted by id_col and targets are realigned accordingly.
    Returns: model, y_pred_infer, train_metrics, cv_metrics, cat_cols, cv_results, tt_info
    """
    if params is None:
        params = {}
    if force_cat_cols is None:
        force_cat_cols = []

    # (Optional) sort full frame up-front for a stable baseline order
    if id_col is not None and id_col in df_train.columns:
        df_train = df_train.sort_values(id_col).reset_index(drop=True)

    X_full = df_train[features].copy()
    y_full = df_train[target_col].values

    # Only cast to categorical here; no global rare grouping (avoid leakage)
    X_full, cat_cols = _prep_categoricals(X_full, force_cat_cols=force_cat_cols, rare_threshold=0)

    # Objectives & metrics defaults
    if task == "regression":
        params.setdefault("objective", "regression")
        params.setdefault("metric", "rmse")
    elif task == "classification":
        ncls = int(np.unique(y_full).size)
        if ncls > 2:
            params.setdefault("objective", "multiclass")
            params.setdefault("metric", "multi_logloss")
            params["num_class"] = ncls
        else:
            params.setdefault("objective", "binary")
            params.setdefault("metric", "binary_logloss")
    else:
        raise ValueError("task must be 'regression' or 'classification'.")

    # Determinism / speed knobs
    if deterministic:
        params.update({
            "seed": random_state,
            "feature_fraction_seed": random_state,
            "bagging_seed": random_state,
            "data_random_seed": random_state,
            "drop_seed": random_state,
            "deterministic": True,
            "num_threads": 1,  # single-threaded for bitwise reproducibility
        })
    else:
        params.setdefault("num_threads", -1)
        params.setdefault("seed", random_state)
        params.setdefault("feature_fraction_seed", random_state)
        params.setdefault("bagging_seed", random_state)
        params.setdefault("data_random_seed", random_state)
        params.setdefault("drop_seed", random_state)

    params.setdefault("force_col_wise", True)
    params.setdefault("first_metric_only", True)
    params.setdefault("verbosity", -1)

    # If distribution is not normal, False is default, unless True is passed
    params["boost_from_average"] = bool(boost_from_average)

    # Target transform (if requested)
    if task == "regression" and target_transform is not None:
        z_full, inv, tt_info = _prepare_target_transform(y_full, target_transform)
        use_tt = True
        feval_orig = make_feval_rmse_original_scale(inv)
        use_feval_orig = True
    else:
        z_full, inv, tt_info = y_full, (lambda zhat, smear=1.0: zhat), {"enabled": False}
        use_tt = False
        feval_orig = None
        use_feval_orig = False

    # Learning rate schedule + ETA
    max_boost_round = int(params.get("num_boost_round", 10_000) or 10_000)
    callbacks_common = []
    if verbose is not None:
        callbacks_common.append(lgb.callback.log_evaluation(period=verbose))
    callbacks_common.append(lgb.callback.early_stopping(stopping_rounds=early_stopping_rounds))
    callbacks_common += _build_lr_callback(max_boost_round, lr_schedule)
    callbacks_common.append(make_eta_callback(total_rounds=max_boost_round, period=max(100, max_boost_round // 40)))

    # Folds
    if task == "regression" and reg_stratify and reg_stratify.get("enabled", False):
        folds = _make_reg_stratified_folds(
            y=z_full if use_tt else y_full,
            n_splits=cv_nfold,
            strategy=reg_stratify.get("strategy", "quantiles"),
            n_bins=int(reg_stratify.get("n_bins", 10)),
            shuffle=bool(reg_stratify.get("shuffle", True)),
            random_state=int(reg_stratify.get("random_state", random_state)),
        )
    else:
        folds = list(KFold(n_splits=cv_nfold, shuffle=True, random_state=random_state).split(X_full))

    metric_name = "rmse_orig" if use_feval_orig else (params.get("metric") or "rmse")
    fold_scores: List[float] = []
    fold_best_iter: List[int] = []

    ds_binning = _extract_binning_params(params)

    # Manual CV (leakage-safe) with per-fold rare mapping + stable sorting by id_col
    for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_trn, X_val = X_full.iloc[tr_idx].copy(), X_full.iloc[va_idx].copy()
        y_trn = (z_full if use_tt else y_full)[tr_idx]
        y_val = (z_full if use_tt else y_full)[va_idx]

        # --- Stable ordering for strict determinism (sort by id_col if provided) ---
        if id_col is not None and id_col in df_train.columns:
            # Build argsort on IDs for train and valid, then reorder X and y accordingly
            tr_ids = df_train.iloc[tr_idx][id_col].to_numpy()
            va_ids = df_train.iloc[va_idx][id_col].to_numpy()
            order_tr = np.argsort(tr_ids, kind="mergesort")
            order_va = np.argsort(va_ids, kind="mergesort")

            X_trn = X_trn.iloc[order_tr].reset_index(drop=True)
            X_val = X_val.iloc[order_va].reset_index(drop=True)
            y_trn = y_trn[order_tr]
            y_val = y_val[order_va]

        # Leakage-safe rare mapping per fold (order does not change counts)
        thr = _resolve_rare_threshold(len(X_trn), thr_abs=rare_threshold, frac=rare_fraction)
        if thr > 0 and len(cat_cols) > 0:
            rare_map = _fit_rare_map(X_trn, cat_cols, threshold=int(thr))
            X_trn, X_val = _apply_rare_map_pair(X_trn, X_val, rare_map, use_unseen=True)

        dtr = lgb.Dataset(X_trn, y_trn, categorical_feature=cat_cols, params=(ds_binning or None))
        dva = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols, params=(ds_binning or None), reference=dtr)

        params_cv = dict(params)
        callbacks = list(callbacks_common)

        feval = None
        if use_feval_orig:
            params_cv["metric"] = "None"
            feval = feval_orig

        bst = lgb.train(params_cv, dtr, num_boost_round=max_boost_round, valid_sets=[dva], feval=feval, callbacks=callbacks)

        try:
            score = float(bst.best_score["valid_0"][metric_name])
        except KeyError:
            vk = list(bst.best_score.get("valid_0", {}).keys())
            score = float(bst.best_score["valid_0"][vk[0]]) if vk else float("nan")

        fold_scores.append(score)
        fold_best_iter.append(int(bst.best_iteration))
        if verbose:
            print(f"[final-CV] Fold {fold_id}/{cv_nfold} | {metric_name}={score:.6g} | best_iter={bst.best_iteration}")

    best_iter = int(np.median(fold_best_iter)) if fold_best_iter else max_boost_round
    cv_mean = float(np.mean(fold_scores)) if fold_scores else float("nan")
    cv_std  = float(np.std(fold_scores)) if fold_scores else float("nan")

    cv_metrics = {
        metric_name + "-mean": cv_mean,
        metric_name + "-std":  cv_std,
        "_best_iter": int(best_iter),
        "_cap": int(max_boost_round),
        "_cap_ratio": float(best_iter / max_boost_round if max_boost_round else 0.0),
        "_target_transform_info": tt_info,
    }
    cv_results = {
        "metric_name": metric_name,
        "fold_metrics": [float(s) for s in fold_scores],
        "fold_best_iter": [int(bi) for bi in fold_best_iter],
        "rare_threshold": int(np.ceil(float(rare_threshold))) if rare_threshold and float(rare_threshold) > 1 else None,
        "rare_fraction": float(rare_fraction) if rare_fraction is not None else (
            float(rare_threshold) if (rare_threshold is not None and 0 < float(rare_threshold) <= 1) else None
        ),
    }

    if (cv_metrics["_cap_ratio"] > 0.9) and verbose:
        print("⚠️ best_iter is close to the cap; consider increasing num_boost_round.")
    elif (cv_metrics["_cap_ratio"] < 0.2) and verbose:
        print("ℹ️ best_iter is far from the cap; you could lower num_boost_round.")

    # --- Final train on FULL train set up to best_iter ---
    X_final = X_full.copy()

    # Stable ordering for final fit (sort by id_col if provided) + realign targets
    if id_col is not None and id_col in df_train.columns:
        fin_ids = df_train[id_col].to_numpy()
        order_fin = np.argsort(fin_ids, kind="mergesort")
        X_final = X_final.iloc[order_fin].reset_index(drop=True)
        y_full = y_full[order_fin]
        if use_tt:
            z_full = z_full[order_fin]

    # Optional rare mapping on FULL train (leakage-safe because applied only to train)
    thr_full = _resolve_rare_threshold(len(X_final), thr_abs=rare_threshold, frac=rare_fraction)
    if thr_full > 0 and len(cat_cols) > 0:
        rare_map_full = _fit_rare_map(X_final, cat_cols, threshold=int(thr_full))
        X_tmp_tr, _ = _apply_rare_map_pair(X_final, X_final, rare_map_full, use_unseen=False)
        X_final = X_tmp_tr

    lgb_train = lgb.Dataset(X_final, (z_full if use_tt else y_full), categorical_feature=cat_cols, params=(ds_binning or None))

    model = lgb.train(params, lgb_train, num_boost_round=int(best_iter), callbacks=[lgb.callback.log_evaluation(period=0)])

    raw_pred = model.predict(X_final, num_iteration=model.best_iteration)

    if task == "regression" and use_tt:
        smear = _compute_smearing_factor(z_full, raw_pred) if tt_info.get("smearing", True) else 1.0
        y_pred_infer = inv(raw_pred, smear=smear)     # production-like predictions
        y_pred_metrics = inv(raw_pred, smear=1.0)     # fair metrics (no smearing)
        tt_info["smear"] = float(smear)
        try:
            setattr(model, "_target_transform_info", tt_info)
        except Exception:
            pass
    else:
        y_pred_infer = raw_pred
        y_pred_metrics = raw_pred

    if task == "regression":
        rmse = float(np.sqrt(np.mean((y_full - y_pred_metrics) ** 2)))
        mae  = float(mean_absolute_error(y_full, y_pred_metrics))
        r2   = float(r2_score(y_full, y_pred_metrics))
        train_metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        train_metrics["_rmse_infer_smearing"] = float(np.sqrt(np.mean((y_full - y_pred_infer) ** 2)))
    else:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        if np.unique(y_full).size > 2:
            ylab = np.argmax(y_pred_metrics, axis=1)
            avg = "macro"
        else:
            ylab = (y_pred_metrics >= 0.5).astype(int)
            avg = "binary"
        train_metrics = {
            "accuracy": float(accuracy_score(y_full, ylab)),
            "precision": float(precision_score(y_full, ylab, average=avg, zero_division=0)),
            "recall": float(recall_score(y_full, ylab, average=avg, zero_division=0)),
            "f1": float(f1_score(y_full, ylab, average=avg, zero_division=0)),
        }

    if save_model or save_params:
        model_dir = os.path.join(pot_raziskovanja, "model")
        os.makedirs(model_dir, exist_ok=True)
        if save_params:
            params_to_save = dict(params)
            params_to_save["num_boost_round"] = int(best_iter)
            _atomic_write_json(os.path.join(model_dir, f"{save_prefix}_params.json"), params_to_save)
        if save_model:
            out_base = os.path.join(model_dir, f"{save_prefix}_model")
            _safe_save_booster(model, out_base, num_iteration=int(best_iter))
            meta = {
                "task": task,
                "features": features,
                "target_col": target_col,
                "cv_nfold": cv_nfold,
                "best_iter": int(best_iter),
                "deterministic": deterministic,
                "_target_transform_info": tt_info,
                "rare_threshold": int(np.ceil(float(rare_threshold))) if rare_threshold and float(rare_threshold) > 1 else None,
                "rare_fraction": float(rare_fraction) if rare_fraction is not None else (
                    float(rare_threshold) if (rare_threshold is not None and 0 < float(rare_threshold) <= 1) else None
                ),
                "id_col": id_col,
            }
            _atomic_write_json(os.path.join(model_dir, f"{save_prefix}_meta.json"), meta)

    return model, y_pred_infer, train_metrics, cv_metrics, cat_cols, cv_results, tt_info



# -----------------------------
# Loader helper
# -----------------------------

def load_trained_model(pot_raziskovanja: str, prefix: str = "GBM"):
    model_dir = os.path.join(pot_raziskovanja, "model")
    pkl_path = os.path.join(model_dir, f"{prefix}_model.pkl")
    txt_path = os.path.join(model_dir, f"{prefix}_model.txt")
    params_path = os.path.join(model_dir, f"{prefix}_params.json")
    meta_path = os.path.join(model_dir, f"{prefix}_meta.json")

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
    elif os.path.exists(txt_path):
        model = lgb.Booster(model_file=txt_path)
    else:
        raise FileNotFoundError(f"Model not found at {pkl_path} or {txt_path}")

    params = None
    meta = None
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    tt_info = {}
    if meta and "_target_transform_info" in meta:
        tt_info = meta["_target_transform_info"]
        try:
            setattr(model, "_target_transform_info", tt_info)
        except Exception:
            pass

    return model, params, meta, tt_info

# -----------------------------
# Convenience helpers
# -----------------------------

def load_best_params(pot_raziskovanja: str, prefix: str = "GBM") -> Dict[str, Any]:
    params_path = os.path.join(pot_raziskovanja, "model", f"{prefix}_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Params file not found: {params_path}")
    with open(params_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_model_and_params(
    model: lgb.Booster,
    params: Optional[Dict[str, Any]],
    pot_raziskovanja: str,
    prefix: str = "GBM",
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    model_dir = os.path.join(pot_raziskovanja, "model")
    os.makedirs(model_dir, exist_ok=True)

    out_base = os.path.join(model_dir, f"{prefix}_model")
    best_iter = getattr(model, "best_iteration", None)
    _safe_save_booster(model, out_base, num_iteration=best_iter)

    if params is not None:
        _atomic_write_json(os.path.join(model_dir, f"{prefix}_params.json"), params)

    if meta is not None:
        _atomic_write_json(os.path.join(model_dir, f"{prefix}_meta.json"), meta)

def predict_with_inverse(
    model: lgb.Booster,
    X: pd.DataFrame,
    tt_info: Optional[Dict[str, Any]] = None,
    num_iteration: Optional[int] = None,
) -> np.ndarray:
    y_hat = model.predict(X, num_iteration=num_iteration or getattr(model, "best_iteration", None))
    info = tt_info if tt_info is not None else getattr(model, "_target_transform_info", None)
    if isinstance(info, dict) and info.get("enabled", False):
        return _inverse_from_info(y_hat, info)
    return y_hat
