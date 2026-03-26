# missing

import numpy as np
import pandas as pd
from typing import Literal, Optional

def mcar_missing(df, columns, target_missing_fraction, new_status, random_state=None):
    """
    Add MCAR missingness to specified columns, and update STATUS_ZAPISA cord version)
    to new_status only for rows where missing values were introduced.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        columns (list or str): Column or list of columns where missingness should be introduced.
        target_missing_fraction (float): Fraction of rows to introduce missingness.
        new_status (int): New record version set for STATUS_ZAPISA in affected rows.
        random_state (int, optional): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Modified dataframe with missing values and updated status.
    """
    df_copy = df.copy()
    rng = np.random.default_rng(random_state)

    if isinstance(columns, str):
        columns = [columns]

    n = len(df_copy)
    missing_indices_total = set()

    for col in columns:
        missing_count = int(target_missing_fraction * n)
        missing_indices = rng.choice(df_copy.index, size=missing_count, replace=False)
        df_copy.loc[missing_indices, col] = np.nan
        missing_indices_total.update(missing_indices)

    # Update STATUS_ZAPISA only for rows where missing values were introduced
    df_copy.loc[df_copy.index.isin(missing_indices_total), 'STATUS_ZAPISA'] = new_status

    return df_copy

# -------- 1) Record-level MAR (weighted by related_column) 
# Related column is continous variable


def mar_missing(
    df: pd.DataFrame,
    target_column: str,
    related_column: str,
    new_status: int,
    target_missing_fraction: Optional[float] = None,  # e.g., 0.05
    k: Optional[int] = None,                          # exact count overrides fraction
    type: Literal["max", "min"] = "max",
    random_state: Optional[int] = None,
    exclude_existing_nas: bool = True,
    status_column: str = "STATUS_ZAPISA",
    rounding: Literal["floor","ceil","round"] = "floor",
):
    """
    MAR with EXACT-size weighted sampling on rows, using related_column for weights.

    - If type="max": higher related_column → higher chance to be missing
      type="min": lower related_column → higher chance
    - Exact size is computed using the chosen rounding to mirror MCAR.
    """
    if target_column not in df.columns:
        raise KeyError(f"Column '{target_column}' not in DataFrame.")
    if related_column not in df.columns:
        raise KeyError(f"Related column '{related_column}' not in DataFrame.")
    if status_column not in df.columns:
        raise KeyError(f"Status column '{status_column}' not in DataFrame.")
    if target_missing_fraction is None and k is None:
        raise ValueError("Provide either target_missing_fraction or k.")

    rng = np.random.default_rng(random_state)
    out = df.copy()

    # Eligible rows (optionally exclude pre-existing NaNs)
    eligible_mask = out[target_column].notna() if exclude_existing_nas else pd.Series(True, index=out.index)
    idx = out.index[eligible_mask].to_numpy()
    n_eligible = idx.size
    if n_eligible == 0:
        return out

    # Build weights from related_column
    rel = out.loc[eligible_mask, related_column].astype(float).to_numpy()
    rmin, rmax = np.nanmin(rel), np.nanmax(rel)
    if rmax == rmin or not np.isfinite(rmin) or not np.isfinite(rmax):
        weights = np.ones(n_eligible, dtype=float)
    else:
        if type == "max":
            weights = (rel - rmin) / (rmax - rmin)
        elif type == "min":
            weights = (rmax - rel) / (rmax - rmin)
        else:
            raise ValueError("type must be 'max' or 'min'")
        weights = np.clip(weights, 0.0, None)
        if weights.sum() == 0:
            weights = np.ones_like(weights, dtype=float)

    p = weights / weights.sum()

    # Compute exact rounding
    if k is None:
        raw = float(target_missing_fraction) * n_eligible
        if rounding == "floor":
            k = int(raw)
        elif rounding == "ceil":
            from math import ceil
            k = int(ceil(raw))
        elif rounding == "round":
            k = int(round(raw))
        else:
            raise ValueError("rounding must be 'floor', 'ceil', or 'round'")
    k = max(0, min(int(k), n_eligible))
    if k == 0:
        return out

    chosen = rng.choice(idx, size=k, replace=False, p=p)
    out.loc[chosen, target_column] = np.nan
    out.loc[chosen, status_column] = new_status
    return out


# -------- 2) Company-level MAR with exact count (whole companies + optional partial last) --------


def mar_company_missing(
    df: pd.DataFrame,
    company_col: str,
    target_column: str,
    new_status: int,
    target_missing_fraction: Optional[float] = None,    # e.g., 0.10
    k: Optional[int] = None,                            # exact count; if set, overrides fraction
    alpha: float = 1.0,                                 # higher -> stronger preference for small companies
    random_state: Optional[int] = None,
    exclude_existing_nas: bool = True,                  # align denominator with MAR/MCAR
    status_column: str = "STATUS_ZAPISA",
    rounding: Literal["floor","ceil","round"] = "floor"
) -> pd.DataFrame:
    """
    MAR (Missing At Random) by company size (observed): smaller companies → higher missingness.

    Mechanism
    ---------
    - Company size is measured as the number of (eligible) rows per company.
    - We select *whole companies* first, with weights w_i ∝ (1/size_i)^alpha.
    - If the exact target 'k' (computed from 'target_missing_fraction' with chosen rounding)
      cannot be met using whole companies only, the function *automatically*:
        * takes a partial sample from a company to hit k exactly, OR
        * drops a whole company, if that yields a closer total to k.
      The goal is to be as close as possible to k, typically exact.

    -------------------
    Missingness depends on an *observed* variable (company size).

    Parameters
    ----------
    df : DataFrame
        Input data.
    company_col : str
        Column with company identifiers.
    target_column : str
        Column in which to introduce NaNs.
    new_status : int
        Value written into `status_column` where missingness is applied.
    target_missing_fraction : float, optional
        Desired fraction of missing among eligible rows (0..1). Ignored if `k` is provided.
    k : int, optional
        Exact number of rows to set missing (overrides fraction).
    alpha : float
        Strength of the size effect. Typical 0.5-1.5. Larger ⇒ smaller firms favored more.
    random_state : int, optional
        RNG seed for reproducibility (uses NumPy Generator).
    exclude_existing_nas : bool
        If True, only non-NaN rows in `target_column` are eligible (denominator consistency).
    status_column : str
        Status column name to flag modified rows.
    rounding : {"floor","ceil","round"}
        Rounding rule when converting fraction to count (match MAR/MCAR conventions).

    Returns
    -------
    DataFrame
        Modified copy with NaNs inserted and diagnostics in `.attrs`:
          - mechanism, actual_k, actual_fraction, target_fraction, denominator.
    """

    # --- Basic checks and prep -------------------------------------------------
    if company_col not in df.columns:
        raise KeyError(f"Company column '{company_col}' not in DataFrame.")
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not in DataFrame.")

    # Ensure status column exists (keep function self-contained)
    if status_column not in df.columns:
        df = df.copy()
        df[status_column] = np.nan

    if target_missing_fraction is None and k is None:
        raise ValueError("Provide either target_missing_fraction or k.")

    rng = np.random.default_rng(random_state)
    out = df.copy()

    # Eligible rows: optionally exclude rows with existing NaN in target
    eligible_mask = out[target_column].notna() if exclude_existing_nas else pd.Series(True, index=out.index)
    n_elig = int(eligible_mask.sum())
    if n_elig == 0:
        out.attrs.update({
            "mechanism": f"MAR_by_company_size(alpha={alpha})",
            "actual_k": 0,
            "actual_fraction": 0.0,
            "target_fraction": 0.0,
            "denominator": "eligible_non_na" if exclude_existing_nas else "all_rows",
            "note": "No eligible rows."
        })
        return out

    # Compute exact k from fraction using selected rounding (to mirror MAR/MCAR)
    if k is None:
        raw = float(target_missing_fraction) * n_elig
        if rounding == "floor":
            k = int(raw)
        elif rounding == "ceil":
            from math import ceil
            k = int(ceil(raw))
        elif rounding == "round":
            k = int(round(raw))
        else:
            raise ValueError("rounding must be 'floor', 'ceil', or 'round'")
    k = max(0, min(int(k), n_elig))
    if k == 0:
        out.attrs.update({
            "mechanism": f"MAR_by_company_size(alpha={alpha})",
            "actual_k": 0,
            "actual_fraction": 0.0,
            "target_fraction": (target_missing_fraction if target_missing_fraction is not None else 0.0),
            "denominator": "eligible_non_na" if exclude_existing_nas else "all_rows",
        })
        return out

    # --- Company sizes (measured on the eligible subset) -----------------------
    comp_sizes = out.loc[eligible_mask, company_col].value_counts().astype(int)  # size_i
    companies = comp_sizes.index.to_numpy()

    # Weights: smaller company ⇒ higher weight; normalize over companies
    w = (1.0 / comp_sizes.astype(float)) ** float(alpha)
    w = w / w.sum()

    # --- Select whole companies without replacement by weight ------------------
    chosen_companies = []
    cum = 0
    remaining = k
    available = set(companies)

    while remaining > 0 and available:
        # Normalize weights among the still-available companies
        w_av = w[w.index.isin(available)]
        w_av = w_av / w_av.sum()
        comp = rng.choice(w_av.index.to_numpy())
        size = int(comp_sizes.loc[comp])

        chosen_companies.append(comp)
        available.remove(comp)
        cum += size
        remaining -= size

        if cum >= k:
            break

    if not chosen_companies:
        out.attrs.update({
            "mechanism": f"MAR_by_company_size(alpha={alpha})",
            "actual_k": 0,
            "actual_fraction": 0.0,
            "target_fraction": (target_missing_fraction if target_missing_fraction is not None else k / n_elig),
            "denominator": "eligible_non_na" if exclude_existing_nas else "all_rows",
            "note": "No companies selected."
        })
        return out

    # Mask for all fully selected companies
    mask_whole = (out[company_col].isin(chosen_companies)) & eligible_mask

    if cum == k:
        # Exact hit using only whole companies
        out.loc[mask_whole, target_column] = np.nan
        out.loc[mask_whole, status_column] = new_status
        actual_k = k

    else:
        # Overshoot: decide partial vs. dropping the last company (choose the closer to k)
        last_comp = chosen_companies[-1]
        last_size = int(comp_sizes.loc[last_comp])
        cum_without_last = cum - last_size

        need_from_last = k - cum_without_last         # how many rows we need from the last company
        diff_partial   = abs(cum - k)                 # distance if we kept the whole last company
        diff_drop      = abs(cum_without_last - k)    # distance if we drop the last one

        # If dropping already hits k or is closer (and partial would be trivial), drop; else take partial to hit k.
        do_drop = (cum_without_last == k) or (diff_drop < diff_partial and need_from_last in (0, last_size))

        if do_drop:
            # Use only fully selected companies except the last one
            if len(chosen_companies) > 1:
                mask_whole = (out[company_col].isin(chosen_companies[:-1])) & eligible_mask
            else:
                mask_whole = pd.Series(False, index=out.index)

            out.loc[mask_whole, target_column] = np.nan
            out.loc[mask_whole, status_column] = new_status
            actual_k = cum_without_last

            # If still short of k, take a partial from the most suitable company (size closest to remaining)
            remain = k - actual_k
            if remain > 0:
                candidates = [last_comp] + list(available)
                cand_sizes = {c: int(comp_sizes.loc[c]) for c in candidates}
                chosen_partial_comp = min(cand_sizes, key=lambda c: abs(cand_sizes[c] - remain))
                idx_pool = out.index[eligible_mask & (out[company_col] == chosen_partial_comp)].to_numpy()
                select_idx = rng.choice(idx_pool, size=min(remain, len(idx_pool)), replace=False)
                out.loc[select_idx, target_column] = np.nan
                out.loc[select_idx, status_column] = new_status
                actual_k += len(select_idx)
        else:
            # Take exactly the needed number from the last selected company
            if len(chosen_companies) > 1:
                mask_full_except_last = (out[company_col].isin(chosen_companies[:-1])) & eligible_mask
                out.loc[mask_full_except_last, target_column] = np.nan
                out.loc[mask_full_except_last, status_column] = new_status
                actual_k = cum_without_last
            else:
                actual_k = 0

            idx_last = out.index[eligible_mask & (out[company_col] == last_comp)].to_numpy()
            select_idx = rng.choice(idx_last, size=need_from_last, replace=False)
            out.loc[select_idx, target_column] = np.nan
            out.loc[select_idx, status_column] = new_status
            actual_k += need_from_last

    # --- Diagnostics -----------------------------------------------------------
    denom = n_elig if exclude_existing_nas else len(out)
    out.attrs["mechanism"] = f"MAR_by_company_size(alpha={alpha})"
    out.attrs["target_fraction"] = (target_missing_fraction if target_missing_fraction is not None else k / denom)
    out.attrs["actual_fraction"] = actual_k / denom if denom > 0 else None
    out.attrs["actual_k"] = int(actual_k)
    out.attrs["denominator"] = "eligible_non_na" if exclude_existing_nas else "all_rows"

    return out

#mnar


def mnar_missing(
    df: pd.DataFrame,
    col: str,
    target_missing_fraction: float,
    new_status: int,
    top_frac: float = 0.01,      # upper tail proportion that receives fixed weight 1
    beta: float = 1.0,           # MNAR strength
    type: Literal["max", "min"] = "max",  # "max"→ higher values more missing, "min"→ lower values more missing
    random_state: Optional[int] = None,
    status: str = "STATUS_ZAPISA",
) -> pd.DataFrame:
    """
    Introduce MNAR missingness into a single column using a logistic mechanism.

    - Missingness probability depends on the value in `col` (MNAR).
    - `type="max"` → higher values are more likely to become missing.
    - `type="min"` → lower values are more likely to become missing.
    - `top_frac` always refers to the upper tail; those values get weight = 1.
    - Total number of new missings is computed with the same rounding rule as MCAR:
        k = int(target_missing_fraction * n)   (floor for positive values).
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    x = out[col].astype(float).values
    n = len(x)

    # -------------------------------------------------
    # 1) Standardization (for logistic transformation)
    # -------------------------------------------------
    mu = x.mean()
    sd = x.std()

    if sd == 0 or np.isnan(sd):
        # If the variable has no variation → MNAR degenerates → revert to MCAR-like weights
        weights = np.ones(n, dtype=float)

    else:
        z = (x - mu) / sd  # standardized values

        # -------------------------------------------------
        # 2) MNAR direction:
        # "max" → higher values more likely missing
        # "min" → lower values more likely missing
        # -------------------------------------------------
        if type == "max":
            logits = beta * z            # positive slope for upper tail
        elif type == "min":
            logits = -beta * z           # flipped effect → lower values get higher probability
        else:
            raise ValueError("type must be 'max' or 'min'")

        base = 1.0 / (1.0 + np.exp(-logits)) + 1e-12   # logistic > 0 probability for all observations

        # -------------------------------------------------
        # 3) Top fraction ALWAYS refers to the upper tail
        #    Values in this upper segment receive weight = 1
        # -------------------------------------------------
        if 0 < top_frac < 1:
            cutoff = np.quantile(x, 1 - top_frac)
            extreme_mask = x >= cutoff      # upper tail only
            weights = base.copy()
            weights[extreme_mask] = 1.0     # fixed weight for extreme high values
        else:
            weights = base.copy()

        # -------------------------------------------------
        # 4) Scale to population level:
        #    mean weight = 1 → sum(weights) = N
        #    ensures a stable probability distribution
        # -------------------------------------------------
        weights *= (n / weights.sum())

    # -------------------------------------------------
    # 5) Convert weights to sampling probabilities
    #    k uses the SAME rounding rule as in MCAR:
    #    k = int(target_missing_fraction * n)
    # -------------------------------------------------
    p = weights / weights.sum()

    k = int(target_missing_fraction * n)    # floor-like for positive values (consistent with MCAR)
    k = max(0, min(k, n))

    if k > 0:
        chosen = rng.choice(np.arange(n), size=k, replace=False, p=p)
        out.loc[chosen, col] = np.nan
        out.loc[chosen, status] = new_status

    return out
