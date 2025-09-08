import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    brier_score_loss, matthews_corrcoef, cohen_kappa_score,
    mean_absolute_error, r2_score, explained_variance_score, mean_squared_log_error
)
from sklearn.calibration import calibration_curve
from sklearn.utils import check_random_state
from scipy.stats import probplot
from sklearn.model_selection import learning_curve
import shap
import lightgbm as lgb

from sklearn.model_selection import train_test_split
    

def train_lightgbm(
    df_train, 
    target_col, 
    features=None, 
    force_cat_cols=None,
    task='regression', 
    params=None, 
    random_state=42, 
    deterministic=False,
    verbose=100
):
    """
    Train a LightGBM model with automatic categorical handling and optional feature selection,
    using cross-validation to find optimal number of boosting rounds.
    
    Returns:
        model (lightgbm.Booster): Trained LightGBM model.
        y_pred (np.ndarray): Predictions on training data.
        train_metrics (dict): Training performance metrics.
        cv_metrics (dict): CV performance at best iteration.
        cat_cols (list[str]): Categorical columns used.
        cv_results (dict): Full CV history for plotting learning curves.
    """
    import numpy as np
    import lightgbm as lgb
    from sklearn.metrics import (
        root_mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score
    )

    if force_cat_cols is None:
        force_cat_cols = []

    # --- select features ---
    if features is None:
        features = [c for c in df_train.columns if c != target_col]

    missing = [f for f in features if f not in df_train.columns]
    if missing:
        raise ValueError(f"Features not found in dataframe: {missing}")

    X_train = df_train[features].copy()
    y_train = df_train[target_col]

    # --- find categorical columns ---
    cat_cols = list(X_train.select_dtypes(include=['object', 'category']).columns)
    for col in force_cat_cols:
        if col not in features:
            raise ValueError(f"Forced categorical column '{col}' not in features list.")
        if col not in cat_cols:
            cat_cols.append(col)

    # --- handle rare categories ---
    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        counts = X_train[col].value_counts()
        rare_categories = counts[counts < 5].index  # threshold = 5
        new_categories = [c for c in X_train[col].cat.categories if c not in rare_categories] + ["Other"]

        # Add 'Other' to categories if needed
        if "Other" not in X_train[col].cat.categories:
            X_train[col] = X_train[col].cat.add_categories(["Other"])

        # rare categories (keep category dtype)
        mask = X_train[col].isin(rare_categories)
        X_train[col] = X_train[col].where(~mask, "Other")
        X_train[col] = X_train[col].cat.remove_unused_categories()
        X_train[col] = X_train[col].cat.set_categories(new_categories, ordered=True)
        # <<< END FIX

    # --- default params ---
    if params is None:
        params = {}

    if task == 'regression':
        params.setdefault('objective', 'regression')
        params.setdefault('metric', 'rmse')
    elif task == 'classification':
        num_classes = y_train.nunique()
        if num_classes > 2:
            params.setdefault('objective', 'multiclass')
            params.setdefault('metric', 'multi_logloss')
            params['num_class'] = num_classes
        else:
            params.setdefault('objective', 'binary')
            params.setdefault('metric', 'binary_logloss')
    else:
        raise ValueError("Task must be 'regression' or 'classification'.")

    # --- deterministic option ---
    if deterministic:
        params.update({
            'feature_fraction_seed': random_state,
            'bagging_seed': random_state,
            'deterministic': True,
            'num_threads': 1
        })
    else:
        params.setdefault('num_threads', -1)

    # --- prepare dataset ---
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)

    # --- CV with early stopping ---
    max_boost_round = params.get('num_boost_round', 1000) or 1000
    callbacks = []
    if verbose is not None:
        callbacks.append(lgb.callback.log_evaluation(period=verbose))
    callbacks.append(lgb.callback.early_stopping(stopping_rounds=50))

    cv_results = lgb.cv(
        params,
        lgb_train,
        num_boost_round=max_boost_round,
        nfold=5,
        stratified=False if task == 'regression' else True,
        seed=random_state,
        callbacks=callbacks
    )

    # --- best iteration from CV ---
    best_num_boost_round = len(next(iter(cv_results.values())))

    # --- train final model ---
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=best_num_boost_round,
        callbacks=[]
    )

    # --- predictions on training data ---
    y_pred = model.predict(X_train, num_iteration=model.best_iteration)

    # --- train metrics ---
    if task == 'regression':
        rmse = root_mean_squared_error(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        train_metrics = {'rmse': rmse, 'mae': mae, 'r2': r2}
    else:
        if y_train.nunique() > 2:
            y_pred_label = np.argmax(y_pred, axis=1)
            average_type = "macro"
        else:
            y_pred_label = (y_pred > 0.5).astype(int)
            average_type = "binary"
        accuracy = accuracy_score(y_train, y_pred_label)
        precision = precision_score(y_train, y_pred_label, average=average_type, zero_division=0)
        recall = recall_score(y_train, y_pred_label, average=average_type, zero_division=0)
        f1 = f1_score(y_train, y_pred_label, average=average_type, zero_division=0)
        train_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    # --- extract CV metrics at best iteration ---
    cv_metrics = {}
    idx = best_num_boost_round - 1
    for k in list(cv_results.keys()):
        if k.endswith("-mean"):
            mean_val = cv_results[k][idx]
            std_key = k.replace("-mean", "-std")
            std_val = cv_results.get(std_key, [None]*best_num_boost_round)[idx]
            cv_metrics[k] = mean_val
            cv_metrics[std_key] = std_val

    return model, y_pred, train_metrics, cv_metrics, cat_cols, cv_results

def evaluate_lightgbm(
    model,
    df_train,
    target_col,
    features,
    task="regression",
    # NEW: lightweight vs full metrics
    fast=False,
    # existing toggles (safe defaults)
    compute_curves=True,           # ROC/PR/Calibration/QQ
    compute_confusion=True,        # Confusion matrix (classification)
    compute_shap=False,            # SHAP disabled by default (heavy)
    fast_shap=True,                # Use LightGBM pred_contrib when possible
    compute_permutation=False,     # Permutation importance disabled by default (heavy)
    k_perm_features=30,            # Only top-k features by gain for permutation
    n_perm_repeats=3,              # Fewer repeats by default
    # subsampling
    sample_size=5000,              # Used for SHAP and (optionally) curves
    subsample_curves_if_n_gt=200_000,
    random_state=42,
    plot=True,
    cv_results=None
):
    """
    Evaluate a trained LightGBM model with optional heavy components gated by flags.
    Metrics are computed via helper functions with a 'fast' switch.
    Returns a dict of metrics and artifacts.
    """
   
    # --- prepare data ---
    X = df_train[features]
    y = df_train[target_col].values
    y_pred = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    results = {}

    # ==========================
    #   METRICS (now via helpers with fast flag)
    # ==========================
    if task == "regression":
        # Use lightweight or full regression metrics
        reg_metrics = _reg_extra_metrics(y, y_pred, fast=fast)
        results.update(reg_metrics)

        # Plots (unchanged, optionally subsampled)
        if plot and compute_curves:
            Xy = np.c_[y, y_pred]
            if len(Xy) > subsample_curves_if_n_gt:
                from sklearn.utils import check_random_state
                rng = check_random_state(random_state)
                idx = rng.choice(len(Xy), size=subsample_curves_if_n_gt, replace=False)
                y_plot, y_pred_plot = Xy[idx, 0], Xy[idx, 1]
            else:
                y_plot, y_pred_plot = y, y_pred

            # Predicted vs Actual
            plt.figure(); plt.scatter(y_plot, y_pred_plot, alpha=0.5)
            lims = [min(y_plot.min(), y_pred_plot.min()), max(y_plot.max(), y_pred_plot.max())]
            plt.plot(lims, lims, "--")
            plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Predicted vs Actual")
            plt.show()

            # Residuals vs Fitted
            resid = y_plot - y_pred_plot
            plt.figure(); plt.scatter(y_pred_plot, resid, alpha=0.5)
            plt.axhline(0, ls="--")
            plt.xlabel("Fitted"); plt.ylabel("Residuals")
            plt.title("Residuals vs Fitted"); plt.show()

            # QQ-plot of residuals
            plt.figure(); probplot(resid, dist="norm", plot=plt)
            plt.title("Residuals QQ-plot"); plt.show()

    else:
        # Classification: compute probabilities/labels + metrics via helper
        y_proba, y_label, cls_metrics = _classification_core_metrics(y, y_pred, fast=fast)
        results.update(cls_metrics)

        # Plots (optionally subsampled)
        if plot and compute_curves:
            if y_proba.ndim == 1:  # binary
                if len(y) > subsample_curves_if_n_gt:
                    from sklearn.utils import check_random_state
                    rng = check_random_state(random_state)
                    idx = rng.choice(len(y), size=subsample_curves_if_n_gt, replace=False)
                    y_plot = y[idx]; p_plot = y_proba[idx]
                else:
                    y_plot = y; p_plot = y_proba

                if compute_confusion:
                    cm = confusion_matrix(y_plot, (p_plot >= 0.5).astype(int))
                    ConfusionMatrixDisplay(cm).plot(values_format="d")
                    plt.title("Confusion Matrix"); plt.show()

                fpr, tpr, _ = roc_curve(y_plot, p_plot)
                plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
                try:
                    auc_val = roc_auc_score(y_plot, p_plot)
                except Exception:
                    auc_val = np.nan
                plt.xlabel("FPR"); plt.ylabel("TPR")
                plt.title(f"ROC (AUC={auc_val:.3f})"); plt.show()

                prec_c, rec_c, _ = precision_recall_curve(y_plot, p_plot)
                plt.figure(); plt.plot(rec_c, prec_c)
                ap_val = average_precision_score(y_plot, p_plot)
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.title(f"Precision–Recall (AP={ap_val:.3f})"); plt.show()

                prob_true, prob_pred = calibration_curve(y_plot, p_plot, n_bins=15, strategy="quantile")
                plt.figure(); plt.plot(prob_pred, prob_true, marker="o"); plt.plot([0, 1], [0, 1], "--")
                brier_val = brier_score_loss(y_plot, p_plot)
                plt.xlabel("Predicted probability"); plt.ylabel("Empirical frequency")
                plt.title(f"Calibration (Brier={brier_val:.4f})"); plt.show()

            else:  # multiclass
                from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
                if compute_confusion and (y_proba.shape[1] <= 20):
                    cm = confusion_matrix(y, y_label)
                    ConfusionMatrixDisplay(cm).plot(values_format="d")
                    plt.title("Confusion Matrix"); plt.show()
                print("\nClassification Report:"); print(classification_report(y, y_label, zero_division=0))

    # ==========================
    #   FEATURE IMPORTANCE (cheap)
    # ==========================
    try:
        gain = model.feature_importance(importance_type="gain")
        split = model.feature_importance(importance_type="split")
        fi = pd.DataFrame({"feature": model.feature_name(), "gain": gain, "split": split})
        fi["gain_norm"] = fi["gain"] / (fi["gain"].sum() + 1e-12)
        fi = fi.sort_values("gain", ascending=False).reset_index(drop=True)
        results["feature_importance_gain_split"] = fi
    except Exception as e:
        results["feature_importance_error"] = str(e)
        fi = None

    # ==========================
    #   PERMUTATION IMPORTANCE (optional/heavy)
    # ==========================
    if compute_permutation:
        # choose top-k features by gain to cut cost; fallback to first k features
        if fi is not None and "feature" in fi.columns:
            top_cols = fi["feature"].head(k_perm_features).tolist()
        else:
            top_cols = features[:k_perm_features]
        # subsample rows to cut cost if very large
        X_perm = X; y_perm = y
        if len(X_perm) > subsample_curves_if_n_gt:
            from sklearn.utils import check_random_state
            rng = check_random_state(random_state)
            idx = rng.choice(len(X_perm), size=subsample_curves_if_n_gt, replace=False)
            X_perm = X_perm.iloc[idx]
            y_perm = y_perm[idx]
        pi_df, base_score = _permutation_importance_light(
            model, X_perm, y_perm, task,
            metric="auto", n_repeats=n_perm_repeats,
            random_state=random_state, cols_subset=top_cols
        )
        results["permutation_importance"] = pi_df
        results["permutation_base_score"] = base_score

    # ==========================
    #   SHAP (optional/heavy)
    # ==========================
    if compute_shap:
        try:
            from sklearn.utils import check_random_state
            if len(X) > sample_size:
                rng = check_random_state(random_state)
                idx = rng.choice(len(X), size=sample_size, replace=False)
                X_shap = X.iloc[idx]
            else:
                X_shap = X

            if fast_shap:
                contrib = model.predict(X_shap, pred_contrib=True, num_iteration=getattr(model, "best_iteration", None))
                contrib = np.asarray(contrib)
                shap_values = contrib[:, :-1]  # drop bias term
                shap_abs_mean = np.abs(shap_values).mean(axis=0)
                top_idx = int(np.argmax(shap_abs_mean))
                top_feature = features[top_idx]
                results["shap_top_feature"] = top_feature

                if plot:
                    order = np.argsort(shap_abs_mean)[::-1][:20]
                    plt.figure()
                    plt.bar(np.array(features)[order], shap_abs_mean[order])
                    plt.xticks(rotation=90)
                    plt.title("Mean |SHAP| (fast pred_contrib) — top 20")
                    plt.tight_layout()
                    plt.show()
            else:
                explainer = shap.TreeExplainer(model)
                sv = explainer(X_shap)
                if plot:
                    shap.summary_plot(sv, X_shap, show=plot)
                top_feature = features[int(np.argmax(np.abs(sv.values).mean(0)))]
                results["shap_top_feature"] = top_feature
        except Exception as e:
            results["shap_error"] = str(e)

    # ==========================
    #   CV LEARNING CURVE (optional)
    # ==========================
    if plot and isinstance(cv_results, dict) and len(cv_results) > 0 and compute_curves:
        mean_keys = [k for k in cv_results if k.endswith("-mean")]
        if mean_keys:
            k = mean_keys[0]
            means = np.array(cv_results[k])
            std_key = k.replace("-mean", "-std")
            stds = np.array(cv_results.get(std_key, [0] * len(means)))
            iters = np.arange(1, len(means) + 1)
            plt.figure(); plt.plot(iters, means)
            plt.fill_between(iters, means - stds, means + stds, alpha=0.2)
            plt.xlabel("Boosting rounds"); plt.ylabel(k)
            plt.title("CV learning curve"); plt.show()

    return results

def _reg_extra_metrics(y, y_pred, eps=1e-12, fast=False):
    """
    Extra regression metrics.
    If fast=True: only return RMSE, MAE, R2.
    """
    resid = y - y_pred
    rmse = np.sqrt(np.mean(resid ** 2))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    if fast:
        # Lightweight mode: only key metrics
        return {"rmse": rmse, "mae": mae, "r2": r2}

    # --- Full mode ---
    medae = np.median(np.abs(resid))
    evs = explained_variance_score(y, y_pred)

    # MAPE / sMAPE / RMSLE (safe near 0 values)
    y_safe = np.where(np.abs(y) < eps, eps, y)
    mape = np.mean(np.abs((y - y_pred) / y_safe)) * 100.0
    smape = 100.0 * np.mean(2.0 * np.abs(y_pred - y) / (np.abs(y) + np.abs(y_pred) + eps))

    # RMSLE requires nonnegative values
    if np.all(y >= 0) and np.all(y_pred >= 0):
        rmsle = np.sqrt(mean_squared_log_error(y, y_pred + eps))
    else:
        rmsle = np.nan

    # Normalized errors
    rng = np.max(y) - np.min(y)
    nrmse_range = rmse / rng if rng > 0 else np.nan
    nmae_range = mae / rng if rng > 0 else np.nan
    mean_abs_y = np.mean(np.abs(y)) + eps
    nrmse_mean = rmse / mean_abs_y
    nmae_mean = mae / mean_abs_y

    return {
        "rmse": rmse, "mae": mae, "medae": medae, "r2": r2, "evs": evs,
        "mape_%": mape, "smape_%": smape, "rmsle": rmsle,
        "nrmse_range": nrmse_range, "nmae_range": nmae_range,
        "nrmse_mean": nrmse_mean, "nmae_mean": nmae_mean,
        "residuals": resid
    }


def _classification_core_metrics(y, y_pred_raw, fast=False):
    """
    Core classification metrics.
    If fast=True: skip MCC, Cohen's kappa, multiclass ROC AUC.
    Returns: y_proba, y_pred_label, metrics dict.
    """
    n_classes = len(np.unique(y))
    if n_classes == 2:
        # --- Binary classification ---
        # y_pred_raw is the probability for the positive class (or logits, assume probs)
        y_proba = y_pred_raw.ravel()
        y_label = (y_proba >= 0.5).astype(int)

        # Basic metrics
        acc = accuracy_score(y, y_label)
        prec = precision_score(y, y_label, zero_division=0)
        rec = recall_score(y, y_label, zero_division=0)
        f1 = f1_score(y, y_label, zero_division=0)

        # Probabilistic metrics
        try:
            auc = roc_auc_score(y, y_proba)
        except Exception:
            auc = np.nan
        ap = average_precision_score(y, y_proba)
        ll = log_loss(y, np.vstack([1 - y_proba, y_proba]).T, labels=[0, 1])
        brier = brier_score_loss(y, y_proba)

        metrics = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": auc, "pr_auc": ap, "log_loss": ll, "brier": brier,
        }

        if not fast:
            # Add extra metrics in full mode
            mcc = matthews_corrcoef(y, y_label)
            kappa = cohen_kappa_score(y, y_label)
            metrics.update({"mcc": mcc, "cohen_kappa": kappa})

        return y_proba, y_label, metrics

    else:
        # --- Multiclass classification ---
        # y_pred_raw has shape (n, C) with probabilities
        y_proba = y_pred_raw
        y_label = np.argmax(y_proba, axis=1)

        # Basic metrics
        acc = accuracy_score(y, y_label)
        prec = precision_score(y, y_label, average="macro", zero_division=0)
        rec = recall_score(y, y_label, average="macro", zero_division=0)
        f1 = f1_score(y, y_label, average="macro", zero_division=0)
        ll = log_loss(y, y_proba)

        metrics = {
            "accuracy": acc, "precision_macro": prec, "recall_macro": rec,
            "f1_macro": f1, "log_loss": ll
        }

        if not fast:
            # Multiclass ROC AUC is expensive, only compute in full mode
            try:
                auc = roc_auc_score(y, y_proba, multi_class="ovr")
            except Exception:
                auc = np.nan
            metrics["roc_auc_ovr"] = auc

        return y_proba, y_label, metrics


