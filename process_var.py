import pandas as pd
import numpy as np

def derive_variable(df, new_var, var_type='numeric', length=None, condition=None, rule=None):
    # Build boolean mask from condition
    if condition:
        cond_var, cond_val = condition.split('=')
        filtered = df[cond_var].astype(str) == cond_val
    else:
        filtered = pd.Series([True] * len(df), index=df.index)

    # Always create/overwrite the column fresh
    if var_type == 'numeric':
        df[new_var] = np.nan   # use NaN for numeric
    elif var_type == 'character':
        df[new_var] = None     # use None for string/object
    else:
        raise ValueError("var_type must be 'numeric' or 'character'")

    # --- Numeric derivation ---
    if var_type == 'numeric':
        try:
            df.loc[filtered, new_var] = df.loc[filtered].eval(rule, engine='python')
        except Exception:
            df.loc[filtered, new_var] = pd.eval(
                rule, engine='python',
                local_dict=df.loc[filtered].to_dict(orient='series')
            )

    # --- Character derivation ---
    elif var_type == 'character':
        try:
            df.loc[filtered, new_var] = df.loc[filtered].eval(rule, engine='python').astype(str)
        except Exception:
            df.loc[filtered, new_var] = df.loc[filtered, rule].astype(str)
        if length:
            df.loc[filtered, new_var] = df.loc[filtered, new_var].str[:length]

    return df




def aggregate_variable(df, new_var, group_by_var=None, rule=None, operation='sum', condition=None, percentile_value=0.5, **kwargs):
    """
    Aggregates a variable with optional grouping and filtering.
    Supports sum, mean, count, median, mode, percentiles, and count distinct.
    Always returns the original DataFrame with a new column [new_var].

    Args:
        df (pd.DataFrame): Input DataFrame
        new_var (str): Name of the resulting variable
        group_by_var (str, list, or None): Columns to group by (default=None → full DataFrame)
        rule (str or None): Column to aggregate (required for most operations except count)
        operation (str or callable): Aggregation ('sum','mean','count','median','mode','percentile','count distinct', etc.)
        condition (pd.Series or None): Boolean mask to filter rows before aggregation
        percentile_value (float): For 'percentile' operation (default=0.5)
        **kwargs: Extra args for aggregation function

    Returns:
        pd.DataFrame: Original DataFrame with new column [new_var]
    """

    # Ensure group_by_var is a list
    if group_by_var is None:
        group_by_var = []
    elif isinstance(group_by_var, str):
        group_by_var = [group_by_var]

    # Apply condition only for aggregation
    if condition is not None:
        df_to_agg = df[condition].copy()
    else:
        df_to_agg = df.copy()

    # No grouping → full DataFrame aggregation
    if len(group_by_var) == 0:
        if operation.lower() == 'count':
            df[new_var] = len(df_to_agg)
        elif operation.lower() in ['count distinct', 'nunique']:
            if rule is None:
                raise ValueError("rule must be specified for 'count distinct'.")
            df[new_var] = df_to_agg[rule].nunique(**kwargs)
        elif operation.lower() == 'median':
            df[new_var] = df_to_agg[rule].median(**kwargs)
        elif operation.lower() == 'mode':
            df[new_var] = df_to_agg[rule].mode().iloc[0]
        elif operation.lower() == 'percentile':
            df[new_var] = df_to_agg[rule].quantile(percentile_value, **kwargs)
        else:
            df[new_var] = getattr(df_to_agg[rule], operation)(**kwargs)
        return df

    # Determine if grouping has one or multiple columns
    single_group = len(group_by_var) == 1

    # GROUPED AGGREGATION
    if operation.lower() == 'count':
        col_to_count = rule if rule is not None else df.columns[0]
        agg = df_to_agg.groupby(group_by_var)[col_to_count].count()
        if single_group:
            df[new_var] = df[group_by_var[0]].map(agg)
        else:
            df[new_var] = df[group_by_var].apply(lambda row: agg[tuple(row)], axis=1)

    elif operation.lower() in ['count distinct', 'nunique']:
        if rule is None:
            raise ValueError("rule must be specified for 'count distinct'.")
        agg = df_to_agg.groupby(group_by_var)[rule].nunique()
        if single_group:
            df[new_var] = df[group_by_var[0]].map(agg)
        else:
            df[new_var] = df[group_by_var].apply(lambda row: agg[tuple(row)], axis=1)

    elif operation.lower() == 'median':
        agg = df_to_agg.groupby(group_by_var)[rule].median()
        if single_group:
            df[new_var] = df[group_by_var[0]].map(agg)
        else:
            df[new_var] = df[group_by_var].apply(lambda row: agg[tuple(row)], axis=1)

    elif operation.lower() == 'mode':
        agg = df_to_agg.groupby(group_by_var)[rule].apply(lambda x: x.mode().iloc[0])
        if single_group:
            df[new_var] = df[group_by_var[0]].map(agg)
        else:
            df[new_var] = df[group_by_var].apply(lambda row: agg[tuple(row)], axis=1)

    elif operation.lower() == 'percentile':
        agg = df_to_agg.groupby(group_by_var)[rule].quantile(percentile_value)
        if single_group:
            df[new_var] = df[group_by_var[0]].map(agg)
        else:
            df[new_var] = df[group_by_var].apply(lambda row: agg[tuple(row)], axis=1)

    else:
        agg = df_to_agg.groupby(group_by_var)[rule].agg(operation, **kwargs)
        if single_group:
            df[new_var] = df[group_by_var[0]].map(agg)
        else:
            df[new_var] = df[group_by_var].apply(lambda row: agg[tuple(row)], axis=1)

    return df


def transform_var(df, var_types=None):
    """
    Applies mappings or dtype conversions to variables based on var_types.

    Parameters:
    - df : pd.DataFrame
        The input dataframe.
    - var_types : dict (optional)
        Dictionary in the form:
        {
            'var_name': ('type', dtype_or_mapping)
        }
        where 'type' is one of ['numerical', 'ordinal', 'nominal'] and
        dtype_or_mapping is either None, a dict (for ordinal mappings), or a string dtype.

    Returns:
    - df : pd.DataFrame
        DataFrame with transformed variables.
    - type_map : dict
        Dictionary mapping variable names to their types ('numerical', 'ordinal', 'nominal').
    """
    if var_types is not None:
        for var, (vtype, conv) in var_types.items():
            if var not in df.columns:
                raise ValueError(f"Variable '{var}' not found in DataFrame.")

            if conv is None:
                # Try converting ordinal string variables to numeric
                if vtype == 'ordinal':
                    if df[var].dtype == 'object' or pd.api.types.is_string_dtype(df[var]):
                        try:
                            df[var] = pd.to_numeric(df[var])
                        except ValueError:
                            # Leave as is if conversion fails (treated as nominal)
                            pass
                continue

            elif isinstance(conv, dict):
                # Map ordinal categories to numeric ranks
                df[var] = df[var].map(conv)

            elif isinstance(conv, str):
                # Convert to specified dtype (e.g., 'category')
                df[var] = df[var].astype(conv)

            else:
                raise ValueError(f"Invalid dtype_or_mapping for variable '{var}': {conv}")

    # If var_types not provided, infer types based on dtype
    if var_types is None:
        var_types = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                var_types[col] = ('numerical', None)
            else:
                var_types[col] = ('nominal', None)

    type_map = {var: vtype for var, (vtype, _) in var_types.items()}
    return df, type_map




