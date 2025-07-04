# makes missing valuess

import numpy as np
import pandas as pd

def mcar_missing(df, columns, missing_fraction, new_status, random_state=None):
    """
    Add MCAR missingness to specified columns, and update STATUS_ZAPISA to new_status only for rows
    where missing values were introduced.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        columns (list or str): Column or list of columns where missingness should be introduced.
        missing_fraction (float): Fraction of rows to introduce missingness.
        new_status (int): New record version set for STATUS_ZAPISA in affected rows.
        random_state (int, optional): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Modified dataframe with missing values and updated status.
    """
    df_copy = df.copy()
    np.random.seed(random_state)

    if isinstance(columns, str):
        columns = [columns]

    n = len(df_copy)
    missing_indices_total = set()

    for col in columns:
        missing_count = int(missing_fraction * n)
        missing_indices = np.random.choice(df_copy.index, size=missing_count, replace=False)
        df_copy.loc[missing_indices, col] = np.nan
        missing_indices_total.update(missing_indices)

    # Update STATUS_ZAPISA only for rows where missing values were introduced
    df_copy.loc[df_copy.index.isin(missing_indices_total), 'STATUS_ZAPISA'] = new_status

    return df_copy
