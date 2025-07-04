def derive_variable(df, new_var, var_type='numeric', length=None, condition=None, rule=None):
    if condition:
        cond_var, cond_val = condition.split('=')
        filtered = df[cond_var].astype(str) == cond_val
    else:
        filtered = pd.Series([True] * len(df))

    if var_type == 'numeric':
        # Directly use eval only on numeric rules
        try:
            df.loc[filtered, new_var] = df.loc[filtered].eval(rule)
        except Exception:
            # If eval fails, fall back to plain Python eval
            df.loc[filtered, new_var] = eval(rule, {}, df.loc[filtered].to_dict(orient='series'))

    elif var_type == 'character':
        try:
            df.loc[filtered, new_var] = df.loc[filtered].eval(rule).astype(str)
        except Exception:
            df.loc[filtered, new_var] = df.loc[filtered, rule].astype(str)
        if length:
            df.loc[filtered, new_var] = df.loc[filtered, new_var].str[:length]
    else:
        raise ValueError("var_type must be 'numeric' or 'character'")

    return df

