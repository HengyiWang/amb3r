import pandas as pd


def prepend_level(df, level_name, level, axis=0):
    return pd.concat({level: df}, axis=axis, names=[level_name])
