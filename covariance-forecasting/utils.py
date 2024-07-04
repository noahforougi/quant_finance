import pandas as pd
import numpy as np


def calc_msfe(realized, forecast):
    sfe = (forecast - realized) ** 2
    return np.triu(sfe).sum()


def load_data(filepath):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df
