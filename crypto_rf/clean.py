import pandas as pd
import os
from tqdm import tqdm 

files = os.listdir("../data/raw/")
df_list = list()
for f in tqdm(files): 
    asset= f.split("_")[1]
    asset = asset[:-3]
    tmp = pd.read_csv(f"../data/raw/{f}", skiprows=1)
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp = tmp.rename(columns={"Volume USD": "volume_usd", f"Volume {asset}": "volume_asset"})
    tmp = tmp.set_index(["date", "symbol"]).drop(columns=["unix"])[["open", "high", "low", "close", "volume_usd", "volume_asset"]]
    df_list.append(tmp)
df = pd.concat(df_list)
df = df.loc[df.index.get_level_values('date') >= '2022-06-01']
df.to_csv("../data/clean/prices.csv")