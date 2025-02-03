import sys
import os

sys.path.append("/opt/ml/processing/input/")

import requests
import pandas as pd
from typing import List
from enum import Enum
from crypto_rf import utils


class TimeFrame(Enum):
    MINUTE = "minute"
    HOUR = "hour"


def _pull_crypto_data(
    symbol: str, timeframe: TimeFrame, periods: int, currency="USD", exchange="Coinbase"
):
    if timeframe == TimeFrame.HOUR and periods > 24:
        raise ValueError("Max 24 hours of hourly data.")
    if timeframe == TimeFrame.MINUTE and periods > 60:
        raise ValueError("Max 60 minutes of minute data.")

    endpoint = f"https://min-api.cryptocompare.com/data/v2/histo{timeframe.value}"
    params = {"fsym": symbol, "tsym": currency, "limit": periods, "e": exchange}

    response = requests.get(endpoint, params=params)
    data = response.json().get("Data", {}).get("Data", [])

    return (
        pd.DataFrame(data).assign(
            time=lambda df: pd.to_datetime(df["time"], unit="s"), symbol=symbol
        )
        if data
        else None
    )


def pull_crypto_data(
    symbols: List[str],
    timeframe: TimeFrame,
    periods: int,
    currency="USD",
    exchange="Coinbase",
):
    return pd.concat(
        [_pull_crypto_data(s, timeframe, periods, currency, exchange) for s in symbols]
    )


def load_symbols_universe():
    all_univ = utils.list_s3_files(
        bucket_name="quant-finance-data", prefix="raw_crypto/universe/"
    )
    all_univ.sort()
    most_recent_univ = all_univ[-1]
    univ = utils.read_s3_file(most_recent_univ)
    symbols = univ.symbol.drop_duplicates().str.upper().tolist()
    date = most_recent_univ.replace("raw_crypto/universe/universe_", "").replace(
        ".csv", ""
    )
    return symbols, date


def main():
    u, d = load_symbols_universe()
    df = pull_crypto_data(u, TimeFrame.HOUR, periods=24)
    if df is None or df.empty:
        print("No data to export. Skipping S3 upload.")
        return
    utils.write_s3_file_parquet(df, f"raw_crypto/hour/crypto_prices_{d}.parquet")


if __name__ == "__main__":
    main()
