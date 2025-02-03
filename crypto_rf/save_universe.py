import sys
import os

sys.path.append("/opt/ml/processing/input/")

import requests
import pandas as pd
from crypto_rf import utils


def top_crypto_market_cap(vs_currency="usd", limit=250):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    headers = {"accept": "application/json"}
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    df = pd.DataFrame(response.json())
    return df[["id", "symbol", "name", "market_cap", "current_price", "total_volume"]]


def main():
    df = top_crypto_market_cap()
    utc_current_day = pd.Timestamp.now(tz="EST").strftime("%Y_%m_%d")
    utils.write_s3_file(
        df, file_key=f"raw_crypto/universe/universe_{utc_current_day}.csv"
    )


if __name__ == "__main__":
    main()
