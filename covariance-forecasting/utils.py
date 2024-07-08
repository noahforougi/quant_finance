from io import StringIO

import boto3
import config
import numpy as np
import pandas as pd
from rpy2.robjects import NULL


def calc_msfe(realized, forecast):
    if forecast is NULL:
        return np.nan  # Or return a specific value to indicate non-convergence

    forecast_array = np.array(forecast)
    if forecast_array.dtype.kind in "iufc":  # Check for numeric types
        if np.isnan(forecast_array).all():
            return np.nan  # Or return a specific value to indicate non-convergence
    else:
        return np.nan

    realized_array = np.array(realized, dtype=float)
    sfe = (forecast_array - realized_array) ** 2
    return np.triu(sfe).sum()


def load_data(filepath):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def read_s3_file(file_key, bucket_name=config.BUCKET_NAME):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    return pd.read_csv(obj["Body"])


def write_s3_file(df, file_key, bucket_name=config.BUCKET_NAME):
    s3 = boto3.client("s3")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())


def list_s3_files(bucket_name, prefix):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    file_keys = [content["Key"] for content in response.get("Contents", [])]
    return file_keys
