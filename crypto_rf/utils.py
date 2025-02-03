from io import StringIO, BytesIO
from crypto_rf import configs as config
import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
import joblib


def read_joblib_from_s3(s3_key, bucket_name=config.BUCKET_NAME):
    # Initialize S3 client
    s3 = boto3.client("s3")

    # Create a buffer to hold the file data
    joblib_buffer = BytesIO()

    # Download the joblib file into the buffer
    s3.download_fileobj(bucket_name, s3_key, joblib_buffer)

    # Seek to the beginning of the buffer
    joblib_buffer.seek(0)

    # Load the object from the buffer using joblib
    obj = joblib.load(joblib_buffer)

    return obj


def save_model_to_s3(model, s3_path, bucket_name=config.BUCKET_NAME):
    # Construct full S3 URI
    full_s3_path = f"s3://{bucket_name}/{s3_path}"

    # Save model to S3
    s3 = s3fs.S3FileSystem()
    with s3.open(full_s3_path, "wb") as f:
        joblib.dump(model, f)
    print(f"Model saved to {full_s3_path}")


def read_s3_file(file_key, bucket_name=config.BUCKET_NAME, skiprows=0):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    return pd.read_csv(obj["Body"], skiprows=skiprows)


def write_s3_file(df, file_key, bucket_name=config.BUCKET_NAME):
    """
    Uploads a dataframe to S3.

    Args:
        df (pd.DataFrame): The dataframe to upload.
        file_key (str): The key (path) for the file in S3.
        bucket_name (str): The name of the S3 bucket.
    """
    s3 = boto3.client("s3")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())
    print("Uploaded entire dataframe to S3.")
    print(df.shape)


def write_s3_file_parquet(
    df: pd.DataFrame, file_key: str, bucket: str = config.BUCKET_NAME
):
    s3 = boto3.client("s3")
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, engine="pyarrow", index=False)
    s3.put_object(Bucket=bucket, Key=file_key, Body=parquet_buffer.getvalue())
    print("Uploaded entire dataframe to S3.")
    print(df.shape)


def multipart_upload_df_to_s3(
    df, file_key, bucket_name=config.BUCKET_NAME, chunk_size=100000
):
    """
    Efficiently uploads a large dataframe to S3 as a single file using multipart upload.

    Args:
        df (pd.DataFrame): The dataframe to upload.
        file_key (str): The S3 file key.
        bucket_name (str): S3 bucket name.
        chunk_size (int): Number of rows per chunk.
    """
    s3 = boto3.client("s3")
    multipart_upload = s3.create_multipart_upload(Bucket=bucket_name, Key=file_key)
    part_info = {"Parts": []}

    try:
        part_number = 1
        for chunk_start in range(0, df.shape[0], chunk_size):
            chunk = df.iloc[chunk_start : chunk_start + chunk_size]
            csv_buffer = BytesIO()
            chunk.to_csv(csv_buffer, index=False, header=chunk_start == 0)
            csv_buffer.seek(0)

            # Upload the part
            response = s3.upload_part(
                Bucket=bucket_name,
                Key=file_key,
                PartNumber=part_number,
                UploadId=multipart_upload["UploadId"],
                Body=csv_buffer,
            )
            part_info["Parts"].append(
                {"PartNumber": part_number, "ETag": response["ETag"]}
            )
            part_number += 1
            print(f"Uploaded chunk {part_number - 1}.")

        # Complete the upload
        s3.complete_multipart_upload(
            Bucket=bucket_name,
            Key=file_key,
            UploadId=multipart_upload["UploadId"],
            MultipartUpload=part_info,
        )
        print("Multipart upload completed successfully.")

    except Exception as e:
        # Abort the upload if something goes wrong
        s3.abort_multipart_upload(
            Bucket=bucket_name, Key=file_key, UploadId=multipart_upload["UploadId"]
        )
        print("Multipart upload aborted.")
        raise e


def list_s3_files(prefix, bucket_name=config.BUCKET_NAME):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    file_keys = [content["Key"] for content in response.get("Contents", [])]
    file_keys = [f for f in file_keys if f != prefix]
    return file_keys
