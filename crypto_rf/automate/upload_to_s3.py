import boto3
import os
import zipfile


def package_and_upload(local_folder, bucket_name, s3_key):
    zip_filename = "/tmp/crypto_rf.zip"

    # Create a zip archive of the entire folder
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_folder)
                zipf.write(local_path, relative_path)

    print(f"Packaged {local_folder} into {zip_filename}")

    # Upload to S3
    s3_client = boto3.client("s3")
    s3_client.upload_file(zip_filename, bucket_name, s3_key)
    print(f"Uploaded {zip_filename} to s3://{bucket_name}/{s3_key}")


if __name__ == "__main__":
    package_and_upload(
        local_folder="./crypto_rf",  # Run this from quant_finance directory
        bucket_name="quant-finance-data",
        s3_key="code/crypto_rf.zip",
    )
