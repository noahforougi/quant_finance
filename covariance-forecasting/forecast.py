import config
import covariance
import pandas as pd
import utils

file_keys = utils.list_s3_files(config.BUCKET_NAME, "clean/")

for file_key in file_keys:
    df = utils.read_s3_file(file_key).set_index("date")
    df.index = pd.to_datetime(df.index)
    msfe_list = covariance.forecast_cov_matrices(df, config.DATES, config.L)
    msfe_df = pd.concat(msfe_list)
    print(f"Mean MSFE for {file_key}: {msfe_df.mean()}")
    original_filename = file_key.split("/")[-1]
    new_filename = original_filename.replace(".csv", "_forecast_error.csv")
    output_file_key = f"output/{new_filename}"
    utils.write_s3_file(msfe_df, output_file_key)
