import pandas as pd
import os
import config


def clean_data(df):
    df = df.loc[lambda x: x.index < df[df[df.columns[1]].isnull()].iloc[0].name].copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df.astype(float)


def process_files(input_dir, output_dir, column_dict):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f in column_dict.keys():
        # Read the file into a pandas DataFrame
        df = pd.read_csv(os.path.join(input_dir, f), low_memory=False)

        # Apply the cleaning function
        cleaned_df = clean_data(df)
        cleaned_df = cleaned_df[column_dict.get(f)[1]]

        # Write the cleaned DataFrame to the output directory
        cleaned_df.to_csv(os.path.join(output_dir, column_dict.get(f)[0] + ".csv"))

        print(f"Processed and saved {f} to {column_dict.get(f)[0]}.csv")


if __name__ == "__main__":
    process_files(config.INPUT_DIR, config.OUTPUT_DIR, config.COLUMN_DICT)
