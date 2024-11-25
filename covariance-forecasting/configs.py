import pandas as pd

INPUT_DIR = "/Users/noahforougi/research/time_series/data/replication_project/raw"
OUTPUT_DIR = "/Users/noahforougi/research/time_series/data/replication_project/clean/"
COLUMN_DICT = {
    # 10 Portfolios formed on Size
    "Portfolios_Formed_on_ME_daily.csv": (
        "size",
        [
            "lo_10",
            "dec_2",
            "dec_3",
            "dec_4",
            "dec_5",
            "dec_6",
            "dec_7",
            "dec_8",
            "dec_9",
            "hi_10",
        ],
    ),
    # 10 Portfolios formed on Book-to-Market
    "Portfolios_Formed_on_BE-ME_Daily.csv": (
        "btm",
        [
            "lo_10",
            "dec_2",
            "dec_3",
            "dec_4",
            "dec_5",
            "dec_6",
            "dec_7",
            "dec_8",
            "dec_9",
            "hi_10",
        ],
    ),
    # 10 Portfolios based on Momentum
    "10_Portfolios_Prior_12_2_Daily.CSV": (
        "momentum",
        [
            "lo_prior",
            "prior_2",
            "prior_3",
            "prior_4",
            "prior_5",
            "prior_6",
            "prior_7",
            "prior_8",
            "prior_9",
            "hi_prior",
        ],
    ),
    # 6 Portfolios based on size and book to market
    "6_Portfolios_2x3_daily.csv": (
        "sizebtm",
        ["small_lobm", "me1_bm2", "small_hibm", "big_lobm", "me2_bm2", "big_hibm"],
    ),
    # 10 Industry Portfolios
    "10_Industry_Portfolios_Daily.csv": (
        "industry",
        [
            "nodur",
            "durbl",
            "manuf",
            "enrgy",
            "hitec",
            "telcm",
            "shops",
            "hlth",
            "utils",
            "other",
        ],
    ),
    # 6 Portfolios Formed on Size and Momentum
    "6_Portfolios_ME_Prior_12_2_Daily.CSV": (
        "sizemomentum",
        [
            "small_loprior",
            "me1_prior2",
            "small_hiprior",
            "big_loprior",
            "me2_prior2",
            "big_hiprior",
        ],
    ),
    # 6 Portfolios Formed on Size and Short Term Reversal
    "6_Portfolios_ME_Prior_1_0_Daily.CSV": (
        "size_str",
        [
            "small_loprior",
            "me1_prior2",
            "small_hiprior",
            "big_loprior",
            "me2_prior2",
            "big_hiprior",
        ],
    ),
    # 6 Portfolios Formed on Size and Long Term Reversal
    "6_Portfolios_ME_Prior_60_13_Daily.CSV": (
        "size_ltr",
        [
            "small_loprior",
            "me1_prior2",
            "small_hiprior",
            "big_loprior",
            "me2_prior2",
            "big_hiprior",
        ],
    ),
}

BUCKET_NAME = "quant-finance-data"
DATES = pd.date_range("19900101", "20240401", freq="MS")
L = 120
