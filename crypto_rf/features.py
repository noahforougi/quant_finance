import pandas as pd
import talib
from tqdm import tqdm
import numpy as np
### TODO: if we are trading in 6 hour forward periods, we are likely going to want to 
### exclude the next hour returns as we will have to retrain the model at certain 
### points. 

def add_targets(df, periods):
    for p in periods:
        # Calculate lagged close price and returns
        df["lag_close"] = df.groupby("symbol")["close"].shift(p)
        df[f"ret_{p}"] = df["close"] / df["lag_close"] - 1

        # Merge median returns
        median_ret = df.groupby("date")[f"ret_{p}"].median().to_frame().rename(columns={f"ret_{p}": f"ret_{p}_med"}).reset_index()
        df = df.reset_index().merge(median_ret, on="date", how="left").set_index(["date", "symbol"])

        # Calculate above-median flag and forward-looking metrics
        df[f"ret_{p}_above_med"] = (df[f"ret_{p}"] > df[f"ret_{p}_med"]).astype(int)
        for suffix in ["", "_med", "_above_med"]:
            col = f"ret_{p}{suffix}"
            df[f"fwd_{col}"] = df[col].groupby("symbol").shift(-p)

    # Cleanup
    df.drop(columns=["lag_close"], inplace=True)
    df.sort_index(inplace=True)

    return df

def _create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Existing features
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
    df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])

    # Return-based features
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['volatility_10'] = df['return_1'].rolling(window=10).std()

    # Volume-based features
    df['volume_ma_10'] = df['volume_asset'].rolling(window=10).mean()
    df['volume_oscillator'] = (df['volume_asset'] - df['volume_ma_10']) / df['volume_ma_10']

    # Additional features
    ## Moving Averages and Crossovers
    df['sma_200'] = talib.SMA(df['close'], timeperiod=200)  # Long-term trend
    df['ema_50'] = talib.EMA(df['close'], timeperiod=50)    # Medium-term trend
    df['price_sma_ratio'] = df['close'] / df['sma_50']      # Price relative to SMA
    df['sma_crossover'] = (df['sma_50'] > df['sma_200']).astype(int)  # SMA crossover signal

    ## Bollinger Bands
    upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['bb_upper'] = upperband
    df['bb_middle'] = middleband
    df['bb_lower'] = lowerband
    df['bb_bandwidth'] = (upperband - lowerband) / middleband  # Bollinger Bandwidth
    df['close_bb_position'] = (df['close'] - lowerband) / (upperband - lowerband)  # Position in BB

    ## Relative Strength
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)  # Commodity Channel Index

    ## Trends and Reversals
    df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)  # Williams %R
    df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)  # Parabolic SAR

    ## Advanced Volatility
    df['rolling_volatility_20'] = df['return_1'].rolling(window=20).std()  # Long-term rolling volatility
    df['rolling_skew_20'] = df['return_1'].rolling(window=20).skew()  # Skewness of returns

    ## Volume and Price Interaction
    df['price_volume_trend'] = (df['return_1'] * df['volume_asset']).cumsum()  # Price-volume trend
    df['obv'] = talib.OBV(df['close'], df['volume_asset'])  # On-balance volume

    ## Feature Interactions
    df['return_vol_ratio'] = df['return_1'] / (df['volatility_10'] + 1e-9)  # Return-to-volatility ratio
    df['volume_price_ratio'] = df['volume_asset'] / (df['close'] + 1e-9)  # Volume relative to price

    return df


def add_features(df):
    features = list()
    for _, group in tqdm(df.groupby(level='symbol')):
        group = _create_features(group)
        features.append(group)
    return pd.concat(features)

def get_features(df): 
    return [x for x in df.columns if x not in ["open", "high", "low", "close", "volume_usd", "volume_asset",
                                               "ret1", "ret3", "ret6", "ret24", "ret72", "ret168",
                                                "ret_1_med", "ret_1_above_med", "fwd_ret_1", "fwd_ret_1_med", "fwd_ret_1_above_med", 
                                                "ret_3_med", "ret_3_above_med", "fwd_ret_3", "fwd_ret_3_med", "fwd_ret_3_above_med", 
                                                "ret_6_med", "ret_6_above_med", "fwd_ret_6", "fwd_ret_6_med", "fwd_ret_6_above_med", 
                                                "ret_24_med", "ret_24_above_med", "fwd_ret_24", "fwd_ret_24_med", "fwd_ret_24_above_med", 
                                                "ret_72_med", "ret_72_above_med", "fwd_ret_72", "fwd_ret_72_med", "fwd_ret_72_above_med", 
                                                "ret_168_med", "ret_168_above_med", "fwd_ret_168", "fwd_ret_168_med", "fwd_ret_168_above_med"]]
def get_zscore_df(df: pd.DataFrame): 
    tmp = df.copy()
    FEATURES = get_features(tmp)
        
    # Compute means and standard deviations for each date and feature
    grouped = tmp.groupby(level="date")[FEATURES]
    means = grouped.transform('mean')
    stds = grouped.transform('std')

    # Compute z-scores
    tmp_zscore = (tmp[FEATURES] - means) / stds
    tmp.update(tmp_zscore)

    return tmp

def process_data(df, periods):
    df = add_targets(df, periods)
    df = add_features(df)
    df = df.replace([np.nan, np.inf, -np.inf], 0)
    df_zscore = get_zscore_df(df)
    df_zscore = df_zscore.replace([np.nan, np.inf, -np.inf], 0)
    return df, df_zscore
    

def main(): 
    PERIODS = [6]#[1,3,6,24,72,168]

    df = pd.read_csv("../data/clean/prices.csv").set_index(["date", "symbol"])
    df.sort_values(["symbol", "date"], inplace=True)
    df_with_features, df_with_features_z = process_data(df, PERIODS)
    df_with_features.to_csv("../data/clean/df_with_features.csv")
    df_with_features_z.to_csv("../data/clean/dfz_with_features.csv")

if __name__ == "__main__": 
    main()