import pandas as pd
import pandas_ta as ta

df = pd.read_csv(
    "C:/Users/matth/crypto_hmm_trader/btc_clean.csv",
    index_col="datetime",
    parse_dates=True
)

# 1. Volume percent change
df["volume_pct_change"] = df["volume"].pct_change()

# 2. Rolling volatility: 20-period std of log returns
df["returns"] = df["close"].pct_change()
df["volatility"] = df["returns"].rolling(window=20).std()

# 3. Bollinger Band width using pandas-ta
bbands = ta.bbands(df["close"], length=20)
# bbands returns columns like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
bb_width_col = [c for c in bbands.columns if c.startswith("BBB_")][0]
df["bb_width"] = bbands[bb_width_col]

# Drop helper column and NaNs
df.drop(columns=["returns"], inplace=True)
df.dropna(inplace=True)

df.to_csv("C:/Users/matth/crypto_hmm_trader/btc_features.csv")

print(f"Total rows: {len(df)}\n")
print(df[["close", "volume_pct_change", "volatility", "bb_width"]].head())
