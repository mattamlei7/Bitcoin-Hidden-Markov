import pandas as pd

df = pd.read_csv(
    "C:/Users/matth/crypto_hmm_trader/Binance_BTCUSDT_1h.csv",
    skiprows=1
)

df = df[["Date", "Open", "High", "Low", "Close", "Volume BTC"]]
df.columns = ["datetime", "open", "high", "low", "close", "volume"]

# Strip any trailing milliseconds (e.g. .000) before parsing
df["datetime"] = df["datetime"].astype(str).str.split(".").str[0]
df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
df.set_index("datetime", inplace=True)

df.sort_index(inplace=True)
df.dropna(inplace=True)

df.to_csv("C:/Users/matth/crypto_hmm_trader/btc_clean.csv")

print(f"Saved {len(df)} rows to btc_clean.csv\n")
print(df.head())
