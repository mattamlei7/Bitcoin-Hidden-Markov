import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import pandas_ta as ta
from datetime import datetime

# ── 1. Load model and scaler ─────────────────────────────────────────────────
print("Loading model and scaler...")
scaler = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/scaler.joblib")
model  = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/model.joblib")

# ── 2. Fetch last 5 days of BTC hourly data via yfinance ─────────────────────
print("Fetching BTC-USD hourly data from Yahoo Finance...")
raw = yf.download("BTC-USD", period="5d", interval="1h", progress=False, auto_adjust=True)

if raw.empty:
    print("ERROR: No data returned from yfinance.")
    exit(1)

# Flatten MultiIndex columns if present
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
df.columns = ["open", "high", "low", "close", "volume"]
df.index.name = "datetime"
df = df.dropna()
print(f"  Fetched {len(df)} candles  ({df.index[0]} → {df.index[-1]})")

# ── 3. Calculate features ────────────────────────────────────────────────────
print("Calculating features...")
df["volume_pct_change"] = df["volume"].pct_change()
df["returns"]           = df["close"].pct_change()
df["volatility"]        = df["returns"].rolling(window=20).std()

bbands = ta.bbands(df["close"], length=20)
if bbands is None or bbands.empty:
    print("ERROR: Not enough data to calculate Bollinger Bands (need 20+ candles).")
    exit(1)

bb_width_col = [c for c in bbands.columns if c.startswith("BBB_")][0]
df["bb_width"] = bbands[bb_width_col]

df.drop(columns=["returns"], inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

if len(df) == 0:
    print("ERROR: No valid rows after feature calculation.")
    exit(1)

# ── 4. Predict current state ─────────────────────────────────────────────────
X_scaled = scaler.transform(df[["volume_pct_change", "volatility", "bb_width"]].values)
states   = model.predict(X_scaled)
df["state"] = states

current_state = int(states[-1])
current_price = float(df["close"].iloc[-1])
current_time  = df.index[-1]
recent_states = list(states[-5:].astype(int))

# ── 5. Determine signal ──────────────────────────────────────────────────────
if current_state == 0:
    signal = "BUY"
    icon   = "✅"
elif current_state == 2:
    signal = "SELL"
    icon   = "❌"
else:
    signal = "HOLD"
    icon   = "⏸ "

# ── 6. Print report ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("   BTC/USD  LIVE TRADING SIGNAL")
print("=" * 50)
print(f"  Signal Time:       {current_time}")
print(f"  Current Price:     ${current_price:,.2f}")
print(f"  Current State:     {current_state}")
print(f"  Recent States:     {recent_states}  (last 5 hours)")
print(f"  Volatility:        {df['volatility'].iloc[-1]:.6f}")
print(f"  BB Width:          {df['bb_width'].iloc[-1]:.4f}")
print(f"  Volume % Change:   {df['volume_pct_change'].iloc[-1]:.4f}")
print("-" * 50)
print(f"  SIGNAL:   {icon}  {signal}")
print("-" * 50)
print("  State key:  0 = BUY  |  2 = SELL  |  other = HOLD")
print("=" * 50)
print(f"\nGenerated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
