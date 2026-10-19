import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import joblib
import os
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from backtesting import Backtest, Strategy

# ── 1. Load and clean ETH data ───────────────────────────────────────────────
print("Step 1: Loading and cleaning ETH data...")
df = pd.read_csv("C:/Users/matth/Downloads/Binance_ETHUSDT_1h.csv", skiprows=1)
df = df[["Date", "Open", "High", "Low", "Close", "Volume ETH"]]
df.columns = ["datetime", "open", "high", "low", "close", "volume"]
df["datetime"] = df["datetime"].astype(str).str.split(".").str[0]
df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)
df.dropna(inplace=True)
print(f"  {len(df)} rows loaded")

# ── 2. Calculate features ────────────────────────────────────────────────────
print("Step 2: Calculating features...")
df["volume_pct_change"] = df["volume"].pct_change()
df["returns"]           = df["close"].pct_change()
df["volatility"]        = df["returns"].rolling(window=20).std()

bbands = ta.bbands(df["close"], length=20)
bb_width_col = [c for c in bbands.columns if c.startswith("BBB_")][0]
df["bb_width"] = bbands[bb_width_col]

df.drop(columns=["returns"], inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"  {len(df)} rows after feature calculation")

# ── 3. Train ETH-specific HMM ────────────────────────────────────────────────
print("\nStep 3: Training ETH HMM (5 states)...")
features = ["volume_pct_change", "volatility", "bb_width"]
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GaussianHMM(n_components=5, covariance_type="full",
                    n_iter=100, random_state=42)
model.fit(X_scaled)

df["state"] = model.predict(X_scaled)
print(f"  Log Likelihood: {model.score(X_scaled):.4f}")
print(f"  State distribution:")
for s, c in df["state"].value_counts().sort_index().items():
    flag = "  <-- WARNING: too few" if c < 1000 else ""
    print(f"    State {s}: {c:>6} rows{flag}")

# Save ETH model and scaler
os.makedirs("C:/Users/matth/crypto_hmm_trader/trained_models", exist_ok=True)
joblib.dump(model,  "C:/Users/matth/crypto_hmm_trader/trained_models/eth_model.joblib")
joblib.dump(scaler, "C:/Users/matth/crypto_hmm_trader/trained_models/eth_scaler.joblib")
print("  ETH model and scaler saved to trained_models/")

# ── 4. Prepare backtest dataframe ────────────────────────────────────────────
df = df.rename(columns={"open": "Open", "high": "High",
                         "low": "Low",  "close": "Close", "volume": "Volume"})

# ── 5. Strategy ──────────────────────────────────────────────────────────────
class HMMStrategy(Strategy):
    buy_state  = 0
    sell_state = 1

    def init(self):
        self.hmm_state = self.I(lambda x: x, self.data.state, name="hmm_state", plot=False)

    def next(self):
        s = self.hmm_state[-1]
        if s == self.buy_state and not self.position:
            units = int(self.equity * 0.95 / self.data.Close[-1])
            if units > 0:
                self.buy(size=units)
        elif s == self.sell_state and self.position:
            self.position.close()

# ── 6. Loop all buy/sell combinations ────────────────────────────────────────
print("\nStep 4: Testing all buy/sell state combinations...")
results = []

for buy in range(5):
    for sell in range(5):
        if buy == sell:
            continue
        bt    = Backtest(df, HMMStrategy, cash=10_000, commission=0.002, exclusive_orders=True)
        stats = bt.run(buy_state=buy, sell_state=sell)
        ret    = round(stats["Return [%]"], 2)
        sharpe = round(stats["Sharpe Ratio"], 4)
        trades = stats["# Trades"]
        results.append({"buy": buy, "sell": sell, "return_%": ret, "sharpe": sharpe, "trades": trades})
        print(f"  buy={buy} sell={sell} | Return: {ret:>9.2f}% | Sharpe: {sharpe:>7.4f} | Trades: {trades}")

# ── 7. Top 5 by Sharpe ───────────────────────────────────────────────────────
results.sort(key=lambda r: r["sharpe"], reverse=True)

print("\n" + "=" * 60)
print("TOP 5 ETH COMBINATIONS BY SHARPE RATIO")
print("=" * 60)
print(f"{'Buy':>4} {'Sell':>5} {'Return %':>10} {'Sharpe':>8} {'Trades':>7}")
print("-" * 60)
for r in results[:5]:
    print(f"{r['buy']:>4} {r['sell']:>5} {r['return_%']:>10.2f} {r['sharpe']:>8.4f} {r['trades']:>7}")
print("=" * 60)
