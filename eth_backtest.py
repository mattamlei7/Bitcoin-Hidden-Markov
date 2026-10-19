import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import joblib
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.io as pio
from backtesting import Backtest, Strategy

# ── 1. Load and clean ETH data ───────────────────────────────────────────────
print("Step 1: Loading ETH data...")
df = pd.read_csv("C:/Users/matth/Downloads/Binance_ETHUSDT_1h.csv", skiprows=1)
df = df[["Date", "Open", "High", "Low", "Close", "Volume ETH"]]
df.columns = ["datetime", "open", "high", "low", "close", "volume"]
df["datetime"] = df["datetime"].astype(str).str.split(".").str[0]
df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)
df.dropna(inplace=True)
print(f"  {len(df)} rows loaded")

# ── 2. Calculate features (same as BTC) ─────────────────────────────────────
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

# ── 3. Load BTC model and scaler, predict states ─────────────────────────────
print("Step 3: Predicting states using BTC-trained model...")
scaler = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/scaler.joblib")
model  = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/model.joblib")
X_scaled   = scaler.transform(df[["volume_pct_change", "volatility", "bb_width"]].values)
df["state"] = model.predict(X_scaled)
print(f"  State distribution:\n{df['state'].value_counts().sort_index().to_string()}")

# ── 4. Rename for backtesting.py ─────────────────────────────────────────────
df = df.rename(columns={"open": "Open", "high": "High",
                         "low": "Low",  "close": "Close", "volume": "Volume"})

print(f"\nData ready: {len(df)} rows\n")

# ── 5. Strategy ──────────────────────────────────────────────────────────────
class HMMStrategy(Strategy):
    buy_state  = 0
    sell_state = 2

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

# ── 6. Run backtest ──────────────────────────────────────────────────────────
print("Running ETH backtest (buy=0, sell=2)...")
bt    = Backtest(df, HMMStrategy, cash=10_000, commission=0.002, exclusive_orders=True)
stats = bt.run()

# ── 7. Print results ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ETH FULL BACKTEST RESULTS")
print("=" * 55)
for key, val in stats.items():
    if key not in ["_strategy", "_trades", "_equity_curve"]:
        print(f"  {key:<35} {val}")

print("\n" + "=" * 55)
print("KEY METRICS SUMMARY")
print("=" * 55)
print(f"  Exposure Time (in market)      {stats['Exposure Time [%]']:.2f}%")
print(f"  Time in Cash                   {100 - stats['Exposure Time [%]']:.2f}%")
print(f"  Return                         {stats['Return [%]']:.2f}%")
print(f"  Buy & Hold Return              {stats['Buy & Hold Return [%]']:.2f}%")
print(f"  Sharpe Ratio                   {stats['Sharpe Ratio']:.4f}")
print(f"  Max Drawdown                   {stats['Max. Drawdown [%]']:.2f}%")
print(f"  Number of Trades               {stats['# Trades']}")
print(f"  Win Rate                       {stats['Win Rate [%]']:.2f}%")
print(f"  Avg Trade Duration             {stats['Avg. Trade Duration']}")
print(f"  Best Trade                     {stats['Best Trade [%]']:.2f}%")
print(f"  Worst Trade                    {stats['Worst Trade [%]']:.2f}%")
print(f"  Profit Factor                  {stats['Profit Factor']:.4f}")
print(f"  Final Equity                   ${stats['Equity Final [$]']:,.2f}")
print("=" * 55)

# ── 8. Save equity curve ─────────────────────────────────────────────────────
print("\nSaving equity curve...")
equity = stats["_equity_curve"]["Equity"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines",
                         line=dict(color="purple", width=1.5), name="ETH Strategy"))
fig.add_hline(y=10_000, line=dict(color="gray", dash="dash"),
              annotation_text="Starting Capital $10,000")
fig.update_layout(
    title="ETH Backtest - BTC-Trained HMM (buy=0, sell=2)",
    xaxis_title="Date", yaxis_title="Portfolio Value (USD)", height=500
)
pio.write_html(fig, "C:/Users/matth/crypto_hmm_trader/eth_equity_curve.html",
               include_plotlyjs="cdn")
print("Saved to eth_equity_curve.html")
