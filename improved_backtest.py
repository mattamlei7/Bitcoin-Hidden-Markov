import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.io as pio
from backtesting import Backtest, Strategy

# ── 1. Load and clean ────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("C:/Users/matth/crypto_hmm_trader/btc_features.csv",
                 index_col="datetime", parse_dates=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ── 2. Predict states ────────────────────────────────────────────────────────
print("Predicting states...")
scaler = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/scaler.joblib")
model  = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/model.joblib")
X_scaled = scaler.transform(df[["volume_pct_change", "volatility", "bb_width"]].values)
df["state"] = model.predict(X_scaled)

# ── 3. Precompute consecutive hours in state 0 ───────────────────────────────
print("Computing consecutive state 0 streaks...")
streak = []
count = 0
for s in df["state"]:
    count = count + 1 if s == 0 else 0
    streak.append(count)
df["state0_streak"] = streak
print(f"  Max consecutive state-0 hours: {df['state0_streak'].max()}")

# ── 4. Rename for backtesting.py ─────────────────────────────────────────────
df = df.rename(columns={"open": "Open", "high": "High",
                         "low": "Low",  "close": "Close", "volume": "Volume"})

print(f"Data ready: {len(df)} rows\n")

MAX_HOLD_BARS   = 30 * 24   # 30 days in hourly bars
MIN_STREAK      = 3         # consecutive state-0 hours before buying

class ImprovedStrategy(Strategy):
    def init(self):
        self.state   = self.I(lambda x: x, self.data.state,         name="state",   plot=False)
        self.streak  = self.I(lambda x: x, self.data.state0_streak,  name="streak",  plot=False)
        self.entry_bar = 0

    def next(self):
        current_state  = self.state[-1]
        current_streak = self.streak[-1]
        bars_held      = len(self.data) - self.entry_bar

        if not self.position:
            # Buy only when state 0 has been stable for at least 3 consecutive bars
            if current_streak >= MIN_STREAK:
                units = int(self.equity * 0.95 / self.data.Close[-1])
                if units > 0:
                    self.buy(size=units)
                    self.entry_bar = len(self.data)
        else:
            # Sell on state 2 OR if held longer than 30 days
            if current_state == 2 or bars_held >= MAX_HOLD_BARS:
                self.position.close()

# ── 5. Run backtest ──────────────────────────────────────────────────────────
print("Running backtest...")
bt    = Backtest(df, ImprovedStrategy, cash=10_000, commission=0.002, exclusive_orders=True)
stats = bt.run()

# ── 6. Print results ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("FULL BACKTEST RESULTS")
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

# ── 7. Save equity curve ─────────────────────────────────────────────────────
print("\nSaving equity curve...")
equity = stats["_equity_curve"]["Equity"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines",
                         line=dict(color="green", width=1.5), name="Improved Strategy"))
fig.add_hline(y=10_000, line=dict(color="gray", dash="dash"),
              annotation_text="Starting Capital $10,000")
fig.update_layout(
    title="Improved HMM Strategy Equity Curve (state-0 streak ≥3, 30-day max hold)",
    xaxis_title="Date", yaxis_title="Portfolio Value (USD)", height=500
)
pio.write_html(fig, "C:/Users/matth/crypto_hmm_trader/improved_equity_curve.html",
               include_plotlyjs="cdn")
print("Saved to improved_equity_curve.html")
