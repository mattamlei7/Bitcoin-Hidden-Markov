import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from backtesting import Backtest, Strategy

# ── 1. Load full dataset ─────────────────────────────────────────────────────
print("Step 1: Loading full btc_features.csv...")
df = pd.read_csv("C:/Users/matth/crypto_hmm_trader/btc_features.csv",
                 index_col="datetime", parse_dates=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"  Full dataset: {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")

# ── 2. Train on FULL dataset ─────────────────────────────────────────────────
print("\nStep 2: Training HMM on full dataset (5 states)...")
features = ["volume_pct_change", "volatility", "bb_width"]
X_all = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

model = GaussianHMM(n_components=5, covariance_type="full",
                    n_iter=100, random_state=42)
model.fit(X_scaled)
df["state"] = model.predict(X_scaled)

print(f"  Log Likelihood: {model.score(X_scaled):.4f}")
print(f"  Full dataset state distribution:")
for s, c in df["state"].value_counts().sort_index().items():
    print(f"    State {s}: {c:>6} rows")

os.makedirs("C:/Users/matth/crypto_hmm_trader/trained_models", exist_ok=True)
joblib.dump(model,  "C:/Users/matth/crypto_hmm_trader/trained_models/recent_model.joblib")
joblib.dump(scaler, "C:/Users/matth/crypto_hmm_trader/trained_models/recent_scaler.joblib")
print("  Saved recent_model.joblib and recent_scaler.joblib")

# ── 3. Filter backtest window: last 12 months ────────────────────────────────
print("\nStep 3: Filtering to last 12 months for backtest...")
bt_df = df.loc["2025-03-01":"2026-03-31"].copy()
print(f"  Rows: {len(bt_df)}  ({bt_df.index[0].date()} → {bt_df.index[-1].date()})")

# ── 4. State distribution in backtest window ─────────────────────────────────
print("\n" + "=" * 50)
print("STATE DISTRIBUTION  (Last 12 Months)")
print("=" * 50)
state_counts = bt_df["state"].value_counts().sort_index()
valid_states = []
for s, c in state_counts.items():
    flag = "  EXCLUDED (< 100 rows)" if c < 100 else "  included"
    print(f"  State {s}: {c:>5} rows{flag}")
    if c >= 100:
        valid_states.append(s)
print(f"\n  Valid states for testing: {valid_states}")
print("=" * 50)

# ── 5. Prepare dataframe for backtesting.py ──────────────────────────────────
bt_df = bt_df.rename(columns={"open": "Open", "high": "High",
                                "low": "Low",  "close": "Close", "volume": "Volume"})

# ── 6. Strategy ──────────────────────────────────────────────────────────────
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

# ── 7. Test all valid combinations ───────────────────────────────────────────
print("\nStep 4: Testing valid buy/sell state combinations...")
results = []

for buy in valid_states:
    for sell in valid_states:
        if buy == sell:
            continue
        bt    = Backtest(bt_df, HMMStrategy, cash=10_000, commission=0.002, exclusive_orders=True)
        stats = bt.run(buy_state=buy, sell_state=sell)
        ret      = round(stats["Return [%]"], 2)
        sharpe   = round(stats["Sharpe Ratio"], 4)
        trades   = stats["# Trades"]
        drawdown = round(stats["Max. Drawdown [%]"], 2)
        winrate  = round(stats["Win Rate [%]"], 2)
        results.append({"buy": buy, "sell": sell, "return_%": ret,
                         "sharpe": sharpe, "trades": trades,
                         "max_dd": drawdown, "win_rate": winrate})
        print(f"  buy={buy} sell={sell} | Return: {ret:>8.2f}% | "
              f"Sharpe: {sharpe:>7.4f} | Trades: {trades:>4} | "
              f"MaxDD: {drawdown:.2f}%")

# ── 8. Top 5 ─────────────────────────────────────────────────────────────────
results.sort(key=lambda r: r["sharpe"], reverse=True)

print("\n" + "=" * 72)
print("TOP 5 COMBINATIONS  (Full-Dataset Model, Last 12 Months)")
print("=" * 72)
print(f"{'Buy':>4} {'Sell':>5} {'Return %':>10} {'Sharpe':>8} "
      f"{'Trades':>7} {'Max DD %':>9} {'Win Rate':>9}")
print("-" * 72)
for r in results[:5]:
    print(f"{r['buy']:>4} {r['sell']:>5} {r['return_%']:>10.2f} "
          f"{r['sharpe']:>8.4f} {r['trades']:>7} "
          f"{r['max_dd']:>9.2f} {r['win_rate']:>8.2f}%")
print("=" * 72)
