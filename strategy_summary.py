import numpy as np
import pandas as pd
import joblib

# ── Load and prepare full dataset once ──────────────────────────────────────
print("Loading btc_features.csv...")
df = pd.read_csv("C:/Users/matth/crypto_hmm_trader/btc_features.csv",
                 index_col="datetime", parse_dates=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Predicting states on full dataset...")
scaler = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/scaler.joblib")
model  = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/model.joblib")
X_scaled   = scaler.transform(df[["volume_pct_change", "volatility", "bb_width"]].values)
df["state"] = model.predict(X_scaled)
print(f"  Full dataset: {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})\n")

# ── Simulation function ──────────────────────────────────────────────────────
def run_simulation(df_slice, label):
    cash        = 10_000.0
    btc_held    = 0.0
    in_position = False
    entry_price = 0.0
    entry_time  = None
    trades      = []

    prices = df_slice["close"].values
    states = df_slice["state"].values
    times  = df_slice.index

    for i in range(len(df_slice)):
        price     = prices[i]
        state     = int(states[i])
        timestamp = times[i]

        if state == 0 and not in_position:
            btc_held    = (cash * 0.95) / price
            cash       -= btc_held * price
            in_position = True
            entry_price = price
            entry_time  = timestamp

        elif state == 2 and in_position:
            proceeds = btc_held * price
            pnl_pct  = (price - entry_price) / entry_price * 100
            cash    += proceeds
            trades.append({"pnl_pct": round(pnl_pct, 2), "win": proceeds > btc_held * entry_price})
            btc_held    = 0.0
            in_position = False

    # Close open position at last price
    if in_position:
        proceeds = btc_held * prices[-1]
        pnl_pct  = (prices[-1] - entry_price) / entry_price * 100
        cash    += proceeds
        trades.append({"pnl_pct": round(pnl_pct, 2), "win": proceeds > btc_held * entry_price})

    final_value  = cash
    total_return = (final_value - 10_000) / 10_000 * 100
    buy_hold     = (prices[-1] / prices[0] - 1) * 100
    n_trades     = len(trades)
    wins         = sum(1 for t in trades if t["win"])
    win_rate     = (wins / n_trades * 100) if n_trades > 0 else 0
    max_loss     = min((t["pnl_pct"] for t in trades), default=0)

    return {
        "label":         label,
        "start":         str(times[0].date()),
        "end":           str(times[-1].date()),
        "rows":          len(df_slice),
        "return_%":      round(total_return, 2),
        "buy_hold_%":    round(buy_hold, 2),
        "final_value":   round(final_value, 2),
        "n_trades":      n_trades,
        "win_rate":      round(win_rate, 2),
        "max_loss_%":    round(max_loss, 2),
    }

# ── Run on three periods ─────────────────────────────────────────────────────
periods = [
    ("Full Dataset  (2017-2026)", None,        None),
    ("Last 2 Years  (2024-2026)", "2024-01-01", None),
    ("Last 12 Months (2025-2026)", "2025-03-01", "2026-03-31"),
]

results = []
for label, start, end in periods:
    slc = df.loc[start:end] if start else df
    print(f"Running: {label}  ({len(slc)} rows)...")
    results.append(run_simulation(slc, label))

# ── Print summary table ──────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  HMM STRATEGY SUMMARY  —  buy state=0  |  sell state=2")
print("=" * 75)

headers = ["Period", "Return %", "Buy&Hold %", "Final $", "Trades", "Win Rate", "Max Loss %"]
print(f"  {'Period':<28} {'Return %':>9} {'B&H %':>9} {'Final $':>11} "
      f"{'Trades':>7} {'Win Rate':>9} {'Max Loss':>9}")
print("  " + "-" * 73)

for r in results:
    print(f"  {r['label']:<28} "
          f"{r['return_%']:>8.2f}% "
          f"{r['buy_hold_%']:>8.2f}% "
          f"${r['final_value']:>10,.2f} "
          f"{r['n_trades']:>7} "
          f"{r['win_rate']:>8.2f}% "
          f"{r['max_loss_%']:>8.2f}%")

print("=" * 75)
print(f"\n  Strategy: buy when HMM enters state 0, sell when HMM enters state 2")
print(f"  Position size: 95% of cash per trade  |  No leverage")
print(f"  Model: GaussianHMM  5 states  full covariance  trained on full BTC dataset")
