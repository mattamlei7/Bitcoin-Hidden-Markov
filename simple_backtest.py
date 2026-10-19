import numpy as np
import pandas as pd
import joblib

# ── 1. Load and filter ───────────────────────────────────────────────────────
print("Loading btc_features.csv...")
df = pd.read_csv("C:/Users/matth/crypto_hmm_trader/btc_features.csv",
                 index_col="datetime", parse_dates=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df = df.loc["2025-03-01":"2026-03-31"]
print(f"  Rows: {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")

# ── 2. Predict states ────────────────────────────────────────────────────────
print("Predicting states...")
scaler = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/scaler.joblib")
model  = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/model.joblib")
X_scaled   = scaler.transform(df[["volume_pct_change", "volatility", "bb_width"]].values)
df["state"] = model.predict(X_scaled)

print(f"  State distribution:")
for s, c in df["state"].value_counts().sort_index().items():
    print(f"    State {s}: {c} rows")

# ── 3. Manual simulation ─────────────────────────────────────────────────────
print("\nRunning manual simulation...")

cash        = 10_000.0
btc_held    = 0.0
in_position = False
entry_price = 0.0
entry_time  = None
trades      = []
prev_state  = None

prices = df["close"].values
states = df["state"].values
times  = df.index

for i in range(len(df)):
    price      = prices[i]
    state      = states[i]
    timestamp  = times[i]

    # Buy when state transitions TO 0 and not in position
    if state == 0 and not in_position:
        btc_held    = (cash * 0.95) / price
        cash       -= btc_held * price
        in_position = True
        entry_price = price
        entry_time  = timestamp

    # Sell when state transitions TO 2 and in position
    elif state == 2 and in_position:
        proceeds    = btc_held * price
        pnl         = proceeds - (btc_held * entry_price)
        pnl_pct     = (price - entry_price) / entry_price * 100
        cash       += proceeds
        trades.append({
            "entry_time":  entry_time,
            "exit_time":   timestamp,
            "entry_price": round(entry_price, 2),
            "exit_price":  round(price, 2),
            "pnl":         round(pnl, 2),
            "pnl_pct":     round(pnl_pct, 2),
            "win":         pnl > 0
        })
        btc_held    = 0.0
        in_position = False

    prev_state = state

# Close any open position at last price
final_price = prices[-1]
if in_position:
    proceeds = btc_held * final_price
    pnl      = proceeds - (btc_held * entry_price)
    pnl_pct  = (final_price - entry_price) / entry_price * 100
    cash    += proceeds
    trades.append({
        "entry_time":  entry_time,
        "exit_time":   times[-1],
        "entry_price": round(entry_price, 2),
        "exit_price":  round(final_price, 2),
        "pnl":         round(pnl, 2),
        "pnl_pct":     round(pnl_pct, 2),
        "win":         pnl > 0
    })

# ── 4. Results ───────────────────────────────────────────────────────────────
final_value   = cash
total_return  = (final_value - 10_000) / 10_000 * 100
n_trades      = len(trades)
wins          = sum(1 for t in trades if t["win"])
win_rate      = (wins / n_trades * 100) if n_trades > 0 else 0
buy_hold      = (final_price / prices[0] - 1) * 100

print("\n" + "=" * 55)
print("SIMPLE MANUAL BACKTEST RESULTS  (Last 12 Months)")
print("=" * 55)
print(f"  Starting Capital:    $10,000.00")
print(f"  Final Value:         ${final_value:,.2f}")
print(f"  Total Return:        {total_return:.2f}%")
print(f"  Buy & Hold Return:   {buy_hold:.2f}%")
print(f"  Number of Trades:    {n_trades}")
print(f"  Win Rate:            {win_rate:.2f}%")
print(f"  Winning Trades:      {wins}")
print(f"  Losing Trades:       {n_trades - wins}")
print("=" * 55)

if trades:
    print("\nAll Trades:")
    print(f"  {'Entry Time':<22} {'Exit Time':<22} {'Entry $':>10} {'Exit $':>10} {'P&L':>10} {'P&L %':>7}")
    print("  " + "-" * 85)
    for t in trades:
        flag = " WIN" if t["win"] else " LOSS"
        print(f"  {str(t['entry_time']):<22} {str(t['exit_time']):<22} "
              f"{t['entry_price']:>10,.2f} {t['exit_price']:>10,.2f} "
              f"{t['pnl']:>+10,.2f} {t['pnl_pct']:>+6.2f}%{flag}")
