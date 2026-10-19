import numpy as np
import pandas as pd
import joblib
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timedelta

MODEL_DIR = "C:/Users/matth/Desktop/crypto_hmm_trader/trained_models"
OUTPUT    = "C:/Users/matth/Desktop/crypto_hmm_trader/last_month_equity_curve.html"

# ── 1. Fetch last month of hourly BTC data from Binance US ─────────────────
print("Fetching last 30 days of BTC/USDT hourly candles from Binance...")
import requests

end_ms   = int(datetime.utcnow().timestamp() * 1000)
start_ms = int((datetime.utcnow() - timedelta(days=32)).timestamp() * 1000)

all_candles = []
cursor = start_ms
while cursor < end_ms:
    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "startTime": cursor,
        "endTime": end_ms,
        "limit": 1000,
    }
    r = requests.get("https://api.binance.us/api/v3/klines", params=params, timeout=15)
    r.raise_for_status()
    batch = r.json()
    if not batch:
        break
    all_candles.extend(batch)
    cursor = batch[-1][6] + 1  # close_time + 1

df = pd.DataFrame(all_candles, columns=[
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
])
df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
df.set_index("datetime", inplace=True)
df = df[["open", "high", "low", "close", "volume"]].astype(float)

# Trim to exactly 30 days
cutoff = datetime.utcnow() - timedelta(days=30)
df = df[df.index >= pd.Timestamp(cutoff)]
print(f"  {len(df)} candles  ({df.index[0]} -> {df.index[-1]})")

# ── 2. Feature engineering ──────────────────────────────────────────────────
print("Computing features...")
df["volume_pct_change"] = df["volume"].pct_change()
df["returns"]           = df["close"].pct_change()
df["volatility"]        = df["returns"].rolling(20).std()

bbands = ta.bbands(df["close"], length=20)
bb_col = [c for c in bbands.columns if c.startswith("BBB_")][0]
df["bb_width"] = bbands[bb_col]

df.drop(columns=["returns"], inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"  {len(df)} rows after feature calc")

# ── 3. Load model and predict states ────────────────────────────────────────
print("Loading model and predicting HMM states...")
scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
model  = joblib.load(f"{MODEL_DIR}/model.joblib")

X_scaled    = scaler.transform(df[["volume_pct_change", "volatility", "bb_width"]].values)
df["state"] = model.predict(X_scaled)

print(f"  State distribution:")
for s, c in df["state"].value_counts().sort_index().items():
    print(f"    State {s}: {c} rows")

# ── 4. Simulate trades (same logic as paper_trader) ─────────────────────────
print("\nRunning backtest simulation...")
BUY_STATE  = 0
SELL_STATE = 2

cash        = 10_000.0
btc_held    = 0.0
in_position = False
entry_price = 0.0
entry_time  = None
trades      = []

prices = df["close"].values
states = df["state"].values
times  = df.index

portfolio_values = []

for i in range(len(df)):
    price = prices[i]
    state = states[i]
    ts    = times[i]

    if state == BUY_STATE and not in_position:
        btc_held    = (cash * 0.95) / price
        cash       -= btc_held * price
        in_position = True
        entry_price = price
        entry_time  = ts

    elif state == SELL_STATE and in_position:
        proceeds = btc_held * price
        pnl      = proceeds - (btc_held * entry_price)
        pnl_pct  = (price - entry_price) / entry_price * 100
        cash    += proceeds
        trades.append({
            "entry_time":  entry_time,
            "exit_time":   ts,
            "entry_price": round(entry_price, 2),
            "exit_price":  round(price, 2),
            "pnl":         round(pnl, 2),
            "pnl_pct":     round(pnl_pct, 2),
            "win":         pnl > 0,
        })
        btc_held    = 0.0
        in_position = False

    portfolio_values.append(cash + btc_held * price)

# Close open position at last price
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
        "win":         pnl > 0,
    })
    btc_held    = 0.0
    in_position = False
    portfolio_values[-1] = cash

# ── 5. Results ──────────────────────────────────────────────────────────────
final_value  = cash
total_return = (final_value - 10_000) / 10_000 * 100
n_trades     = len(trades)
wins         = sum(1 for t in trades if t["win"])
losses       = n_trades - wins
win_rate     = (wins / n_trades * 100) if n_trades > 0 else 0
buy_hold     = (final_price / prices[0] - 1) * 100

avg_win  = np.mean([t["pnl"] for t in trades if t["win"]]) if wins else 0
avg_loss = np.mean([t["pnl"] for t in trades if not t["win"]]) if losses else 0

pv_series   = pd.Series(portfolio_values, index=times)
peak        = pv_series.cummax()
drawdown    = (pv_series - peak) / peak * 100
max_dd      = drawdown.min()

print("\n" + "=" * 60)
print(f"  BACKTEST RESULTS — LAST 30 DAYS")
print(f"  {df.index[0].date()} to {df.index[-1].date()}")
print("=" * 60)
print(f"  Starting Capital:      $10,000.00")
print(f"  Final Value:           ${final_value:,.2f}")
print(f"  Strategy Return:       {total_return:+.2f}%")
print(f"  Buy & Hold Return:     {buy_hold:+.2f}%")
print(f"  Max Drawdown:          {max_dd:.2f}%")
print(f"  Number of Trades:      {n_trades}")
print(f"  Win Rate:              {win_rate:.1f}%  ({wins}W / {losses}L)")
print(f"  Avg Win:               ${avg_win:+,.2f}")
print(f"  Avg Loss:              ${avg_loss:+,.2f}")
print("=" * 60)

if trades:
    print(f"\n  {'Entry Time':<22} {'Exit Time':<22} {'Entry $':>10} {'Exit $':>10} {'P&L':>10} {'%':>7}")
    print("  " + "-" * 85)
    for t in trades:
        tag = "WIN" if t["win"] else "LOSS"
        print(f"  {str(t['entry_time']):<22} {str(t['exit_time']):<22} "
              f"{t['entry_price']:>10,.2f} {t['exit_price']:>10,.2f} "
              f"{t['pnl']:>+10,.2f} {t['pnl_pct']:>+6.2f}%  {tag}")

# ── 6. Equity curve chart ───────────────────────────────────────────────────
print("\nSaving equity curve...")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=times, y=portfolio_values,
    mode="lines", line=dict(color="royalblue", width=1.5),
    name="HMM Strategy",
))

# Buy & hold line for comparison
bh_values = 10_000 * (prices / prices[0])
fig.add_trace(go.Scatter(
    x=times, y=bh_values,
    mode="lines", line=dict(color="orange", width=1, dash="dash"),
    name="Buy & Hold",
))

fig.add_hline(y=10_000, line=dict(color="gray", dash="dot"), annotation_text="$10,000")
fig.update_layout(
    title=f"HMM Strategy vs Buy & Hold — Last 30 Days ({df.index[0].date()} to {df.index[-1].date()})",
    xaxis_title="Date", yaxis_title="Portfolio Value (USD)",
    height=500, legend=dict(x=0.01, y=0.99),
)
pio.write_html(fig, OUTPUT, include_plotlyjs="cdn")
print(f"Saved to {OUTPUT}")
