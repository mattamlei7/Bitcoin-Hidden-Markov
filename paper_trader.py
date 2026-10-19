import numpy as np
import pandas as pd
import joblib
import requests
import pandas_ta as ta
import time
import os
from datetime import datetime

TRADES_FILE   = "C:/Users/matth/crypto_hmm_trader/paper_trades.csv"
STARTING_CASH = 10_000.0
BUY_STATE     = 0
SELL_STATE    = 2
POSITION_SIZE = 0.95   # fraction of cash to deploy on buy

# ── Load model and scaler once ───────────────────────────────────────────────
print("Loading BTC model and scaler...")
scaler = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/scaler.joblib")
model  = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/model.joblib")

# ── Portfolio state ──────────────────────────────────────────────────────────
cash        = STARTING_CASH
btc_held    = 0.0
entry_price = 0.0
in_position = False
trade_count = 0

# Initialise CSV if it doesn't exist
if not os.path.exists(TRADES_FILE):
    pd.DataFrame(columns=["timestamp", "action", "price", "state",
                           "btc_amount", "cash_after", "portfolio_value"]
                 ).to_csv(TRADES_FILE, index=False)
    print(f"Created {TRADES_FILE}")
else:
    print(f"Appending to existing {TRADES_FILE}")

def fetch_candles():
    url    = "https://api.binance.us/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 100}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    df  = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("datetime", inplace=True)
    df = df[["open","high","low","close","volume"]].astype(float)
    return df

def calculate_features(df):
    df = df.copy()
    df["volume_pct_change"] = df["volume"].pct_change()
    df["returns"]           = df["close"].pct_change()
    df["volatility"]        = df["returns"].rolling(20).std()
    bbands = ta.bbands(df["close"], length=20)
    bb_col = [c for c in bbands.columns if c.startswith("BBB_")][0]
    df["bb_width"] = bbands[bb_col]
    df.drop(columns=["returns"], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def log_trade(timestamp, action, price, state, btc_amount, cash_after, portfolio_value):
    row = pd.DataFrame([{
        "timestamp":       timestamp,
        "action":          action,
        "price":           round(price, 2),
        "state":           state,
        "btc_amount":      round(btc_amount, 6),
        "cash_after":      round(cash_after, 2),
        "portfolio_value": round(portfolio_value, 2)
    }])
    row.to_csv(TRADES_FILE, mode="a", header=False, index=False)

def portfolio_value(price):
    return cash + btc_held * price

# ── Main loop ────────────────────────────────────────────────────────────────
print(f"\nPaper trader started. Portfolio: ${STARTING_CASH:,.2f}")
print(f"Strategy: BUY on state {BUY_STATE}  |  SELL on state {SELL_STATE}")
print(f"Checking every hour. Press Ctrl+C to stop.\n")

while True:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        df      = fetch_candles()
        df      = calculate_features(df)
        X       = scaler.transform(df[["volume_pct_change","volatility","bb_width"]].values)
        states  = model.predict(X)
        state   = int(states[-1])
        price   = float(df["close"].iloc[-1])
        pv      = portfolio_value(price)

        action = "HOLD"

        if state == BUY_STATE and not in_position:
            btc_bought  = (cash * POSITION_SIZE) / price
            cash       -= btc_bought * price
            btc_held    = btc_bought
            entry_price = price
            in_position = True
            trade_count += 1
            action      = "BUY"
            pv          = portfolio_value(price)
            log_trade(now, "BUY", price, state, btc_bought, cash, pv)
            print(f"[{now}]  ✅ BUY   {btc_bought:.6f} BTC @ ${price:,.2f}  |  Cash: ${cash:,.2f}")

        elif state == SELL_STATE and in_position:
            proceeds    = btc_held * price
            pnl         = proceeds - (btc_held * entry_price)
            cash       += proceeds
            action      = "SELL"
            pv          = cash
            log_trade(now, "SELL", price, state, btc_held, cash, pv)
            print(f"[{now}]  ❌ SELL  {btc_held:.6f} BTC @ ${price:,.2f}  |  P&L: ${pnl:+,.2f}  |  Cash: ${cash:,.2f}")
            btc_held    = 0.0
            in_position = False
            trade_count += 1

        else:
            print(f"[{now}]  ⏸  HOLD  State={state}  Price=${price:,.2f}  "
                  f"Portfolio=${pv:,.2f}  Position={'IN' if in_position else 'OUT'}")

    except requests.exceptions.RequestException as e:
        print(f"[{now}]  ⚠️  API error: {e}  — skipping this hour")
    except Exception as e:
        print(f"[{now}]  ⚠️  Error: {e}  — skipping this hour")

    time.sleep(3600)
