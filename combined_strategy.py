import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from backtesting import Backtest, Strategy

# ── 1. Load and clean ────────────────────────────────────────────────────────
print("Step 1: Loading btc_features.csv...")
df = pd.read_csv("C:/Users/matth/crypto_hmm_trader/btc_features.csv",
                 index_col="datetime", parse_dates=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"  {len(df)} rows loaded")

# ── 2. Predict HMM states ────────────────────────────────────────────────────
print("Step 2: Predicting HMM states...")
scaler = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/scaler.joblib")
model  = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/model.joblib")
X_scaled = scaler.transform(df[["volume_pct_change", "volatility", "bb_width"]].values)
df["state"] = model.predict(X_scaled)
print(f"  State distribution:\n{df['state'].value_counts().sort_index().to_string()}")

# ── 3. Target: direction (1 if next close > current close) ───────────────────
print("\nStep 3: Creating direction target...")
df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

# ── 4. XGBoost features ──────────────────────────────────────────────────────
print("Step 4: Building XGBoost features...")
df["returns"] = df["close"].pct_change()
df["returns_mean_5"]        = df["returns"].rolling(5).mean()
df["returns_mean_20"]       = df["returns"].rolling(20).mean()
df["volume_pct_mean_5"]     = df["volume_pct_change"].rolling(5).mean()
# state is already a column — included directly as a feature

df.dropna(inplace=True)
print(f"  {len(df)} rows after dropping NaN")

# ── 5. Train / test split (80 / 20) ─────────────────────────────────────────
print("\nStep 5: Splitting 80/20 train/test...")
split = int(len(df) * 0.8)
xgb_features = ["returns_mean_5", "returns_mean_20", "volume_pct_mean_5",
                 "volatility", "bb_width", "volume_pct_change", "state"]

X_all   = df[xgb_features].values
y_all   = df["direction"].values
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]
print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

# ── 6. Train XGBoost ─────────────────────────────────────────────────────────
print("\nStep 6: Training XGBoost classifier...")
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                    random_state=42, eval_metric="logloss", verbosity=0)
xgb.fit(X_train, y_train)

test_preds = xgb.predict(X_test)
acc = accuracy_score(y_test, test_preds)
print(f"  Test accuracy: {acc:.4f} ({acc*100:.2f}%)")

# ── 7. Feature importance + combined signal ──────────────────────────────────
print("\nStep 7: XGBoost feature importance...")
importance = pd.Series(xgb.feature_importances_, index=xgb_features).sort_values(ascending=False)
for feat, score in importance.items():
    print(f"  {feat:<30} {score:.4f}")

print("\nBuilding combined signal (3-period majority vote)...")
all_preds = xgb.predict(X_all)
df["xgb_pred"] = all_preds
# Majority vote: up if at least 2 of last 3 predictions are 1
df["xgb_vote"] = df["xgb_pred"].rolling(3, min_periods=3).sum() >= 2
df["combined_signal"] = ((df["state"] == 0) & (df["xgb_vote"])).astype(int)
df.dropna(inplace=True)
print(f"  Signal=1 (buy):  {df['combined_signal'].sum()} rows")
print(f"  Signal=0 (cash): {(df['combined_signal'] == 0).sum()} rows")

# ── 8. Backtest ──────────────────────────────────────────────────────────────
print("\nStep 8: Running backtest...")
bt_df = df.rename(columns={"open": "Open", "high": "High",
                            "low": "Low",  "close": "Close", "volume": "Volume"})

MIN_HOLD_BARS = 24  # minimum 24 hourly bars = 1 day

class CombinedStrategy(Strategy):
    def init(self):
        self.signal     = self.I(lambda x: x, self.data.combined_signal, name="signal", plot=False)
        self.entry_bar  = 0

    def next(self):
        sig = self.signal[-1]
        bars_held = len(self.data) - self.entry_bar

        if sig == 1 and not self.position:
            units = int(self.equity * 0.95 / self.data.Close[-1])
            if units > 0:
                self.buy(size=units)
                self.entry_bar = len(self.data)
        elif sig == 0 and self.position and bars_held >= MIN_HOLD_BARS:
            self.position.close()

bt    = Backtest(bt_df, CombinedStrategy, cash=10_000,
                 commission=0.002, exclusive_orders=True)
stats = bt.run()

# ── 9. Print results ─────────────────────────────────────────────────────────
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

# ── 10. Save equity curve ────────────────────────────────────────────────────
print("\nSaving equity curve...")
equity = stats["_equity_curve"]["Equity"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines",
                         line=dict(color="royalblue", width=1.5), name="Combined Strategy"))
fig.add_hline(y=10_000, line=dict(color="gray", dash="dash"),
              annotation_text="Starting Capital $10,000")
fig.update_layout(title="HMM + XGBoost Combined Strategy Equity Curve",
                  xaxis_title="Date", yaxis_title="Portfolio Value (USD)", height=500)
pio.write_html(fig, "C:/Users/matth/crypto_hmm_trader/combined_equity_curve.html",
               include_plotlyjs="cdn")
print("Saved to combined_equity_curve.html")
