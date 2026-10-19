print("Step 1: importing numpy")
import numpy as np

print("Step 2: importing pandas")
import pandas as pd

print("Step 3: importing joblib")
import joblib

print("Step 4: importing plotly")
import plotly.graph_objects as go
import plotly.io as pio

print("Step 5: loading btc_features.csv")
df = pd.read_csv("C:/Users/matth/crypto_hmm_trader/btc_features.csv", index_col="datetime", parse_dates=True)
print(f"  Loaded {len(df)} rows")

print("Step 6: cleaning features")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"  Clean rows: {len(df)}")

print("Step 7: loading scaler and model")
scaler = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/scaler.joblib")
model = joblib.load("C:/Users/matth/crypto_hmm_trader/trained_models/model.joblib")

print("Step 8: scaling and predicting states")
X_scaled = scaler.transform(df[["volume_pct_change", "volatility", "bb_width"]].values)
df["state"] = model.predict(X_scaled)
print(f"  States predicted, unique: {sorted(df['state'].unique())}")

print("Step 9: downsampling to weekly and smoothing states")
df_ds = df.iloc[::168].copy()  # one point per week
# Smooth states with rolling mode over 4 weeks to reduce flickering
df_ds["state"] = df_ds["state"].rolling(window=4, min_periods=1, center=True).apply(
    lambda x: pd.Series(x).mode()[0]
).astype(int)
print(f"  Downsampled to {len(df_ds)} rows")

print("Step 10: building chart")
colors = ["red", "blue", "green", "orange", "purple", "pink", "yellow"]
fig = go.Figure()

# Price line
fig.add_trace(go.Scatter(
    x=df_ds.index,
    y=df_ds["close"].values,
    mode="lines",
    line=dict(color="black", width=1),
    name="BTC Close"
))

# Group consecutive same-state rows into contiguous regions (one vrect each)
states_arr = df_ds["state"].values
index_arr = df_ds.index
regions = []
start = index_arr[0]
cur = states_arr[0]
for i in range(1, len(states_arr)):
    if states_arr[i] != cur:
        regions.append((start, index_arr[i], cur))
        start = index_arr[i]
        cur = states_arr[i]
regions.append((start, index_arr[-1], cur))
print(f"  Total regions: {len(regions)}")

for x0, x1, s in regions:
    fig.add_vrect(x0=x0, x1=x1, fillcolor=colors[s], opacity=0.3, layer="below", line_width=0)

# Add dummy scatter traces for legend
for s in range(7):
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color=colors[s], size=10),
        name=f"State {s}"
    ))

print("Step 11: saving chart")
pio.write_html(fig, "C:/Users/matth/crypto_hmm_trader/regime_chart.html", include_plotlyjs="cdn")

print("Chart saved")
