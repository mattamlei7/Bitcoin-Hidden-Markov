import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# Load features
df = pd.read_csv(
    "C:/Users/matth/crypto_hmm_trader/btc_features.csv",
    index_col="datetime",
    parse_dates=True
)

features = ["volume_pct_change", "volatility", "bb_width"]
X = df[features].values

# Replace infinities with NaN then drop
X = pd.DataFrame(X, columns=features)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
X = X.values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train HMM
model = GaussianHMM(
    n_components=5,
    covariance_type="full",
    n_iter=100,
    random_state=42
)
model.fit(X_scaled)

# State distribution check
states = model.predict(X_scaled)
counts = pd.Series(states).value_counts().sort_index()
print(f"Log Likelihood: {model.score(X_scaled):.4f}")
print(f"\nState distribution:")
for s, c in counts.items():
    flag = "  <-- WARNING: too few" if c < 1000 else ""
    print(f"  State {s}: {c:>6} rows{flag}")

# Save model and scaler
os.makedirs("C:/Users/matth/crypto_hmm_trader/trained_models", exist_ok=True)
joblib.dump(model, "C:/Users/matth/crypto_hmm_trader/trained_models/model.joblib")
joblib.dump(scaler, "C:/Users/matth/crypto_hmm_trader/trained_models/scaler.joblib")

print(f"\nTransition Matrix:")
print(np.round(model.transmat_, 4))
print("\nModel and scaler saved to trained_models/")
