import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# Load and clean data
df = pd.read_csv("C:/Users/matth/crypto_hmm_trader/btc_features.csv", index_col="datetime", parse_dates=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

features = ["volume_pct_change", "volatility", "bb_width"]
X = df[features].values
n_samples, n_features = X.shape

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Training HMMs with 2-15 states on {n_samples} samples...\n")

results = []

for n in range(2, 16):
    model = GaussianHMM(
        n_components=n,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    model.fit(X_scaled)

    log_likelihood = model.score(X_scaled)
    total_ll = log_likelihood * n_samples

    # Number of free parameters for BIC
    # transition matrix: n*(n-1), start probs: n-1, means: n*f, full cov: n*f*(f+1)/2
    n_params = (n * (n - 1)) + (n - 1) + (n * n_features) + (n * n_features * (n_features + 1) // 2)
    bic = -2 * total_ll + n_params * np.log(n_samples)
    aic = -2 * total_ll + 2 * n_params

    # Average confidence: mean of max posterior probability per observation
    posteriors = model.predict_proba(X_scaled)
    avg_confidence = posteriors.max(axis=1).mean()

    results.append({
        "states": n,
        "log_likelihood": round(log_likelihood, 4),
        "bic": round(bic, 1),
        "aic": round(aic, 1),
        "avg_confidence": round(avg_confidence, 4)
    })

    print(f"  n={n:2d} | LL: {log_likelihood:9.4f} | BIC: {bic:15.1f} | AIC: {aic:15.1f} | Confidence: {avg_confidence:.4f}")

# Find best models
best_bic = min(results, key=lambda r: r["bic"])
best_aic = min(results, key=lambda r: r["aic"])
best_conf = max(results, key=lambda r: r["avg_confidence"])

print("\n" + "="*75)
print(f"{'States':<8} {'Log-Likelihood':<16} {'BIC':<18} {'AIC':<18} {'Confidence'}")
print("-"*75)
for r in results:
    bic_flag  = " <-- best BIC" if r["states"] == best_bic["states"] else ""
    aic_flag  = " <-- best AIC" if r["states"] == best_aic["states"] else ""
    flag = bic_flag or aic_flag
    print(f"{r['states']:<8} {r['log_likelihood']:<16} {r['bic']:<18} {r['aic']:<18} {r['avg_confidence']}{flag}")

print("="*75)
print(f"\nBest by BIC:        {best_bic['states']} states  (BIC={best_bic['bic']})")
print(f"Best by AIC:        {best_aic['states']} states  (AIC={best_aic['aic']})")
print(f"Best by Confidence: {best_conf['states']} states  (Confidence={best_conf['avg_confidence']})")

# Recommendation: prefer BIC (penalises complexity more)
print(f"\nRecommendation: {best_bic['states']} states (BIC is the most reliable for HMMs — penalises overfitting more than AIC)")
