"""Generate small, public, imbalanced sample datasets for the demo (no real personal data)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer

OUT = Path(__file__).resolve().parent.parent / "samples"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(7)

# 1) transactions_fraud.csv  - numeric, ~2% fraud
X, y = make_classification(n_samples=3000, n_features=8, n_informative=5, n_redundant=1,
                           weights=[0.98], flip_y=0.01, random_state=7)
cols = ["amount", "hour", "dist_from_home", "n_recent_tx", "avg_amount",
        "account_age_days", "n_countries_7d", "velocity_score"]
df = pd.DataFrame(X, columns=cols)
df["amount"] = (df["amount"] * 120 + 160).round(2).clip(lower=0)
df["hour"] = (((df["hour"] * 4 + 12) % 24).round()).astype(int).clip(0, 23)
df["account_age_days"] = (df["account_age_days"] * 300 + 800).round().clip(lower=1).astype(int)
df["n_recent_tx"] = (df["n_recent_tx"] * 3 + 6).round().clip(lower=0).astype(int)
df["n_countries_7d"] = (df["n_countries_7d"] * 1.2 + 1).round().clip(lower=1).astype(int)
df["is_fraud"] = y
df.to_csv(OUT / "transactions_fraud.csv", index=False)

# 2) medical_screening.csv  - numeric, malignant made rare (~8%)
bc = load_breast_cancer(as_frame=True).frame
mal = bc[bc.target == 0].sample(30, random_state=7)      # 0 = malignant in sklearn
ben = bc[bc.target == 1]                                  # 1 = benign
m = pd.concat([mal, ben]).sample(frac=1, random_state=7).reset_index(drop=True)
m = m.rename(columns={"target": "diagnosis"})
m["diagnosis"] = m["diagnosis"].map({0: "malignant", 1: "benign"})
m.columns = [c.replace(" ", "_") for c in m.columns]
m.to_csv(OUT / "medical_screening.csv", index=False)

# 3) customer_churn_mixed.csv  - MIXED types (categoricals), ~11% churn
n = 2500
region = rng.choice(["North", "South", "East", "West"], n)
contract = rng.choice(["monthly", "yearly", "two_year"], n, p=[0.6, 0.3, 0.1])
tenure = rng.randint(0, 72, n)
monthly = rng.uniform(20, 120, n).round(2)
support = rng.randint(0, 6, n)
base = (0.05 + 0.16 * (contract == "monthly") + 0.10 * (tenure < 6)
        - 0.05 * (contract == "two_year") + 0.02 * support)
churn = (rng.uniform(size=n) < base).astype(int)
pd.DataFrame({"region": region, "contract": contract, "tenure_months": tenure,
              "monthly_charge": monthly, "support_calls": support,
              "churned": churn}).to_csv(OUT / "customer_churn_mixed.csv", index=False)

for f in sorted(OUT.glob("*.csv")):
    d = pd.read_csv(f)
    tgt = d.columns[-1]
    vc = d[tgt].value_counts()
    print(f"{f.name:28} {len(d):5} rows  target={tgt:12} dist={dict(vc)}")
