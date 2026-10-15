"""Creates a small test CSV (no TARGET column) from application_train.csv for trying the app."""
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "samples"
out.mkdir(exist_ok=True)
df = pd.read_csv(root / "datasets" / "application_train.csv").sample(200, random_state=7)
df.drop(columns=["TARGET"]).to_csv(out / "sample_applicants.csv", index=False)
print("Created samples/sample_applicants.csv with", len(df), "rows")