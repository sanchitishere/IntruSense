import pandas as pd

df = pd.read_csv("data/cicids2017_cleaned.csv", low_memory=False)
df.columns = [c.strip() for c in df.columns]

sample = df.groupby("Attack Type", group_keys=False).apply(
    lambda x: x.sample(frac=0.1, random_state=42)
)
sample.to_csv("data/cicids_sample.csv", index=False)
print(f"Sampled {len(sample)} rows out of {len(df)}")