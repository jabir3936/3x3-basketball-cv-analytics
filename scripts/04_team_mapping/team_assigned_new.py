import pandas as pd
import numpy as np

AUTO_CSV = "data/team_mapping/team_mapping_auto.csv"
MANUAL_CSV = "data/team_mapping/team_mapping.csv"
FULL_CSV = "data/team_mapping/team_mapping_full.csv"

df = pd.read_csv(AUTO_CSV)
df["frames"] = df["end_frame"] - df["start_frame"] + 1
X = df[["mean_S", "mean_V"]].values.astype(float)

# k-means, 3 clusters: white jersey / blue jersey / dark (referee)
rng = np.random.default_rng(42)
cent = X[rng.choice(len(X), 3, replace=False)].copy()
for _ in range(100):
    d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(axis=1)
    for k in range(3):
        if (lab == k).any():
            cent[k] = X[lab == k].mean(axis=0)

order = np.argsort(cent[:, 1])          # sort clusters by brightness V
names = {order[2]: "team_white", order[1]: "team_blue", order[0]: "unknown"}

print("Cluster centroids (S, V):")
for k in range(3):
    print(f"  S={cent[k][0]:6.1f}  V={cent[k][1]:6.1f}  -> {names[k]}")

# If the 'darkest' cluster isn't actually dark, it's not referees - merge into blue
if cent[order[0]][1] > 90:
    names[order[0]] = "team_blue"
    print("  (no truly dark cluster found - treated as blue)")

df["team_id"] = [names[l] for l in lab]

print("\nFrame balance after k-means:")
print(df.groupby("team_id")["frames"].sum())

df.to_csv(AUTO_CSV, index=False)
keep = ["track_id", "start_frame", "end_frame", "team_id"]
m = pd.read_csv(MANUAL_CSV)[keep]
pd.concat([m, df[keep]], ignore_index=True).to_csv(FULL_CSV, index=False)
print(f"\nRewrote {AUTO_CSV} and {FULL_CSV}")