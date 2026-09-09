import pandas as pd
import numpy as np

AUTO_CSV = "data/team_mapping/team_mapping_auto.csv"
MANUAL_CSV = "data/team_mapping/team_mapping.csv"
FULL_CSV = "data/team_mapping/team_mapping_full.csv"

df = pd.read_csv(AUTO_CSV)
df["frames"] = df["end_frame"] - df["start_frame"] + 1

# Show the two color clusters so we can verify separation
for col in ["mean_S", "mean_V"]:
    hist, edges = np.histogram(df[col], bins=10)
    print(f"\n{col} histogram:")
    for h, e in zip(hist, edges):
        print(f"  {e:6.1f}: {'#' * int(h / 2)} ({h})")

# NEW RULE: white = pale (low saturation) + not dark
#           blue  = saturated jersey
#           unknown = black referee / very dark
def classify(row):
    S, V = row["mean_S"], row["mean_V"]
    if V < 50:
        return "unknown"
    if S < 80 and V >= 100:
        return "team_white"
    return "team_blue"

df["team_id"] = df.apply(classify, axis=1)

print("\nAuto-labeled frame balance (should be roughly equal):")
print(df.groupby("team_id")["frames"].sum())

df.to_csv(AUTO_CSV, index=False)

keep = ["track_id", "start_frame", "end_frame", "team_id"]
m = pd.read_csv(MANUAL_CSV)[keep]
pd.concat([m, df[keep]], ignore_index=True).to_csv(FULL_CSV, index=False)
print(f"\nRewrote {AUTO_CSV} and {FULL_CSV}")