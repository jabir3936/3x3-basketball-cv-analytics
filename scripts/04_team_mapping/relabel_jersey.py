import pandas as pd

AUTO_CSV = "data/team_mapping/team_mapping_auto.csv"
MANUAL_CSV = "data/team_mapping/team_mapping.csv"
FULL_CSV = "data/team_mapping/team_mapping_full.csv"

df = pd.read_csv(AUTO_CSV)

def classify(row):
    S, V = row["mean_S"], row["mean_V"]
    if V < 50:
        return "unknown"        # very dark = referee
    if S < 80 and V >= 100:
        return "team_white"     # pale jersey
    return "team_blue"          # saturated dark jersey

df["team_id"] = df.apply(classify, axis=1)
df.to_csv(AUTO_CSV, index=False)

keep = ["track_id", "start_frame", "end_frame", "team_id"]
manual = pd.read_csv(MANUAL_CSV)[keep]
pd.concat([manual, df[keep]], ignore_index=True).to_csv(FULL_CSV, index=False)

print("Auto label counts:")
print(df["team_id"].value_counts())
print(f"Wrote {FULL_CSV}")