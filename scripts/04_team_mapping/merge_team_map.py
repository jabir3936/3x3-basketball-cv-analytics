import pandas as pd
keep = ["track_id","start_frame","end_frame","team_id"]
m = pd.read_csv("data/team_mapping/team_mapping.csv")[keep]
a = pd.read_csv("data/team_mapping/team_mapping_auto.csv")[keep]
pd.concat([m, a], ignore_index=True).to_csv("data/team_mapping/team_mapping_full.csv", index=False)