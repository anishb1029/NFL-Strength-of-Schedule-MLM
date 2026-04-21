import pandas as pd

pbp = pd.read_parquet("play_by_play_2023.parquet")

pbp = pbp[
    (pbp["play_type"].isin(["run", "pass"])) &
    (pbp["drive"].notna())
]

pbp["score_diff_start"] = pbp.groupby(
    ["game_id", "posteam", "drive"]
)["score_differential"].transform("first")

pbp["score_diff_end"] = pbp.groupby(
    ["game_id", "posteam", "drive"]
)["score_differential"].transform("last")

pbp["points_on_drive"] = pbp["score_diff_end"] - pbp["score_diff_start"]

drives = (
    pbp
    .groupby(["game_id", "posteam", "defteam", "drive"], as_index=False)
    .agg(
        start_field_position=("yardline_100", "first"),
        start_quarter=("qtr", "first"),
        start_down=("down", "first"),
        start_distance=("ydstogo", "first"),
        points_on_drive=("points_on_drive", "max")
    )
)

drives.rename(columns={
    "posteam": "offense_team",
    "defteam": "defense_team"
}, inplace=True)

off_epa = pbp.groupby("posteam")["epa"].mean()
def_epa = pbp.groupby("defteam")["epa"].mean()

drives["offense_epa_per_play"] = drives["offense_team"].map(off_epa)
drives["defense_epa_allowed"] = drives["defense_team"].map(def_epa)

drives["home_offense"] = 0
drives["indoor_stadium"] = 0
drives["injuries_count"] = 0
drives["wind_speed"] = 5
drives["temperature"] = 60

drives.to_csv("data/drive_level_data.csv", index=False)
