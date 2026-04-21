import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/drive_level_data.csv")
df = df[df["points_on_drive"].between(-2, 8)]

games = df["game_id"].unique()
np.random.seed(42)
test_games = np.random.choice(games, size=int(0.2 * len(games)), replace=False)

train_df = df[~df["game_id"].isin(test_games)].reset_index(drop=True)
test_df = df[df["game_id"].isin(test_games)].reset_index(drop=True)

features = [
"start_field_position",
"start_quarter",
"start_down",
"start_distance",
"offense_epa_per_play",
"defense_epa_allowed"
]

train_off = pd.get_dummies(train_df["offense_team"], prefix="off", drop_first=True)
train_def = pd.get_dummies(train_df["defense_team"], prefix="def", drop_first=True)

test_off = pd.get_dummies(test_df["offense_team"], prefix="off", drop_first=True)
test_def = pd.get_dummies(test_df["defense_team"], prefix="def", drop_first=True)

train_team = pd.concat([train_off, train_def], axis=1)
test_team = pd.concat([test_off, test_def], axis=1)

test_team = test_team.reindex(columns=train_team.columns, fill_value=0)

X_train = pd.concat([train_df[features], train_team], axis=1)
X_test = pd.concat([test_df[features], test_team], axis=1)

y_train = train_df["points_on_drive"].values
y_test = test_df["points_on_drive"].values

scaler = StandardScaler()

X_train[features] = scaler.fit_transform(X_train[features])
X_test[features] = scaler.transform(X_test[features])

rf = RandomForestRegressor(
n_estimators=500,
max_depth=None,
min_samples_leaf=5,
min_samples_split=10,
max_features="sqrt",
bootstrap=True,
random_state=42,
n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Performance (Test Set)")
print("-------------------------------------")
print("RMSE:", rmse)
print("MAE:", mae)
print("R^2:", r2)

importances = pd.Series(
rf.feature_importances_,
index=X_train.columns
).sort_values(ascending=False)

print("\nTop 15 Important Features")
print(importances.head(15))