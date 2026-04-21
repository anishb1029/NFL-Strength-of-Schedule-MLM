import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/drive_level_data.csv")

df = df[df["points_on_drive"].between(-2,8)]
df["points_on_drive"] = df["points_on_drive"].astype(int)

numeric_features = [
"start_field_position",
"start_quarter",
"start_down",
"start_distance",
"offense_epa_per_play",
"defense_epa_allowed",
"injuries_count",
"wind_speed",
"temperature"
]

categorical_features = [
"home_offense",
"indoor_stadium"
]

df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
df[categorical_features] = df[categorical_features].fillna(0)

strength_features = [
"offense_epa_per_play",
"defense_epa_allowed",
"start_field_position",
"start_down",
"start_distance"
]

df[strength_features] = (
df[strength_features] - df[strength_features].mean()
) / df[strength_features].std()

df["drive_strength_score"] = (
1.0 * df["offense_epa_per_play"]
+ 1.0 * df["defense_epa_allowed"]
+ 0.7 * df["start_field_position"]
- 0.6 * df["start_down"]
- 0.6 * df["start_distance"]
)

df["drive_strength_score"] = (
df["drive_strength_score"] - df["drive_strength_score"].mean()
) / df["drive_strength_score"].std()

np.random.seed(42)

games = df["game_id"].unique()

test_games = np.random.choice(games,size=int(0.2*len(games)),replace=False)
train_games = np.setdiff1d(games,test_games)
val_games = np.random.choice(train_games,size=int(0.2*len(train_games)),replace=False)
train_games_final = np.setdiff1d(train_games,val_games)

train_df = df[df["game_id"].isin(train_games_final)].reset_index(drop=True)
val_df = df[df["game_id"].isin(val_games)].reset_index(drop=True)
test_df = df[df["game_id"].isin(test_games)].reset_index(drop=True)

features = numeric_features + categorical_features + ["drive_strength_score"]

X_train = train_df[features]
X_val = val_df[features]
X_test = test_df[features]

y_train = train_df["points_on_drive"]
y_val = val_df["points_on_drive"]
y_test = test_df["points_on_drive"]

scaler = StandardScaler()

X_train[numeric_features + ["drive_strength_score"]] = scaler.fit_transform(
X_train[numeric_features + ["drive_strength_score"]]
)

X_val[numeric_features + ["drive_strength_score"]] = scaler.transform(
X_val[numeric_features + ["drive_strength_score"]]
)

X_test[numeric_features + ["drive_strength_score"]] = scaler.transform(
X_test[numeric_features + ["drive_strength_score"]]
)

X_train = pd.get_dummies(X_train,columns=categorical_features,drop_first=True)
X_val = pd.get_dummies(X_val,columns=categorical_features,drop_first=True)
X_test = pd.get_dummies(X_test,columns=categorical_features,drop_first=True)

X_val = X_val.reindex(columns=X_train.columns,fill_value=0)
X_test = X_test.reindex(columns=X_train.columns,fill_value=0)

model = OrderedModel(
endog=y_train.reset_index(drop=True),
exog=X_train.reset_index(drop=True),
distr="logit"
)

res = model.fit(method="bfgs",disp=False)

prob_test = res.model.predict(res.params,X_test)

point_values = np.sort(df["points_on_drive"].unique())

epd = prob_test @ point_values

rmse = np.sqrt(mean_squared_error(y_test,epd))
mae = mean_absolute_error(y_test,epd)

pred_round = np.clip(np.round(epd),-2,8)

exact_match = np.mean(pred_round == y_test)
within1 = np.mean(np.abs(pred_round - y_test) <= 1)

print("OBR RMSE:",rmse)
print("OBR MAE:",mae)
print("OBR Exact Match:",exact_match)
print("OBR Within 1:",within1)

coef_df = pd.DataFrame({
"feature": X_train.columns,
"coefficient": res.params[:len(X_train.columns)]
}).sort_values("coefficient",ascending=False)

print(coef_df)