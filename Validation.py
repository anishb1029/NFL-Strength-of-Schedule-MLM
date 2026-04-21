import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel

df = pd.read_csv("data/drive_level_data.csv")

df = df[df["points_on_drive"].between(-2, 8)]
df["points_on_drive"] = df["points_on_drive"].astype(int)

features = [
    "start_field_position",
    "start_quarter",
    "start_down",
    "start_distance"
]

df[features] = (
    df[features] - df[features].mean()
) / df[features].std()

train_df = df[df["start_quarter"] <= 2]
test_df = df[df["start_quarter"] > 2]

X_train = train_df[features]
y_train = train_df["points_on_drive"]

X_test = test_df[features]
y_test = test_df["points_on_drive"]

model = OrderedModel(
    endog=y_train.reset_index(drop=True),
    exog=X_train.reset_index(drop=True),
    distr="logit"
)

res = model.fit(method="bfgs", disp=False)

pred_probs = res.model.predict(
    res.params,
    exog=X_test.reset_index(drop=True)
)

point_values = np.sort(y_train.unique())
epd = pred_probs @ point_values

print("Mean EPD (out-of-sample):", epd.mean())
print("Actual mean points:", y_test.mean())

coef_stability = []

for lam in [0.0, 0.25, 0.5, 1.0]:
    X_scaled = X_train / (1 + lam)
    m = OrderedModel(
        endog=y_train.reset_index(drop=True),
        exog=X_scaled.reset_index(drop=True),
        distr="logit"
    )
    r = m.fit(method="bfgs", disp=False)
    coef_stability.append(r.params[:len(features)].values)

coef_df = pd.DataFrame(coef_stability, columns=features)
print("Coefficient sensitivity to shrinkage:")
print(coef_df.std())
