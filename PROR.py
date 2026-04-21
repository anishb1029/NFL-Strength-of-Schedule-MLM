import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

df = pd.read_csv("data/drive_level_data.csv")
df = df[df["points_on_drive"].between(-2, 8)]
df["points_on_drive"] = df["points_on_drive"].astype(int)


np.random.seed(42)
games = df["game_id"].unique()
test_games = np.random.choice(games, size=int(0.2 * len(games)), replace=False)

df_train = df[~df["game_id"].isin(test_games)].reset_index(drop=True)
df_test = df[df["game_id"].isin(test_games)].reset_index(drop=True)

y_train = df_train["points_on_drive"].values
y_test = df_test["points_on_drive"].values

context_features = [
    "start_field_position",
    "start_quarter",
    "start_down",
    "start_distance",
    "offense_epa_per_play",
    "defense_epa_allowed"
]

Xc_train = df_train[context_features].astype(float)
Xc_test = df_test[context_features].astype(float)

mu = Xc_train.mean()
sd = Xc_train.std()

Xc_train = (Xc_train - mu) / sd
Xc_test = (Xc_test - mu) / sd

off_dummies = pd.get_dummies(df_train["offense_team"], prefix="off", drop_first=True)
def_dummies = pd.get_dummies(df_train["defense_team"], prefix="def", drop_first=True)

Xt_train = pd.concat([off_dummies, def_dummies], axis=1)

Xt_test = pd.concat([
    pd.get_dummies(df_test["offense_team"], prefix="off", drop_first=True),
    pd.get_dummies(df_test["defense_team"], prefix="def", drop_first=True)
], axis=1)

Xt_test = Xt_test.reindex(columns=Xt_train.columns, fill_value=0)

X_train = np.hstack([Xc_train.values, Xt_train.values])
X_test = np.hstack([Xc_test.values, Xt_test.values])

n_context = Xc_train.shape[1]
n_team = Xt_train.shape[1]

categories = np.sort(np.unique(y_train))
category_index = {v: i for i, v in enumerate(categories)}

K = len(categories)
p = X_train.shape[1]

def neg_penalized_loglik(theta, X, y, lam):

    beta = theta[:p]
    c0 = theta[p]
    raw_delta = theta[p+1:]

    delta = np.exp(raw_delta)
    cuts = np.concatenate([[c0], c0 + np.cumsum(delta)])

    eta = X @ beta
    ll = 0.0

    for i in range(len(y)):

        k = category_index[y[i]]

        if k == 0:
            p_i = expit(cuts[0] - eta[i])
        elif k == K - 1:
            p_i = 1 - expit(cuts[-1] - eta[i])
        else:
            p_i = expit(cuts[k] - eta[i]) - expit(cuts[k - 1] - eta[i])

        p_i = np.clip(p_i, 1e-12, 1)
        ll += np.log(p_i)

    beta_team = beta[n_context:]
    penalty = lam * np.sum(beta_team ** 2)

    return -(ll - penalty)

beta_init = np.zeros(p)
c0_init = -1.0
raw_delta_init = np.zeros(K - 2)

theta_init = np.concatenate([beta_init, [c0_init], raw_delta_init])

lambda_grid = [0, 0.1, 0.25, 0.5, 1, 2, 5]

best_result = None
best_lambda = None
best_ll = -np.inf

for lam in lambda_grid:

    res = minimize(
        neg_penalized_loglik,
        theta_init,
        args=(X_train, y_train, lam),
        method="L-BFGS-B"
    )

    if not res.success:
        continue

    ll = -neg_penalized_loglik(res.x, X_test, y_test, lam)

    if ll > best_ll:
        best_ll = ll
        best_lambda = lam
        best_result = res

print("Selected lambda:", best_lambda)
print("Converged:", best_result.success)

result = best_result

beta_hat = result.x[:p]
c0_hat = result.x[p]
raw_delta_hat = result.x[p+1:]

delta_hat = np.exp(raw_delta_hat)
cuts_hat = np.concatenate([[c0_hat], c0_hat + np.cumsum(delta_hat)])

def expected_points(X, beta, cuts):

    eta = X @ beta
    ep = np.zeros(len(eta))

    for i in range(len(eta)):

        probs = []

        for k in range(K):

            if k == 0:
                p_i = expit(cuts[0] - eta[i])
            elif k == K - 1:
                p_i = 1 - expit(cuts[-1] - eta[i])
            else:
                p_i = expit(cuts[k] - eta[i]) - expit(cuts[k - 1] - eta[i])

            probs.append(p_i)

        ep[i] = np.sum(np.array(probs) * categories)

    return ep

ep_test = expected_points(X_test, beta_hat, cuts_hat)

print("Mean expected points per drive (test):", ep_test.mean())