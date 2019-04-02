import statsmodels as sm
import pandas as pd
import numpy as np

def load_data():
    df_read = sm.datasets.nile.load_pandas().data
    s_date = pd.Series(
        [pd.to_datetime(str(int(y_str))) for y_str in df_read['year']]
    )
    df = df_read.set_index(s_date)
    df = df.drop('year', axis=1)

    return df

df_nile = load_data()

df_nile['lag1'] = df_nile['volume'].shift(1) 
df_nile['lag2'] = df_nile['volume'].shift(2)
df_nile['lag3'] = df_nile['volume'].shift(3)

df_nile = df_nile.dropna()

X_train = df_nile[['lag1', 'lag2', 'lag3']][:67].values
X_test = df_nile[['lag1', 'lag2', 'lag3']][67:].values

y_train = df_nile['volume'][:67].values
y_test = df_nile['volume'][67:].values

def step_forward(rf_fitted, X, in_out_sep=67, pred_len=33):
    # predict step by step
    nlags = 3
    idx = in_out_sep - nlags - 1

    lags123 = np.asarray([X[idx, 0],
                          X[idx, 1],
                          X[idx, 2]])

    x_pred_hist = []
    for i in range(nlags + pred_len):
        x_pred = rf_fitted.predict([lags123])
        if i > nlags:
            x_pred_hist.append(x_pred)
        lags123[0] = lags123[1]
        lags123[1] = lags123[2]
        lags123[2] = x_pred

    x_pred_np = np.asarray(x_pred_hist).squeeze()

    return x_pred_np


from sklearn.ensemble import RandomForestRegressor
r_forest = RandomForestRegressor(
            n_estimators=100,
            criterion='mse',
            random_state=1,
            n_jobs=-1
)
r_forest.fit(X_train, y_train)
x_pred_np = step_forward(r_forest, X_test)


