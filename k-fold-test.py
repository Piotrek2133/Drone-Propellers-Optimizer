"""Tryout of (Kfolds approach: 
results: +----------+-----------+
+------------+-----------+
| Metric     |     Value |
+============+===========+
| MAE (Mean) | 0.0282916 |
+------------+-----------+
| MSE (Mean) | 0.0183524 |
+------------+-----------+
| R² (Mean)  | 0.919707  |
+------------+-----------+

    """


import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
input_keys = ['B', 'D', 'P', 'J', 'N', 'c/R', 'r/R', 'beta']
output_keys = ['CT', 'CP', 'eta']

# Create geometric dataframe
path = ''
# Directories of performance data
geom1_dir = os.path.join(path, 'volume1_geom.csv')
geom2_dir = os.path.join(path, 'volume2_geom.csv')
geom3_dir = os.path.join(path, 'volume3_geom.csv')

# Geometric data from volume 1 to volume 3
geom1_df = pd.read_csv(geom1_dir)
geom2_df = pd.read_csv(geom2_dir)
geom3_df = pd.read_csv(geom3_dir)

# Merge them into 1 geom dataframe
geom_df = pd.concat([geom1_df, geom2_df, geom3_df], ignore_index=True)

# Create performance dataframe
# Directories of performance data
perf1_dir = os.path.join(path, 'volume1_exp.csv')
perf2_dir = os.path.join(path, 'volume2_exp.csv')
perf3_dir = os.path.join(path, 'volume3_exp.csv')

# Performance data from volume 1 to volume 3
perf1_df = pd.read_csv(perf1_dir)
perf2_df = pd.read_csv(perf2_dir)
perf3_df = pd.read_csv(perf3_dir)

# Merge them into 1 perf dataframe
perf_df = pd.concat([perf1_df, perf2_df, perf3_df], ignore_index=True)

# Create df dataframe
df = perf_df.merge(geom_df, on=['BladeName', 'D', 'P', 'Family'])

df.to_csv("full_data.csv", sep = ',', index=False,encoding='utf-8')

# Split df into X and y
X = df[input_keys].values
y = df[output_keys].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5 folds

# Initialize XGBoost model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42 
)

mae_scores = []
mse_scores = []
r2_scores = []

for train_index, test_index in kf.split(X):
    X_train_kf, X_test_kf = X[train_index], X[test_index]
    y_train_kf, y_test_kf = y[train_index], y[test_index]

    xgb_model.fit(X_train_kf, y_train_kf)
    y_pred_kf = xgb_model.predict(X_test_kf)

    mae_scores.append(mean_absolute_error(y_test_kf, y_pred_kf))
    mse_scores.append(mean_squared_error(y_test_kf, y_pred_kf))
    r2_scores.append(r2_score(y_test_kf, y_pred_kf))

mae_mean = np.mean(mae_scores)
mse_mean = np.mean(mse_scores)
r2_mean = np.mean(r2_scores)

data_kf = [
    ['MAE (Mean)', mae_mean],
    ['MSE (Mean)', mse_mean],
    ['R² (Mean)', r2_mean]
]
header_kf = ['Metric', 'Value']
from tabulate import tabulate
print(tabulate(data_kf, header_kf, tablefmt='grid'))
