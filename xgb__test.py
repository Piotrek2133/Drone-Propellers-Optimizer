"""Tryout of (e)X(treme)G(radient)B(oosting) approach: 
results: +----------+-----------+
| Metric   |     Value |
+==========+===========+
| mae      | 0.0285186 |
+----------+-----------+
| mse      | 0.0168025 |
+----------+-----------+
| R²       | 0.92084   |

    """


import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


# Train XGBoost model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Calculate evaluation metrics
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Display results
data_xgb = [['mae', mae_xgb], ['mse', mse_xgb], ['R²', r2_xgb]]
header_xgb = ['Metric', 'Value']
from tabulate import tabulate
print(tabulate(data_xgb, header_xgb, tablefmt='grid'))
