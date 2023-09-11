import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from joblib import dump

# Constants
SEED = 42
N_FOLDS = 5

# Load data
folder_path = '/path_to_your_csv_files/'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
data_frames = [pd.read_csv(folder_path + csv_file) for csv_file in csv_files]
data = pd.concat(data_frames, ignore_index=True)

# Scale and split data
scaler = StandardScaler()
data[['SOC', 'T', 'SOH']] = scaler.fit_transform(data[['SOC', 'T', 'SOH']])

X = data[['SOC', 'T']]
y = data['SOH']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Set up cross-validation
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# SVR model
svr_params = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 'scale']
}
svr_grid = GridSearchCV(SVR(kernel='rbf'), svr_params, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)
svr_grid.fit(X_train, y_train)

# Polynomial model
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('ridge', Ridge())
])
poly_params = {
    'poly__degree': [1, 2, 3],
    'ridge__alpha': [0.01, 0.1, 1, 10]
}
poly_grid = GridSearchCV(poly_pipeline, poly_params, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)
poly_grid.fit(X_train, y_train)

# XGBoost model
xgb_params = {
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [50, 100, 200]
}
xgb_grid = GridSearchCV(XGBRegressor(), xgb_params, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)
xgb_grid.fit(X_train, y_train)

# Visualize CV results
def heatmap_cv_results(grid, param1, param2):
    results = pd.DataFrame(grid.cv_results_)
    scores = results.mean_test_score.values.reshape(len(grid.param_grid[param1]), len(grid.param_grid[param2]))
    sns.heatmap(scores, annot=True, fmt=".3f", xticklabels=grid.param_grid[param1], yticklabels=grid.param_grid[param2])
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.show()

heatmap_cv_results(svr_grid, 'C', 'gamma')
heatmap_cv_results(poly_grid, 'poly__degree', 'ridge__alpha')
heatmap_cv_results(xgb_grid, 'learning_rate', 'n_estimators')

# Model evaluation
models = {'SVR': svr_grid.best_estimator_, 'Polynomial': poly_grid.best_estimator_, 'XGBoost': xgb_grid.best_estimator_}
mse_scores, rmse_scores, mae_scores, r2_scores = [], [], [], []

for name, model in models.items():
    y_pred = model.predict(X_test)
    mse_scores.append(mean_squared_error(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots()
ax.bar(x - width, mse_scores, width, label='MSE')
ax.bar(x, rmse_scores, width, label='RMSE')
ax.bar(x + width, mae_scores, width, label='MAE')

ax.set_xlabel('Models')
ax.set_title('Scores by model and metric')
ax.set_xticks(x)
ax.set_xticklabels(models.keys())
ax.legend()

fig.tight_layout()
plt.show()

# R^2 scores
print("\nR^2 Scores:")
for name, r2 in zip(models.keys(), r2_scores):
    print(f"{name}: {r2:.4f}")

# Choose best model
best_model_name = min(models.keys(), key=lambda name: mse_scores[list(models.keys()).index(name)])
print(f"\nThe best model based on MSE is: {best_model_name}")

# Save models
for name, model in models.items():
    dump(model, f"{name}_model.joblib")
