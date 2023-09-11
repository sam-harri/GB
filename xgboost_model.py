import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from joblib import dump

def plot_heatmap(grid_search, params, model_name, param1='learning_rate', param2='n_estimators'):
    results = pd.DataFrame(grid_search.cv_results_)
    scores = results.pivot(index=f'param_{param1}', columns=f'param_{param2}', values='mean_test_score')

    plt.figure(figsize=(10, 6))
    sns.heatmap(scores, annot=True, fmt=".3f", cmap='viridis')
    plt.title(f'Heatmap of Negative MSE for {model_name}')
    plt.show()

# Load data
folder_path = '/path_to_your_csv_files/'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
data_frames = [pd.read_csv(folder_path + csv_file) for csv_file in csv_files]
data = pd.concat(data_frames, ignore_index=True)

# Prepare data
X1 = data[['SOC', 'T']]
y1 = data['SOH']
X2 = data[['SOH', 'T']]
y2 = data['SOC']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Hyperparameter tuning for XGBoost
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 150],
    # 'max_depth': [3, 5, 7],
    # 'subsample': [0.8, 0.9, 1],
    # 'colsample_bytree': [0.8, 0.9, 1]
}

# Model for predicting SOH
model1 = XGBRegressor(random_state=42)
grid_search1 = GridSearchCV(model1, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search1.fit(X1_train, y1_train)

# Model for predicting SOC
model2 = XGBRegressor(random_state=42)
grid_search2 = GridSearchCV(model2, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search2.fit(X2_train, y2_train)

# Save models
dump(grid_search1.best_estimator_, 'best_XGB_model_for_SOH_prediction.joblib')
dump(grid_search2.best_estimator_, 'best_XGB_model_for_SOC_prediction.joblib')

# Function to plot heatmap for GridSearchCV results (focusing on two parameters for simplicity)
def plot_heatmap(grid_search, params, model_name, param1='learning_rate', param2='n_estimators'):
    results = pd.DataFrame(grid_search.cv_results_)
    scores = results.pivot(index=f'param_{param1}', columns=f'param_{param2}', values='mean_test_score')

    plt.figure(figsize=(10, 6))
    sns.heatmap(scores, annot=True, fmt=".3f", cmap='viridis')
    plt.title(f'Heatmap of Negative MSE for {model_name}')
    plt.show()

# Plot results
plot_heatmap(grid_search1, param_grid, 'SOH Prediction', 'learning_rate', 'n_estimators')
plot_heatmap(grid_search2, param_grid, 'SOC Prediction', 'learning_rate', 'n_estimators')
