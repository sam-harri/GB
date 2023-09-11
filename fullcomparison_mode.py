import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from joblib import dump

# 1. Load and split data

folder_path = '/path_to_your_csv_files/'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
data_frames = [pd.read_csv(folder_path + csv_file) for csv_file in csv_files]
data = pd.concat(data_frames, ignore_index=True)

# data = pd.read_csv('/path_to_your_consolidated_csv.csv')


X = data[['SOC', 'T']]
y = data['SOH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define helper functions

def train_and_tune(model, params, X_train, y_train, title):
    grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Heatmap for CV results
    scores = grid_search.cv_results_['mean_test_score']
    scores = np.array(scores).reshape(len(params['C']), len(params['gamma']))
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(scores, annot=True, fmt=".3f", xticklabels=params['gamma'], yticklabels=params['C'])
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.title(f"Cross Validation Heatmap for {title}")
    plt.show()

    # Save model
    dump(grid_search.best_estimator_, f'best_{title}.joblib')
    
    return grid_search.best_estimator_

def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, mae

# 3. Train, tune and visualize each model

# SVR
svr = SVR()
svr_params = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
best_svr = train_and_tune(svr, svr_params, X_train, y_train, 'SVR')

# Polynomial
pipe = Pipeline([
    ('poly', PolynomialFeatures()),
    ('scale', StandardScaler()),
    ('ridge', Ridge())
])
poly_params = {'poly__degree': [2, 3, 4], 'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
best_poly = train_and_tune(pipe, poly_params, X_train, y_train, 'Polynomial')

# XGBoost
xgb = XGBRegressor()
xgb_params = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
best_xgb = train_and_tune(xgb, xgb_params, X_train, y_train, 'XGBoost')

# 4. Compare models

metrics = {'SVR': calculate_metrics(best_svr, X_test, y_test),
           'Polynomial': calculate_metrics(best_poly, X_test, y_test),
           'XGBoost': calculate_metrics(best_xgb, X_test, y_test)}

df = pd.DataFrame(metrics, index=['MSE', 'RMSE', 'MAE']).T
df.plot(kind='bar', figsize=(10, 6))
plt.title("Comparison of Models")
plt.ylabel("Value")
plt.show()

# Choose the best model based on MSE
best_model_name = df['MSE'].idxmin()
print(f"The best model is: {best_model_name}")
