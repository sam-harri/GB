import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from joblib import dump, load

# Load data from multiple CSV files into a single DataFrame
folder_path = '/path_to_your_csv_files/'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
data_frames = [pd.read_csv(folder_path + csv_file) for csv_file in csv_files]
data = pd.concat(data_frames, ignore_index=True)

# Data split for both tasks
X1 = data[['SOC', 'T']]
y1 = data['SOH']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

X2 = data[['SOH', 'T']]
y2 = data['SOC']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Pipeline and Grid search parameters
model = Pipeline([
    ('poly', PolynomialFeatures()),
    ('scale', StandardScaler()),
    ('ridge', Ridge())
])

params = {
    'poly__degree': [1, 2, 3, 4, 5],
    'ridge__alpha': np.logspace(-4, 4, 9)
}

# Grid search for SOH prediction
grid_search1 = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search1.fit(X1_train, y1_train)

# Grid search for SOC prediction
grid_search2 = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search2.fit(X2_train, y2_train)

# Saving models
dump(grid_search1.best_estimator_, 'best_model_for_SOH_prediction.joblib')
dump(grid_search2.best_estimator_, 'best_model_for_SOC_prediction.joblib')

# Heatmap visualization for cross-validation results

def plot_heatmap(cv_results, param_grid, title):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(param_grid['poly__degree']), len(param_grid['ridge__alpha']))

    plt.figure(figsize=(10, 6))
    sns.heatmap(scores_mean, annot=True, fmt=".2f", xticklabels=param_grid['ridge__alpha'], yticklabels=param_grid['poly__degree'])
    plt.xlabel('Ridge Alpha')
    plt.ylabel('Polynomial Degree')
    plt.title(title)
    plt.show()

plot_heatmap(grid_search1.cv_results_, params, 'Cross Validation Heatmap for SOH Prediction')
plot_heatmap(grid_search2.cv_results_, params, 'Cross Validation Heatmap for SOC Prediction')

