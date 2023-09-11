import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from joblib import dump, load
import seaborn as sns

# Load data from multiple CSV files into a single DataFrame
folder_path = '/path_to_your_csv_files/'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
data_frames = [pd.read_csv(folder_path + csv_file) for csv_file in csv_files]
data = pd.concat(data_frames, ignore_index=True)

def plot_heatmap(grid_search, param_grid, model_name):
    results = pd.DataFrame(grid_search.cv_results_)
    scores = results['mean_test_score'].values.reshape(len(param_grid['svr__C']), 
                                                        len(param_grid['svr__gamma']))

    plt.figure(figsize=(10, 6))
    sns.heatmap(scores, annot=True, fmt=".3f", 
                xticklabels=param_grid['svr__gamma'],
                yticklabels=param_grid['svr__C'],
                cmap='viridis')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.title(f'Heatmap of Negative MSE for {model_name}')
    plt.show()

# Data for SOH prediction from SOC and T
X1 = data[['SOC', 'T']]
y1 = data['SOH']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Data for SOC prediction from SOH and T
X2 = data[['SOH', 'T']]
y2 = data['SOC']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Set up pipeline with StandardScaler and SVR
model = Pipeline([
    ('scale', StandardScaler()),
    ('svr', SVR())
])

# Grid search parameters for SVR
params = {
    'svr__kernel': ['rbf'],
    'svr__C': np.logspace(-3, 3, 7),
    'svr__gamma': np.logspace(-4, 4, 9)
}


# Train and tune model for predicting SOH
grid_search1 = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search1.fit(X1_train, y1_train)
print(f"Best parameters for SOH prediction: {grid_search1.best_params_}")
print(f"Best cross-validation score (MSE) for SOH prediction: {-grid_search1.best_score_}")

# Train and tune model for predicting SOC
grid_search2 = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search2.fit(X2_train, y2_train)
print(f"Best parameters for SOC prediction: {grid_search2.best_params_}")
print(f"Best cross-validation score (MSE) for SOC prediction: {-grid_search2.best_score_}")

# Save both models
model_filename1 = 'best_SVR_model_for_SOH_prediction.joblib'
dump(grid_search1.best_estimator_, model_filename1)
print(f"Best SVR model for SOH prediction saved as {model_filename1}")

model_filename2 = 'best_SVR_model_for_SOC_prediction.joblib'
dump(grid_search2.best_estimator_, model_filename2)
print(f"Best SVR model for SOC prediction saved as {model_filename2}")

# Visualize the original data
fig, axes = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': '3d'})

# Visualizing data for SOH predictions
axes[0].scatter(X1_train['SOC'], X1_train['T'], y1_train, c='b', marker='o', alpha=0.6)
axes[0].set_xlabel('SOC')
axes[0].set_ylabel('T')
axes[0].set_zlabel('SOH')
axes[0].set_title('Original Data for SOH Prediction')

# Visualizing data for SOC predictions
axes[1].scatter(X2_train['SOH'], X2_train['T'], y2_train, c='r', marker='o', alpha=0.6)
axes[1].set_xlabel('SOH')
axes[1].set_ylabel('T')
axes[1].set_zlabel('SOC')
axes[1].set_title('Original Data for SOC Prediction')

plot_heatmap(grid_search1, params, "SOH Prediction")
plot_heatmap(grid_search2, params, "SOC Prediction")

plt.tight_layout()
plt.show()
