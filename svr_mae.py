import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import load
import seaborn as sns

# Parameters
folder_path = '/path_to_your_csv_files/'
seed = 42  # Replace with your actual random state or seed

# Load data from multiple CSV files into a single DataFrame
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
data_frames = [pd.read_csv(folder_path + csv_file) for csv_file in csv_files]
data = pd.concat(data_frames, ignore_index=True)

# Data for SOH prediction from SOC and T
X1 = data[['SOC', 'T']]
y1 = data['SOH']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=seed)

# Data for SOC prediction from SOH and T
X2 = data[['SOH', 'T']]
y2 = data['SOC']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=seed)

# Load the best models
best_model_soh = load('best_SVR_model_for_SOH_prediction.joblib')
best_model_soc = load('best_SVR_model_for_SOC_prediction.joblib')

# Predict using the best models
y1_pred = best_model_soh.predict(X1_test)
y2_pred = best_model_soc.predict(X2_test)

# Calculate Absolute Percentage Error (APE)
ape_soh = np.abs((y1_test - y1_pred) / y1_test) * 100
ape_soc = np.abs((y2_test - y2_pred) / y2_test) * 100

# Plot APE for SOH prediction with bars touching
plt.figure(figsize=(18, 6))
sns.barplot(x=y1_test.index, y=ape_soh, linewidth=0)
plt.xlabel("Index")
plt.ylabel("APE")
plt.title("Absolute Percentage Error for SOH Prediction")
plt.show()

# Plot APE for SOC prediction with bars touching
plt.figure(figsize=(18, 6))
sns.barplot(x=y2_test.index, y=ape_soc, linewidth=0)
plt.xlabel("Index")
plt.ylabel("APE")
plt.title("Absolute Percentage Error for SOC Prediction")
plt.show()

# Additional Diagnostic Plots
# You can plot predicted vs. true values scatter plot to check linearity
plt.figure(figsize=(12, 6))
plt.scatter(y1_test, y1_pred, alpha=0.7, edgecolors='w', linewidth=0.5)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Predicted vs True Values for SOH Prediction")
plt.plot([min(y1_test), max(y1_test)], [min(y1_test), max(y1_test)], 'k--', lw=2)  # 45-degree line
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(y2_test, y2_pred, alpha=0.7, edgecolors='w', linewidth=0.5)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Predicted vs True Values for SOC Prediction")
plt.plot([min(y2_test), max(y2_test)], [min(y2_test), max(y2_test)], 'k--', lw=2)  # 45-degree line
plt.show()

