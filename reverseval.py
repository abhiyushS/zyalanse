import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
df = pd.read_csv('Fusariumsp.BVKT.csv')  # Replace with actual filename

# Define feature columns and target variable
X = df[['SORBITOL (%)', 'YEAST EXTRACT (%)', 'pH', 'TEMPERATURE (degree C)', 'AGITATION (rpm)']]
y = df['EXPERIMENTAL XYLANASE ACTIVTIY (U/mL)']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a model (Gradient Boosting Regressor)
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to optimize missing parameters
def optimize_missing_params(known_params, known_indices, target_activity):
    """
    Optimize missing parameters while keeping known parameters fixed.

    known_params: Dictionary of known parameter values.
    known_indices: Indices of known parameters in the feature array.
    target_activity: The desired xylanase activity value.
    """
    
    # Create a full parameter array with NaNs for missing values
    initial_guess = np.full(X.shape[1], np.nan)
    
    # Fill known values (DO NOT OPTIMIZE THESE)
    for idx, val in zip(known_indices, known_params):
        initial_guess[idx] = val
    
    # Find missing indices
    missing_indices = [i for i in range(X.shape[1]) if i not in known_indices]
    
    # Initial guess for missing values (Mean of the dataset for those parameters)
    initial_missing_values = np.mean(X_scaled[:, missing_indices], axis=0)
    
    # Objective function: Minimize error in predicted vs target activity
    def objective_function(missing_values):
        full_params = initial_guess.copy()
        
        # Fill only the missing values (keep known values fixed)
        for i, idx in enumerate(missing_indices):
            full_params[idx] = missing_values[i]

        # Predict and calculate the error
        predicted_activity = model.predict(full_params.reshape(1, -1))[0]
        return abs(predicted_activity - target_activity)

    # Minimize the function (only for missing values)
    result = minimize(objective_function, initial_missing_values, method='Nelder-Mead')

    # Get optimized missing values
    optimized_params = result.x
    
    # Fill missing values in the parameter array (keeping known ones unchanged)
    for i, idx in enumerate(missing_indices):
        initial_guess[idx] = optimized_params[i]

    # Convert back to original scale
    final_params = scaler.inverse_transform(initial_guess.reshape(1, -1))[0]
    
    # Store only missing parameter values
    missing_param_results = {X.columns[i]: final_params[i] for i in missing_indices}
    
    return missing_param_results, result.success

# Example Usage:
print("\nEnter known parameter values:")
known_params = {}
known_indices = []

# User enters known values
for i, col in enumerate(X.columns):
    val = input(f"Enter {col} (or press Enter if unknown): ")
    if val.strip():
        known_params[i] = float(val)
        known_indices.append(i)

target_activity = float(input("\nEnter target Xylanase Activity: "))

# Optimize missing parameters
optimized_params, success = optimize_missing_params(list(known_params.values()), known_indices, target_activity)

if success:
    print("\nOptimized Missing Parameter Values:")
    for key, value in optimized_params.items():
        print(f"{key}: {value:.4f}")
else:
    print("\nOptimization failed. Try different inputs or check parameter constraints.")

