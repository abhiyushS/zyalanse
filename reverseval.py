import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('Fusariumsp.BVKT.csv')  

X = df[['SORBITOL (%)', 'YEAST EXTRACT (%)', 'pH', 'TEMPERATURE (degree C)', 'AGITATION (rpm)']]
y = df['EXPERIMENTAL XYLANASE ACTIVTIY (U/mL)']

#
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


def optimize_missing_params(known_params, known_indices, target_activity):
    """
    Optimize missing parameters while keeping known parameters fixed.

    known_params: Dictionary of known parameter values.
    known_indices: Indices of known parameters in the feature array.
    target_activity: The desired xylanase activity value.
    """
    
  
    initial_guess = np.full(X.shape[1], np.nan)
    
    
    for idx, val in zip(known_indices, known_params):
        initial_guess[idx] = val
    

    missing_indices = [i for i in range(X.shape[1]) if i not in known_indices]
    
    initial_missing_values = np.mean(X_scaled[:, missing_indices], axis=0)

    def objective_function(missing_values):
        full_params = initial_guess.copy()
        
        for i, idx in enumerate(missing_indices):
            full_params[idx] = missing_values[i]

        predicted_activity = model.predict(full_params.reshape(1, -1))[0]
        return abs(predicted_activity - target_activity)

    result = minimize(objective_function, initial_missing_values, method='Nelder-Mead')


    optimized_params = result.x
    
    for i, idx in enumerate(missing_indices):
        initial_guess[idx] = optimized_params[i]


    final_params = scaler.inverse_transform(initial_guess.reshape(1, -1))[0]

    missing_param_results = {X.columns[i]: final_params[i] for i in missing_indices}
    
    return missing_param_results, result.success


print("\nEnter known parameter values:")
known_params = {}
known_indices = []


for i, col in enumerate(X.columns):
    val = input(f"Enter {col} (or press Enter if unknown): ")
    if val.strip():
        known_params[i] = float(val)
        known_indices.append(i)

target_activity = float(input("\nEnter target Xylanase Activity: "))


optimized_params, success = optimize_missing_params(list(known_params.values()), known_indices, target_activity)

if success:
    print("\nOptimized Missing Parameter Values:")
    for key, value in optimized_params.items():
        print(f"{key}: {value:.4f}")
else:
    print("\nOptimization failed. Try different inputs or check parameter constraints.")

