import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('Aspergillusfumigatussyn.csv', encoding='latin1')

# Convert target column to numeric (handle errors)
df['EXPERIMENTAL XYLANASE ACTIVITY (IU/gds)'] = pd.to_numeric(df['EXPERIMENTAL XYLANASE ACTIVITY (IU/gds)'], errors='coerce')
df = df.dropna()

# Define features and target
X = df[['SUBSTRATE CONCENTRATION (g)', 'TEMPERATURE (degree C)', 'INITIAL pH',
        'INITIAL MOISTURE CONTENT (%)', 'INCUBATION TIME (hrs)']]
y = df['EXPERIMENTAL XYLANASE ACTIVITY (IU/gds)']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Function to minimize (find X for given y)
def objective_function(params, target_activity):
    """ Predict xylanase activity for given params and return difference from target. """
    params_scaled = scaler.transform([params])  # Scale input
    predicted_activity = model.predict(params_scaled)[0]
    return abs(predicted_activity - target_activity)  # Minimize absolute error

# Function to find best parameters for a given activity level
def find_optimal_parameters(target_activity):
    """ Find the best combination of input parameters to achieve a target xylanase activity. """
    
    # Initial guess (mean values of each feature)
    initial_guess = X.mean().values  

    # Set bounds for each parameter
    bounds = [(X[col].min(), X[col].max()) for col in X.columns]

    # Minimize the difference between predicted and target activity
    result = minimize(objective_function, initial_guess, args=(target_activity,),
                      bounds=bounds, method='L-BFGS-B')

    # Extract optimized parameter values
    if result.success:
        optimal_params = result.x
        return dict(zip(X.columns, optimal_params))
    else:
        return None

# Get user input and find optimal parameters
target_activity = float(input("Enter desired Xylanase Activity (IU/gds): "))
optimal_params = find_optimal_parameters(target_activity)

if optimal_params:
    print("\nOptimal parameters to achieve the desired activity:")
    for key, value in optimal_params.items():
        print(f"{key}: {value:.4f}")
else:
    print("\nCould not find suitable parameters. Try a different target value.")











