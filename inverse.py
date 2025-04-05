import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv('Aspergillusfumigatussyn.csv', encoding='latin1')

df['EXPERIMENTAL XYLANASE ACTIVITY (IU/gds)'] = pd.to_numeric(df['EXPERIMENTAL XYLANASE ACTIVITY (IU/gds)'], errors='coerce')
df = df.dropna()


X = df[['SUBSTRATE CONCENTRATION (g)', 'TEMPERATURE (degree C)', 'INITIAL pH',
        'INITIAL MOISTURE CONTENT (%)', 'INCUBATION TIME (hrs)']]
y = df['EXPERIMENTAL XYLANASE ACTIVITY (IU/gds)']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)


def objective_function(params, target_activity):
    """ Predict xylanase activity for given params and return difference from target. """
    params_scaled = scaler.transform([params])  # Scale input
    predicted_activity = model.predict(params_scaled)[0]
    return abs(predicted_activity - target_activity)  # Minimize absolute error

def find_optimal_parameters(target_activity):
    """ Find the best combination of input parameters to achieve a target xylanase activity. """
    
    initial_guess = X.mean().values  


    bounds = [(X[col].min(), X[col].max()) for col in X.columns]

    
    result = minimize(objective_function, initial_guess, args=(target_activity,),
                      bounds=bounds, method='L-BFGS-B')

    if result.success:
        optimal_params = result.x
        return dict(zip(X.columns, optimal_params))
    else:
        return None

target_activity = float(input("Enter desired Xylanase Activity (IU/gds): "))
optimal_params = find_optimal_parameters(target_activity)

if optimal_params:
    print("\nOptimal parameters to achieve the desired activity:")
    for key, value in optimal_params.items():
        print(f"{key}: {value:.4f}")
else:
    print("\nCould not find suitable parameters. Try a different target value.")











