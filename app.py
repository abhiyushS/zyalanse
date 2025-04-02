# import pickle
# import numpy as np
# from flask import Flask, render_template, request, flash

# # Load models and scalers for each strain
# models = {
#     "Trichoderma afroharzianum": pickle.load(open("models/gradient_boosting_model1.pkl", "rb")),
#     "Fusarium sp. BVKT": pickle.load(open("models/gradient_boosting_model2.pkl", "rb")),
#     "Bacillus tequilensis": pickle.load(open("models/random_forest_model3.pkl", "rb")),
#     "Aspergillus Niger": pickle.load(open("models/xgboost_model5.pkl", "rb")),
#     "AUM60": pickle.load(open("models/random_forest_model60.pkl", "rb")),
#     "AUM64": pickle.load(open("models/random_forest_model64.pkl", "rb")),
# }

# scalers = {
#     "Trichoderma afroharzianum": pickle.load(open("scalers/scaler1.pkl", "rb")),
#     "Fusarium sp. BVKT": pickle.load(open("scalers/scaler2.pkl", "rb")),
#     "Bacillus tequilensis": pickle.load(open("scalers/scaler3.pkl", "rb")),
#     "Aspergillus Niger": pickle.load(open("scalers/scaler5.pkl", "rb")),
#     "AUM60": pickle.load(open("scalers/scaler60.pkl", "rb")),
#     "AUM64": pickle.load(open("scalers/scaler64.pkl", "rb")),
# }

# app = Flask(__name__)
# app.secret_key = "your_secret_key"  # Needed for flash messages

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         strain_name = request.form.get("strain_name")
        
#         if not strain_name:
#             flash("Please select a strain.", "error")
#             return render_template("index.html")

#         try:
#             # Get the parameters from the form based on strain selected
#             if strain_name == "Trichoderma afroharzianum":
#                 parameters = [
#                     float(request.form["incubation_time"]),
#                     float(request.form["humidity"]),
#                     float(request.form["temperature"]),
#                     float(request.form["inoculum_size"]),
#                 ]
#             elif strain_name == "Fusarium sp. BVKT":
#                 parameters = [
#                     float(request.form["sorbitol"]),
#                     float(request.form["yeast_extract"]),
#                     float(request.form["pH"]),
#                     float(request.form["temperature"]),
#                     float(request.form["agitation"]),
#                 ]
#             elif strain_name == "Bacillus tequilensis":
#                 parameters = [
#                     float(request.form["birchwood_xylan"]),
#                     float(request.form["yeast_extract"]),
#                     float(request.form["temperature"]),
#                     float(request.form["incubation_period"]),
#                 ]
#             elif strain_name == "Aspergillus Niger":
#                 parameters = [
#                     float(request.form["substrate_concentration"]),
#                     float(request.form["temperature"]),
#                     float(request.form["initial_pH"]),
#                     float(request.form["initial_moisture_content"]),
#                     float(request.form["incubation_time"]),
#                 ]
#             elif strain_name == "AUM60":
#                 parameters = [
#                     float(request.form["temperature"]),
#                     float(request.form["pH"]),
#                     float(request.form["fermentation_time"]),
#                     float(request.form["substrate_concentration"]),
#                     float(request.form["agitation_rate"]),
#                 ]
#             elif strain_name == "AUM64":
#                 parameters = [
#                     float(request.form["temperature"]),
#                     float(request.form["pH"]),
#                     float(request.form["fermentation_time"]),
#                     float(request.form["substrate_concentration"]),
#                     float(request.form["agitation_rate"]),
#                 ]
#             else:
#                 flash("Invalid strain selected.", "error")
#                 return render_template("index.html")
            
#             # Use the appropriate scaler and model for the selected strain
#             scaler = scalers[strain_name]
#             model = models[strain_name]
            
#             # Scale input parameters
#             scaled_params = scaler.transform([parameters])
            
#             # Make prediction using the selected model
#             prediction = model.predict(scaled_params)
            
#             return render_template("index.html", prediction=round(prediction[0], 2), strain_name=strain_name)
        
#         except ValueError as e:
#             flash(f"Error processing form inputs: {e}", "error")
#             return render_template("index.html")
#         except KeyError as e:
#             flash(f"Model or scaler not found for {strain_name}.", "error")
#             return render_template("index.html")
    
#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)


import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from flask import Flask, render_template, request, flash
from sklearn.preprocessing import StandardScaler
import os

# Function to safely load models and scalers
def load_models_and_scalers():
    models = {}
    scalers = {}
    
    model_files = {
        "Trichoderma afroharzianum": "models/gradient_boosting_model1.pkl",
        "Fusarium sp. BVKT": "models/gradient_boosting_model2.pkl",
        "Bacillus tequilensis": "models/random_forest_model3.pkl",
        "Aspergillus Niger": "models/xgboost_model5.pkl",
        "AUM60": "models/random_forest_model60.pkl",
        "AUM64": "models/random_forest_model64.pkl",
        "Aspergillus fumigatus": "models/random_forest_model64.pkl",
    }
    
    scaler_files = {
        "Trichoderma afroharzianum": "scalers/scaler1.pkl",
        "Fusarium sp. BVKT": "scalers/scaler2.pkl",
        "Bacillus tequilensis": "scalers/scaler3.pkl",
        "Aspergillus Niger": "scalers/scaler5.pkl",
        "AUM60": "scalers/scaler60.pkl",
        "AUM64": "scalers/scaler64.pkl",
        
        "Aspergillus fumigatus": "scalers/scaler64.pkl",
    }

    # Load models
    for strain, model_path in model_files.items():
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    models[strain] = pickle.load(f)
            except Exception as e:
                print(f"🚨 ERROR: Unable to load model {model_path} -> {e}")
        else:
            print(f"⚠️ Warning: Model file missing for {strain} ({model_path})")

    # Load scalers
    for strain, scaler_path in scaler_files.items():
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scalers[strain] = pickle.load(f)
            except Exception as e:
                print(f"🚨 ERROR: Unable to load scaler {scaler_path} -> {e}")
        else:
            print(f"⚠️ Warning: Scaler file missing for {strain} ({scaler_path})")

    return models, scalers

models, scalers = load_models_and_scalers()

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Standardize dataset column names
column_mapping = {
    "TEMPERATURE (degree C)": "temperature",
    "FERMENTATION TIME (DAYS)": "fermentation_time",
    "SUBSTRATE CONCENTRATION (%)": "substrate_concentration",
    "AGITATION RATE (rpm)": "agitation_rate",
    "INITIAL pH": "pH",
    "INCUBATION TIME (hrs)": "incubation_time",
    "HUMIDITY (%)": "humidity",
    "INOCULUM SIZE (spore/g)": "inoculum_size",
    "BIRCHWOOD XYLAN (%)": "birchwood_xylan",
    "YEAST EXTRACT (%)": "yeast_extract",
    "SORBITOL (%)": "sorbitol",

}

# Define strain-specific parameters
# Define strain-specific parameters with exact case-sensitive names
strain_parameters = {
    "Trichoderma afroharzianum": [
        "INCUBATION TIME (DAYS)",
        "HUMIDITY (%)",
        "TEMPERATURE (degree C)",
        "INOCULUM SIZE (spore/g)"
    ],
    "Fusarium sp. BVKT": [
        "SORBITOL (%)",
        "YEAST EXTRACT (%)",
        "pH",
        "TEMPERATURE (degree C)",
        "AGITATION (rpm)"
    ],
    "Bacillus tequilensis": [
        "BIRCHWOOD XYLAN (%)",
        "YEAST EXTRACT (%)",
        "TEMPERATURE (degree C)",
        "INCUBATION PERIOD (hours)"
    ],
    "Aspergillus Niger": [
        "SUBSTRATE CONCENTRATION (g)",
        "TEMPERATURE (degree C)",
        "INITIAL pH",
        "INITIAL MOISTURE CONTENT (%)",
        "INCUBATION TIME (hrs)"
    ],
    "Aspergillus fumigatus": [
        "SUBSTRATE CONCENTRATION (g)",
        "TEMPERATURE (degree C)",
        "INCUBATION TIME (hrs)",
        "INITIAL MOISTURE CONTENT (%)",
        "INITIAL pH"
    ],
    "AUM60": [
        "TEMPERATURE (degree C)",
        "pH",
        "FERMENTATION TIME (DAYS)",
        "SUBSTRATE CONCENTRATION (%)",
        "AGITATION RATE (rpm)"
    ],
    "AUM64": [
        "TEMPERATURE (degree C)",
        "pH",
        "FERMENTATION TIME (DAYS)",
        "SUBSTRATE CONCENTRATION (%)",
        "AGITATION RATE (rpm)"
    ]
}


# Correct dataset filenames
dataset_mapping = {
    "Trichoderma afroharzianum": "datasets/Trichoderma_afroharzianum.csv",
    "Fusarium sp. BVKT": "datasets/Fusariumsp.BVKT.csv",
    "Bacillus tequilensis": "datasets/Bacillus_tequilensis.csv",
    "Aspergillus Niger": "datasets/AspergillusNiger.csv",
    "AUM60": "datasets/AUM60_synthetic_lst.csv",
    "AUM64": "datasets/AUM64_synthetic.csv",
    "Aspergillus fumigatus": "datasets/Aspergillus_fumigatus.csv",
}





@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        strain_name = request.form.get("strain_name")

        if not strain_name or strain_name not in models:
            flash("Invalid or missing strain selection.", "error")
            return render_template("index.html")

        try:
            params = strain_parameters.get(strain_name, [])
            input_values = []

            for param in params:
                value = request.form.get(param)
                if not value:
                    flash(f"Missing value for {param}.", "error")
                    return render_template("index.html")
                input_values.append(float(value))

            # Scale input parameters
            scaler = scalers[strain_name]
            model = models[strain_name]
            scaled_params = scaler.transform([input_values])

            # Predict enzyme activity
            prediction = model.predict(scaled_params)[0]
            
            return render_template("index.html", prediction=round(prediction, 2), strain_name=strain_name)

        except ValueError as e:
            flash(f"Invalid input format: {e}", "error")
        except KeyError:
            flash(f"Model or scaler missing for {strain_name}.", "error")
        except Exception as e:
            flash(f"Unexpected error: {e}", "error")

    return render_template("index.html")

@app.route("/reverse_prediction", methods=["GET", "POST"])
def reverse_prediction():
    if request.method == "POST":
        strain_name = request.form.get("strain_name")

        if not strain_name or strain_name not in models:
            flash("Invalid or missing strain selection.", "error")
            return render_template("reverse.html")

        try:
            params = strain_parameters[strain_name]
            known_params = {}
            known_indices = []
            target_activity = request.form.get("target_activity", "").strip()

            if not target_activity:
                flash("Target activity is required.", "error")
                return render_template("reverse.html")

            target_activity = float(target_activity)

            for i, param in enumerate(params):
                value = request.form.get(param, "").strip()
                if value:
                    known_params[i] = float(value)
                    known_indices.append(i)

            scaler = scalers[strain_name]
            model = models[strain_name]
            dataset_path = dataset_mapping.get(strain_name)

            if not dataset_path or not os.path.exists(dataset_path):
                print(f"🚨 ERROR: Dataset file missing -> {dataset_path}")
                flash(f"Dataset missing for {strain_name}. Expected file: {dataset_path}", "error")
                return render_template("reverse.html")

            dataset = pd.read_csv(dataset_path)
            X = dataset[params]

            # Scale dataset
            X_scaled = scaler.transform(X)

            # Function to optimize missing parameters
            def optimize_missing_params(known_params, known_indices, target_activity):
                initial_guess = np.full(len(params), np.nan)

                for idx, val in known_params.items():
                    initial_guess[idx] = val

                missing_indices = [i for i in range(len(params)) if i not in known_indices]
                initial_missing_values = np.mean(X_scaled[:, missing_indices], axis=0)

                def objective_function(missing_values):
                    full_params = initial_guess.copy()
                    for i, idx in enumerate(missing_indices):
                        full_params[idx] = missing_values[i]
                    predicted_activity = model.predict(full_params.reshape(1, -1))[0]
                    return abs(predicted_activity - target_activity)

                result = minimize(objective_function, initial_missing_values, method="Nelder-Mead")

                optimized_params = result.x
                for i, idx in enumerate(missing_indices):
                    initial_guess[idx] = optimized_params[i]

                final_params = scaler.inverse_transform(initial_guess.reshape(1, -1))[0]
                return {params[i]: final_params[i] for i in missing_indices}, result.success

            optimized_params, success = optimize_missing_params(known_params, known_indices, target_activity)

            if success:
                return render_template(
                    "reverse.html",
                    optimized_params=optimized_params,
                    strain_name=strain_name,
                    target_activity=target_activity
                )
            else:
                flash("Optimization failed. Try different inputs.", "error")

        except Exception as e:
            flash(f"Unexpected error: {e}", "error")

    return render_template("reverse.html")

if __name__ == "__main__":
    app.run(debug=True)