# Xylanase Enzyme Activity Prediction & Optimization

## Overview
This project aims to **predict and optimize xylanase enzyme activity** based on different strains using machine learning models. It includes:
- **Prediction Model:** Estimates xylanase activity based on given input parameters.
- **Optimization Model:** Computes missing parameters to achieve a target enzyme activity.
- **SHAP Explainability:** Provides interpretability for model predictions.
- **Flask Web App:** A user-friendly interface for accessing these functionalities.

## Features
✅ Supports multiple strains with separate models.
✅ Two core functionalities:
  - **Predict Xylanase Activity** by entering all required inputs.
  - **Optimize Missing Parameters** by providing known values and a target activity.
✅ Uses **Gradient Boosting, Random Forest, and other models** for different strains.
✅ **SQLite integration (optional)** for logging user inputs and predictions.
✅ **SHAP analysis** for model interpretability.

## Technologies Used
- **Programming Language:** Python
- **Frameworks:** Flask (Backend), Scikit-learn (ML Models), SHAP (Explainability)
- **Libraries:** Pandas, NumPy, SciPy, Matplotlib, Joblib (for model storage)
- **Database:** SQLite (Optional for logging)
- **Frontend:** HTML, CSS, JavaScript

## Project Structure
```
project_root/
│-- app.py                    # Flask application
│-- models/
│   │-- strain1_model.pkl      # Trained model for strain 1
│   │-- strain1_scaler.pkl     # Scaler for strain 1
│   │-- strain2_model.pkl      # Trained model for strain 2
│-- static/
│-- templates/
│-- utils/
│   │-- prediction.py          # Prediction logic
│   │-- optimization.py        # Parameter optimization logic
│   │-- shap_analysis.py       # SHAP explainability
│-- requirements.txt           # Dependencies
│-- README.md                  # Project documentation
```

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/xylanase-prediction.git
   cd xylanase-prediction
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask app:
   ```bash
   python app.py
   ```

## Usage
- **Select a strain** from the dropdown.
- **Choose a function:**
  - **Predict Enzyme Activity**: Enter all parameters and get a prediction.
  - **Optimize Parameters**: Provide known values and target activity; get computed missing values.
- **View SHAP analysis** to understand model decisions.

## Future Improvements
🚀 Expand to more enzyme activity models.
📊 Integrate real-time data logging.
🔍 Improve UI/UX with interactive visualizations.

## License
This project is licensed under the **MIT License**.

---
🚀 Developed by **Abhiyush Satyam**

