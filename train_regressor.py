import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATASET_FILE = "clinical_features_dataset.csv"
MODEL_FILE = "anemia_regressor_model.pkl" # New file name!
SCALER_FILE = "feature_scaler.pkl"

def train_hemoglobin_regressor():
    print("Loading dataset for Regression Training...")
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        return

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    feature_columns = [
        'dc_red', 'dc_ir', 'ac_red', 'ac_ir', 
        'r_ratio', 'pi_red', 'skewness_red', 'kurtosis_red'
    ]
    
    X = df[feature_columns]
    # THE UPGRADE: We are predicting the exact number now, not 0 or 1
    y = df['hb_level'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n--- REGRESSION PERFORMANCE ---")
    print(f"Mean Absolute Error (MAE): +/- {mae:.2f} g/dL")
    print(f"R-Squared Score: {r2:.2f}")

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nSuccess! Regressor saved to '{MODEL_FILE}'")

if __name__ == '__main__':
    train_hemoglobin_regressor()