import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# File paths
DATASET_FILE = "clinical_features_dataset.csv"
MODEL_FILE = "anemia_rf_model.pkl"
SCALER_FILE = "feature_scaler.pkl"

def train_anemia_classifier():
    print(f"Loading engineered dataset from {DATASET_FILE}...")
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print("Error: Dataset not found. Run dataset_processor.py first.")
        return

    # Drop any rows where the math failed (NaNs or Infinities)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"Dataset loaded. Total viable samples: {len(df)}")

    # 1. Define our Feature Vector (X) and Target Label (y)
    # Notice we are only using the optical/shape features we can pull live from your finger
    feature_columns = [
        'dc_red', 'dc_ir', 
        'ac_red', 'ac_ir', 
        'r_ratio', 'pi_red', 
        'skewness_red', 'kurtosis_red'
    ]
    
    X = df[feature_columns]
    y = df['label'] # 0 = Normal, 1 = Anemic

    # 2. Split into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Scale the features
    # This ensures a massive number like 'dc_red' doesn't overpower a tiny number like 'kurtosis'
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize and Train the Random Forest Engine
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate the Model
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n--- MODEL PERFORMANCE METRICS ---")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=["Normal (0)", "Anemic (1)"]))

    # 6. Extract Feature Importances (What did the AI care about most?)
    importances = model.feature_importances_
    feature_rank = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
    feature_rank = feature_rank.sort_values('Importance', ascending=False)
    
    print("\n--- FEATURE IMPORTANCE RANKING ---")
    print(feature_rank.to_string(index=False))

    # 7. Save the Brain to Disk
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nSuccess! Model saved to '{MODEL_FILE}'")
    print(f"Scaler saved to '{SCALER_FILE}'")

if __name__ == '__main__':
    train_anemia_classifier()