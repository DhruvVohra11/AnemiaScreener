import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATASET_FILE = "comprehensive_ai_training_data.csv"
MODEL_OUTPUT = "comprehensive_anemia_regressor.pkl"

def train_model():
    print("--- STARTING MODEL TRAINING PHASE ---")
    print(f"Loading compiled feature matrix from {DATASET_FILE}...")
    
    # 1. Load Dataset
    df = pd.read_csv(DATASET_FILE)
    
    # 2. Separate Features and Target
    # We drop Subject_ID because the AI shouldn't memorize patient numbers!
    X = df.drop(columns=["Subject_ID", "Hb_Level"])
    y = df["Hb_Level"]
    
    print(f"Features being fed to AI: {list(X.columns)}")
    print(f"Total training samples: {len(df)}")
    
    # 3. Train/Test Split (80% Training, 20% Testing)
    # stratify isn't used for continuous regression, but a random_state ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and Train the Upgraded Regressor
    print("\nTraining Random Forest Regressor (this may take a moment with 27k+ rows)...")
    model = RandomForestRegressor(
        n_estimators=100,      # Number of trees
        max_depth=15,          # Prevents the model from over-memorizing the data
        min_samples_split=5,   # Ensures generic rules
        random_state=42,
        n_jobs=-1              # Uses all available CPU cores to speed up training
    )
    
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # 5. Evaluate Clinical Performance
    print("\n--- PERFORMANCE METRICS ---")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Mean Absolute Error (MAE):  {mae:.3f} g/dL")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f} g/dL")
    print(f"R-squared (R2) Score:         {r2:.3f}")
    
    # Interpret results clinically
    print("\nClinical Interpretation:")
    if mae <= 0.8:
        print(" -> Status: Exceptional. Near hospital-grade accuracy.")
    elif mae <= 1.2:
        print(" -> Status: Good. Highly viable for a non-invasive screening tool.")
    else:
        print(" -> Status: Variant. The model needs deeper trees or more feature scaling.")

    # 6. Feature Importance Breakdown
    print("\n--- FEATURE IMPORTANCE RANKING ---")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print(f"{f + 1}. {X.columns[indices[f]]:<16} : {importances[indices[f]]*100:.2f}%")

    # 7. Save the Model
    print(f"\nSaving finalized model brain to {MODEL_OUTPUT}...")
    joblib.dump(model, MODEL_OUTPUT)
    print("SUCCESS! Model is ready for deployment.")

if __name__ == '__main__':
    train_model()