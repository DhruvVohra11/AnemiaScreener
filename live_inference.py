from max30102 import MAX30102
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import joblib
import time
import os
import csv
from datetime import datetime

# Load the REGRESSOR model now
MODEL_FILE = "anemia_regressor_model.pkl"
SCALER_FILE = "feature_scaler.pkl"
LOG_FILE = "live_patient_log.csv"

def run_live_diagnostic():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        print("Regressor and Scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading AI files: {e}")
        return
        
    feature_columns = [
        'dc_red', 'dc_ir', 'ac_red', 'ac_ir', 
        'r_ratio', 'pi_red', 'skewness_red', 'kurtosis_red'
    ]
    
    try:
        while True:
            input("\nPress ENTER when finger is perfectly still to begin scan...")
            
            sensor = MAX30102() 
            time.sleep(1) 
            
            red_list = []
            ir_list = []
            print("Collecting live pulse window (4 seconds)...")
            
            while len(red_list) < 200:
                num_bytes = sensor.get_data_present()
                if num_bytes > 0:
                    for _ in range(num_bytes):
                        red, ir = sensor.read_fifo()
                        if red > 0 and ir > 0:
                            red_list.append(red)
                            ir_list.append(ir)
                time.sleep(0.01)
                
            red_arr = np.array(red_list[:200])
            ir_arr = np.array(ir_list[:200])
            
            dc_red, dc_ir = float(np.mean(red_arr)), float(np.mean(ir_arr))
            ac_red = float(np.max(red_arr) - np.min(red_arr))
            ac_ir = float(np.max(ir_arr) - np.min(ir_arr))
            
            if ac_ir > 0 and dc_ir > 0 and dc_red > 0:
                r_ratio = (ac_red / dc_red) / (ac_ir / dc_ir)
                pi_red = (ac_red / dc_red) * 100
                skew_red = float(skew(red_arr))
                kurt_red = float(kurtosis(red_arr))
                
                # --- THE SIGNAL QUALITY GUARDRAIL ---
                if kurt_red > 15.0 or kurt_red < -5.0:
                    print(f"\n[ERROR] Motion Artifact Detected (Kurtosis: {kurt_red:.2f})")
                    print("Reading rejected. Please hold finger completely still and try again.")
                    continue # Skip to the next loop, don't feed garbage to the AI
                
                live_data = pd.DataFrame([[
                    dc_red, dc_ir, ac_red, ac_ir, 
                    r_ratio, pi_red, skew_red, kurt_red
                ]], columns=feature_columns)
                
                live_data_scaled = scaler.transform(live_data)
                
                # Predict exact Hb Level
                hb_prediction = model.predict(live_data_scaled)[0]
                
                print("\n--- DIAGNOSTIC RESULTS ---")
                print(f"Quality Metrics -> R: {r_ratio:.4f} | Kurtosis: {kurt_red:.2f}")
                print(f"=> ESTIMATED HEMOGLOBIN: {hb_prediction:.1f} g/dL")
                
                if hb_prediction < 12.0:
                    print("=> STATUS: WARNING - Below Normal Range")
                else:
                    print("=> STATUS: CLEAR - Normal Healthy Range")
                
            else:
                print("Bad reading. Waveform flatlined.")

    except KeyboardInterrupt:
        print("\nSession terminated.")

if __name__ == '__main__':
    run_live_diagnostic()
    