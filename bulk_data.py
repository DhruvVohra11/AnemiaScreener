import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# --- MASTER CONFIGURATION ---
BASELINE_FILE = "Subject_Info.csv" 
DATA_DIR = "data_csv/" # Ensure this folder name matches your directory exactly!
OUTPUT_FILE = "comprehensive_ai_training_data.csv"

# Update this to match the exact name of your Hemoglobin column in Subject_Info.csv
# Based on your other columns, it might be spelled exactly "Hb(g/L)"
HB_COLUMN_NAME = "Hb_Value" 
# ----------------------------

def build_mega_dataset():
    print("--- BOOTING MULTI-SPECTRAL CLINICAL ETL PIPELINE ---")
    
    # 1. Load the Baseline "Answer Key" with the correct encoding
    print(f"Loading Ground Truth from: {BASELINE_FILE}")
    try:
        df_base = pd.read_csv(BASELINE_FILE, encoding='latin1')
        df_base.rename(columns=lambda x: x.strip(), inplace=True)
    except Exception as e:
        print(f"Error loading baseline file: {e}")
        return
    
    # Check if the Hb column exists
    if HB_COLUMN_NAME not in df_base.columns:
        print(f"\n[ERROR] Could not find column '{HB_COLUMN_NAME}' in {BASELINE_FILE}.")
        print("Available columns are:", df_base.columns.tolist())
        print("Please open the script and update HB_COLUMN_NAME to match.")
        return
        
    # Convert g/L to g/dL
    df_base['Hb_g_dL'] = df_base[HB_COLUMN_NAME] / 10.0 
    
    master_dataset = []
    
    # 2. Find all the patient files
    # If your files end in uppercase .CSV, change this to *.CSV
    patient_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if len(patient_files) == 0:
        print(f"\n[ERROR] Found 0 files in folder '{DATA_DIR}'.")
        print("Current directory contents:")
        print(os.listdir('.'))
        print(f"Make sure the folder containing the 252 files is in this directory and matches DATA_DIR.")
        return
        
    print(f"Found {len(patient_files)} patient telemetry files. Beginning extraction...")
    
    # 3. Loop through every single file
    for filepath in patient_files:
        filename = os.path.basename(filepath)
        
        # Extract the Subject_ID from the filename (Assuming file is named like '101.csv')
        try:
            subject_id = int(filename.split('.')[0])
        except ValueError:
            continue
            
        # Look up this patient's true Hemoglobin
        patient_row = df_base[df_base['Subject_ID'] == subject_id]
        if patient_row.empty:
            continue
            
        true_hb = patient_row['Hb_g_dL'].values[0]
        
        # 4. Load the raw multi-spectral data
        try:
            df_raw = pd.read_csv(filepath)
            w_660 = df_raw["660nm"].values
            w_730 = df_raw["730nm"].values
            w_850 = df_raw["850nm"].values
            w_940 = df_raw["940nm"].values
        except KeyError as e:
            print(f"[ERROR] Missing expected column headers in {filename}. Skpping. Error: {e}")
            continue
            
        # Slice the continuous wave into 200-sample (4-second) windows
        total_samples = len(w_660)
        window_size = 200
        step_size = 100 # Sliding window overlap
        
        valid_windows = 0
        for i in range(0, total_samples - window_size, step_size):
            # Extract current window slices
            r660 = w_660[i : i + window_size]
            r730 = w_730[i : i + window_size]
            r850 = w_850[i : i + window_size]
            r940 = w_940[i : i + window_size]
            
            # Basic DC/AC Calculations
            dc_660, ac_660 = float(np.mean(r660)), float(np.max(r660) - np.min(r660))
            dc_730, ac_730 = float(np.mean(r730)), float(np.max(r730) - np.min(r730))
            dc_850, ac_850 = float(np.mean(r850)), float(np.max(r850) - np.min(r850))
            dc_940, ac_940 = float(np.mean(r940)), float(np.max(r940) - np.min(r940))
            
            if 0 in [dc_660, dc_730, dc_850, dc_940]: continue
            
            # Motion Guardrail using the primary 660nm channel Kurtosis
            kurt_660 = float(kurtosis(r660))
            if kurt_660 > 15.0 or kurt_660 < -5.0:
                continue 
                
            # Feature Engineering
            # Standard R-Ratio using core Red (660nm) and core Infrared (940nm)
            r_ratio = (ac_660 / dc_660) / (ac_940 / dc_940)
            
            # Pulsatility Indices
            pi_660 = (ac_660 / dc_660) * 100
            pi_940 = (ac_940 / dc_940) * 100
            
            # Waveform Morphological Features
            skew_660 = float(skew(r660))
            
            master_dataset.append([
                subject_id, true_hb, 
                dc_660, ac_660, pi_660, skew_660, kurt_660,
                dc_730, ac_730,
                dc_850, ac_850,
                dc_940, ac_940, pi_940,
                r_ratio
            ])
            valid_windows += 1
            
        print(f"Processed Subject {subject_id}: Extracted {valid_windows} pristine windows.")

    # 5. Export the AI-Ready Matrix
    print("\n--- DATA MATRIX COMPILED ---")
    df_final = pd.DataFrame(master_dataset, columns=[
        "Subject_ID", "Hb_Level",
        "dc_660", "ac_660", "pi_660", "skewness_660", "kurtosis_660",
        "dc_730", "ac_730",
        "dc_850", "ac_850",
        "dc_940", "ac_940", "pi_940",
        "r_ratio"
    ])
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS! AI Training Data saved to {OUTPUT_FILE}")
    print(f"Total Rows for Neural Network: {len(df_final)}")

if __name__ == '__main__':
    build_mega_dataset()