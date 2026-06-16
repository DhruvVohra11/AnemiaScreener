import os
import pandas as pd
from pathlib import Path

# --- MASTER CONFIGURATION ---
# Update these paths to match your exact metadata filenames!
ITALY_SUBJECT_FILE = "italy_subject_info.csv" 
INDIA_SUBJECT_FILE = "india_subject_info.csv"
ROOT_IMAGE_DIR = "data_images/"

# Column names inside your CSV files
SUBJECT_ID_COL = "Subject_ID"  # The key linking to the folder names
HB_G_L_COL = "Hb_Value"        # The raw lab Hb column (in g/L)
# ----------------------------

def build_unified_manifest():
    print("--- BOOTING UNIFIED COHORT ETL PIPELINE ---")
    
    # 1. Load and combine the separate metadata files
    combined_rows = []
    
    # Process Italy Metadata
    if os.path.exists(ITALY_SUBJECT_FILE):
        print(f"Loading Italy metadata from {ITALY_SUBJECT_FILE}...")
        df_italy = pd.read_csv(ITALY_SUBJECT_FILE, encoding='latin1')
        df_italy['Cohort_Origin'] = 'Italy'
        combined_rows.append(df_italy)
    else:
        print(f"[WARNING] Could not find Italy subject file: {ITALY_SUBJECT_FILE}")
        
    # Process India Metadata
    if os.path.exists(INDIA_SUBJECT_FILE):
        print(f"Loading India metadata from {INDIA_SUBJECT_FILE}...")
        df_india = pd.read_csv(INDIA_SUBJECT_FILE, encoding='latin1')
        df_india['Cohort_Origin'] = 'India'
        combined_rows.append(df_india)
    else:
        print(f"[WARNING] Could not find India subject file: {INDIA_SUBJECT_FILE}")

    if not combined_rows:
        print("[ERROR] No metadata files loaded! Exiting pipeline.")
        return None

    # Concatenate both cohorts into one master dataframe
    df_master_base = pd.concat(combined_rows, ignore_index=True)
    df_master_base.rename(columns=lambda x: x.strip(), inplace=True) # Strip accidental whitespace
    
    # 2. Convert units: g/L to g/dL
    if HB_G_L_COL not in df_master_base.columns:
        print(f"\n[ERROR] Key '{HB_G_L_COL}' not found in the combined metadata columns.")
        print(f"Available columns are: {df_master_base.columns.tolist()}")
        return None
        
    print(f"Converting global target column '{HB_G_L_COL}' from g/L to g/dL...")
    df_master_base['Hb_g_dL'] = df_master_base[HB_G_L_COL] / 10.0
    
    # Standardize Subject IDs as clean strings for directory cross-referencing
    df_master_base[SUBJECT_ID_COL] = df_master_base[SUBJECT_ID_COL].astype(str).str.strip()
    
    # Create the global lookup map: { 'Subject_001': 14.2 }
    hb_lookup = dict(zip(df_master_base[SUBJECT_ID_COL], df_master_base['Hb_g_dL']))
    
    # 3. Walk the nested folder structures
    parsed_records = []
    root_path = Path(ROOT_IMAGE_DIR)
    
    print(f"\nScanning image directories inside '{ROOT_IMAGE_DIR}'...")
    for country_folder in root_path.iterdir():
        if not country_folder.is_dir():
            continue
            
        print(f" -> Accessing cohort sub-tree: {country_folder.name}")
        
        for patient_folder in country_folder.iterdir():
            if not patient_folder.is_dir():
                continue
                
            patient_id = patient_folder.name.strip()
            
            # Cross-reference folder name with our global master lookup map
            if patient_id not in hb_lookup:
                continue
                
            patient_hb = hb_lookup[patient_id]
            
            # Extract close-up conjunctiva frames (.png), skip full eye photos (.jpg)
            for image_file in patient_folder.iterdir():
                if image_file.suffix.lower() == '.png':
                    parsed_records.append({
                        "absolute_image_path": str(image_file.resolve()),
                        "subject_id": patient_id,
                        "origin_cohort": country_folder.name,
                        "hb_target": patient_hb
                    })

    # 4. Compile and Export Final Manifest Matrix
    df_manifest = pd.DataFrame(parsed_records)
    
    if df_manifest.empty:
        print("\n[ERROR] Compiled manifest contains 0 samples. Verify folder names match Subject IDs.")
        return None
        
    print("\n--- UNIFIED DATASET PARSING COMPLETE ---")
    print(f"Total close-up conjunctiva frames compiled: {len(df_manifest)}")
    print(f"Unique patient systems matched: {df_manifest['subject_id'].nunique()}")
    print("\nVisual data distribution by country layer:")
    print(df_manifest['origin_cohort'].value_counts())
    
    OUTPUT_CSV = "cnn_training_manifest.csv"
    df_manifest.to_csv(OUTPUT_CSV, index=False)
    print(f"\nGlobal training manifest successfully saved to: {OUTPUT_CSV}")
    return df_manifest

if __name__ == '__main__':
    build_unified_manifest()