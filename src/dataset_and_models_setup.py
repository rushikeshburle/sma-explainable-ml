"""
Export Dataset, Train Models, Compute SHAP Values, and Setup Complete Repository
for Spinal Muscular Atrophy (SMA) Type 1 & 2 Explainable Machine Learning Project.
"""

import os
import json
import pickle
import shutil
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

np.random.seed(42)

BASE_DIR = r"C:\Users\karis\.gemini\antigravity\scratch\SMA_ML_Complete_Implementation"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGS_DIR = os.path.join(BASE_DIR, "figures")
SRC_FIGS_DIR = r"C:\Users\karis\.gemini\antigravity\scratch\SMA_ML_Longitudinal_Figures"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. GENERATE COHORT (N=350) EXACT SPECIFICATIONS
# -----------------------------------------------------------------------------
def generate_cohort(n=350):
    n_sma1 = int(n * 0.44) # 154
    n_sma2 = n - n_sma1    # 196

    # SMA Type 1
    type1_smn2 = np.random.choice([2, 3], size=n_sma1, p=[0.85, 0.15])
    type1_onset = np.clip(np.random.gamma(shape=2.5, scale=1.0, size=n_sma1), 0.5, 5.8)
    type1_delay = np.clip(np.random.gamma(shape=2.0, scale=1.1, size=n_sma1), 0.2, 7.0)
    type1_tx_age = type1_onset + type1_delay
    type1_cmap = np.clip(0.3 + 0.3 * (type1_smn2 - 2) + np.random.exponential(scale=0.5, size=n_sma1) - 0.05 * type1_delay, 0.1, 2.8)
    type1_base_chop = np.clip(12 + 6 * type1_smn2 + 4.5 * type1_cmap - 1.8 * type1_delay + np.random.normal(0, 4.0, size=n_sma1), 4, 46)
    type1_nutr = np.random.normal(-1.2, 0.8, size=n_sma1)
    type1_niv = np.clip(np.random.exponential(scale=3.5, size=n_sma1) + 1.2 * (type1_delay > 3.0), 0, 18)
    type1_tx = np.random.choice(['Nusinersen', 'Risdiplam', 'Onasemnogene', 'Untreated'],
                                size=n_sma1, p=[0.42, 0.28, 0.25, 0.05])
    type1_scoliosis = np.clip(np.random.normal(6.0, 3.5, size=n_sma1), 0, 25)

    # SMA Type 2
    type2_smn2 = np.random.choice([3, 4], size=n_sma2, p=[0.82, 0.18])
    type2_onset = np.clip(np.random.gamma(shape=5.0, scale=2.1, size=n_sma2), 6.0, 17.5)
    type2_delay = np.clip(np.random.gamma(shape=3.0, scale=2.0, size=n_sma2), 0.8, 18.0)
    type2_tx_age = type2_onset + type2_delay
    type2_cmap = np.clip(1.2 + 0.5 * (type2_smn2 - 3) + np.random.exponential(scale=0.9, size=n_sma2) - 0.04 * type2_delay, 0.5, 4.8)
    type2_base_hfmse = np.clip(10 + 5 * type2_smn2 + 3.0 * type2_cmap - 0.6 * type2_delay + np.random.normal(0, 5.0, size=n_sma2), 4, 44)
    type2_nutr = np.random.normal(-0.4, 0.9, size=n_sma2)
    type2_niv = np.clip(np.random.exponential(scale=1.5, size=n_sma2), 0, 10)
    type2_tx = np.random.choice(['Nusinersen', 'Risdiplam', 'Onasemnogene', 'Untreated'],
                                size=n_sma2, p=[0.48, 0.38, 0.09, 0.05])
    type2_scoliosis = np.clip(np.random.normal(16.0, 8.0, size=n_sma2), 0, 48)

    df1 = pd.DataFrame({
        'Patient_ID': [f'SMA1_{i+1:03d}' for i in range(n_sma1)],
        'SMA_Type': 'Type 1',
        'SMN2_Copies': type1_smn2,
        'Age_Onset_mo': np.round(type1_onset, 2),
        'Treatment_Delay_mo': np.round(type1_delay, 2),
        'Age_Treatment_Start_mo': np.round(type1_tx_age, 2),
        'Baseline_CMAP_mV': np.round(type1_cmap, 2),
        'Baseline_Motor_Score': np.round(type1_base_chop, 1),
        'Motor_Scale': 'CHOP-INTEND',
        'Normalized_Base_Score': np.round((type1_base_chop / 64.0) * 100.0, 2),
        'Nutritional_ZScore': np.round(type1_nutr, 2),
        'NIV_Hours_Daily': np.round(type1_niv, 1),
        'Scoliosis_Cobb_deg': np.round(type1_scoliosis, 1),
        'Treatment': type1_tx
    })

    df2 = pd.DataFrame({
        'Patient_ID': [f'SMA2_{i+1:03d}' for i in range(n_sma2)],
        'SMA_Type': 'Type 2',
        'SMN2_Copies': type2_smn2,
        'Age_Onset_mo': np.round(type2_onset, 2),
        'Treatment_Delay_mo': np.round(type2_delay, 2),
        'Age_Treatment_Start_mo': np.round(type2_tx_age, 2),
        'Baseline_CMAP_mV': np.round(type2_cmap, 2),
        'Baseline_Motor_Score': np.round(type2_base_hfmse, 1),
        'Motor_Scale': 'HFMSE',
        'Normalized_Base_Score': np.round((type2_base_hfmse / 66.0) * 100.0, 2),
        'Nutritional_ZScore': np.round(type2_nutr, 2),
        'NIV_Hours_Daily': np.round(type2_niv, 1),
        'Scoliosis_Cobb_deg': np.round(type2_scoliosis, 1),
        'Treatment': type2_tx
    })

    df = pd.concat([df1, df2], ignore_index=True)

    tx_multiplier = df['Treatment'].map({
        'Onasemnogene': 1.35,
        'Nusinersen': 1.05,
        'Risdiplam': 1.02,
        'Untreated': -0.45
    })

    delay_penalty = np.exp(-0.12 * df['Treatment_Delay_mo'])
    cmap_benefit = 1.0 + 0.3 * np.log1p(df['Baseline_CMAP_mV'])
    smn_benefit = 1.0 + 0.25 * (df['SMN2_Copies'] - 2)
    ceiling_factor = (100.0 - df['Normalized_Base_Score']) / 100.0

    latent_responsiveness = (
        tx_multiplier * delay_penalty * cmap_benefit * smn_benefit * ceiling_factor
    )

    delta_base = np.where(
        df['SMA_Type'] == 'Type 1',
        14.0 * latent_responsiveness + np.random.normal(0, 3.2, size=len(df)),
        8.5 * latent_responsiveness + np.random.normal(0, 2.5, size=len(df))
    )

    max_scale = np.where(df['SMA_Type'] == 'Type 1', 64.0, 66.0)
    final_raw = np.clip(df['Baseline_Motor_Score'] + delta_base, 0, max_scale)
    actual_delta_24 = final_raw - df['Baseline_Motor_Score']
    df['Delta_Score_24m'] = np.round(actual_delta_24, 2)

    f6 = 0.42 + np.random.normal(0, 0.05, size=len(df))
    f12 = 0.76 + np.random.normal(0, 0.04, size=len(df))
    f18 = 0.90 + np.random.normal(0, 0.03, size=len(df))
    f24 = 1.00

    df['Score_0m'] = df['Baseline_Motor_Score']
    df['Score_6m'] = np.round(np.clip(df['Score_0m'] + df['Delta_Score_24m'] * f6, 0, max_scale), 1)
    df['Score_12m'] = np.round(np.clip(df['Score_0m'] + df['Delta_Score_24m'] * f12, 0, max_scale), 1)
    df['Score_18m'] = np.round(np.clip(df['Score_0m'] + df['Delta_Score_24m'] * f18, 0, max_scale), 1)
    df['Score_24m'] = np.round(np.clip(df['Score_0m'] + df['Delta_Score_24m'] * f24, 0, max_scale), 1)

    df['Delta_Score_6m'] = np.round(df['Score_6m'] - df['Score_0m'], 2)
    df['Delta_Score_12m'] = np.round(df['Score_12m'] - df['Score_0m'], 2)
    df['Delta_Score_18m'] = np.round(df['Score_18m'] - df['Score_0m'], 2)

    df['Is_Responder'] = np.where(
        df['SMA_Type'] == 'Type 1',
        df['Delta_Score_24m'] >= 4.0,
        df['Delta_Score_24m'] >= 3.0
    )

    return df

df_cohort = generate_cohort(350)
csv_path = os.path.join(DATA_DIR, "sma_patient_cohort_N350.csv")
xlsx_path = os.path.join(DATA_DIR, "sma_patient_cohort_N350.xlsx")
df_cohort.to_csv(csv_path, index=False)
df_cohort.to_excel(xlsx_path, index=False, engine='openpyxl')
print(f"Dataset successfully exported:\n - {csv_path}\n - {xlsx_path}")

# -----------------------------------------------------------------------------
# 2. DATA DICTIONARY
# -----------------------------------------------------------------------------
data_dictionary = [
    {"Variable_Name": "Patient_ID", "Domain": "Identifier", "Type": "Categorical", "Unit": "N/A", "Description": "Unique de-identified patient identifier (SMA1_XXX or SMA2_XXX)"},
    {"Variable_Name": "SMA_Type", "Domain": "Phenotypic Subtype", "Type": "Binary", "Unit": "Type 1 / Type 2", "Description": "Clinical classification based on onset age and milestone attainment"},
    {"Variable_Name": "SMN2_Copies", "Domain": "Genomics", "Type": "Discrete Integer", "Unit": "Count (2, 3, 4)", "Description": "SMN2 gene copy number determined via MLPA"},
    {"Variable_Name": "Age_Onset_mo", "Domain": "Chronology", "Type": "Continuous", "Unit": "Months", "Description": "Age at confirmed clinical onset of hypotonia / weakness"},
    {"Variable_Name": "Treatment_Delay_mo", "Domain": "Chronology", "Type": "Continuous", "Unit": "Months", "Description": "Duration from symptom onset to initial DMT administration"},
    {"Variable_Name": "Age_Treatment_Start_mo", "Domain": "Chronology", "Type": "Continuous", "Unit": "Months", "Description": "Patient age at initiation of disease-modifying therapy"},
    {"Variable_Name": "Baseline_CMAP_mV", "Domain": "Neurophysiology", "Type": "Continuous", "Unit": "Millivolts (mV)", "Description": "Baseline ulnar compound muscle action potential amplitude"},
    {"Variable_Name": "Baseline_Motor_Score", "Domain": "Motor Function", "Type": "Continuous", "Unit": "Points", "Description": "Raw baseline score on CHOP-INTEND (Type 1) or HFMSE (Type 2)"},
    {"Variable_Name": "Motor_Scale", "Domain": "Assessment Tool", "Type": "Categorical", "Unit": "Scale Name", "Description": "Standardized scale used (CHOP-INTEND max=64, HFMSE max=66)"},
    {"Variable_Name": "Normalized_Base_Score", "Domain": "Motor Function", "Type": "Continuous", "Unit": "Percent (%)", "Description": "Baseline motor score normalized to maximum scale points"},
    {"Variable_Name": "Nutritional_ZScore", "Domain": "Anthropometry", "Type": "Continuous", "Unit": "SD (Z-score)", "Description": "WHO age-adjusted weight-for-age or BMI Z-score"},
    {"Variable_Name": "NIV_Hours_Daily", "Domain": "Respiratory", "Type": "Continuous", "Unit": "Hours/day", "Description": "Average daily non-invasive ventilation usage"},
    {"Variable_Name": "Scoliosis_Cobb_deg", "Domain": "Orthopedics", "Type": "Continuous", "Unit": "Degrees (deg)", "Description": "Major coronal spinal curvature Cobb angle on radiograph"},
    {"Variable_Name": "Treatment", "Domain": "Therapeutics", "Type": "Categorical", "Unit": "4 Classes", "Description": "DMT administered: Nusinersen, Risdiplam, Onasemnogene, or Untreated"},
    {"Variable_Name": "Score_0m", "Domain": "Longitudinal Outcome", "Type": "Continuous", "Unit": "Points", "Description": "Baseline motor function score"},
    {"Variable_Name": "Score_6m", "Domain": "Longitudinal Outcome", "Type": "Continuous", "Unit": "Points", "Description": "Motor function score at 6 months follow-up"},
    {"Variable_Name": "Score_12m", "Domain": "Longitudinal Outcome", "Type": "Continuous", "Unit": "Points", "Description": "Motor function score at 12 months follow-up"},
    {"Variable_Name": "Score_18m", "Domain": "Longitudinal Outcome", "Type": "Continuous", "Unit": "Points", "Description": "Motor function score at 18 months follow-up"},
    {"Variable_Name": "Score_24m", "Domain": "Longitudinal Outcome", "Type": "Continuous", "Unit": "Points", "Description": "Motor function score at 24 months primary endpoint"},
    {"Variable_Name": "Delta_Score_6m", "Domain": "Longitudinal Change", "Type": "Continuous", "Unit": "Points", "Description": "Change in motor score from baseline to 6 months"},
    {"Variable_Name": "Delta_Score_12m", "Domain": "Longitudinal Change", "Type": "Continuous", "Unit": "Points", "Description": "Change in motor score from baseline to 12 months"},
    {"Variable_Name": "Delta_Score_18m", "Domain": "Longitudinal Change", "Type": "Continuous", "Unit": "Points", "Description": "Change in motor score from baseline to 18 months"},
    {"Variable_Name": "Delta_Score_24m", "Domain": "Primary Target", "Type": "Continuous", "Unit": "Points", "Description": "Primary outcome: change in motor score from baseline to 24 months"},
    {"Variable_Name": "Is_Responder", "Domain": "Clinical Response", "Type": "Binary", "Unit": "True / False", "Description": "Attainment of MCID (CHOP >= +4 pts for Type 1, HFMSE >= +3 pts for Type 2)"}
]

df_dict = pd.DataFrame(data_dictionary)
dict_csv = os.path.join(DATA_DIR, "dataset_data_dictionary.csv")
dict_json = os.path.join(DATA_DIR, "dataset_data_dictionary.json")
df_dict.to_csv(dict_csv, index=False)
with open(dict_json, "w", encoding="utf-8") as f:
    json.dump(data_dictionary, f, indent=4)
print(f"Data dictionary exported:\n - {dict_csv}\n - {dict_json}")

# -----------------------------------------------------------------------------
# 3. TRAIN ML MODELS & EXPORT ARTIFACTS
# -----------------------------------------------------------------------------
feature_cols = [
    'SMN2_Copies', 'Age_Onset_mo', 'Treatment_Delay_mo',
    'Baseline_CMAP_mV', 'Normalized_Base_Score', 'Nutritional_ZScore',
    'NIV_Hours_Daily', 'Scoliosis_Cobb_deg'
]
X_raw = pd.get_dummies(df_cohort[feature_cols + ['SMA_Type', 'Treatment']], drop_first=True)
y_delta = df_cohort['Delta_Score_24m'].values

# Fit Gradient Boosting Regressor (primary explainable model)
gb_model = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.85, random_state=42)
gb_model.fit(X_raw, y_delta)

# Fit Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42)
rf_model.fit(X_raw, y_delta)

# TreeExplainer SHAP
explainer = shap.TreeExplainer(gb_model)
shap_values = explainer.shap_values(X_raw)

# Save models and SHAP artifacts
with open(os.path.join(MODELS_DIR, "gbm_model.pkl"), "wb") as f:
    pickle.dump(gb_model, f)

with open(os.path.join(MODELS_DIR, "rf_model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)

np.save(os.path.join(MODELS_DIR, "shap_values.npy"), shap_values)

with open(os.path.join(MODELS_DIR, "feature_names.json"), "w", encoding="utf-8") as f:
    json.dump(list(X_raw.columns), f, indent=4)

print("Trained models and SHAP matrices saved to models/ directory.")

# -----------------------------------------------------------------------------
# 4. COPY 18 PUBLICATION FIGURES TO FIGURES/ DIRECTORY
# -----------------------------------------------------------------------------
if os.path.exists(SRC_FIGS_DIR):
    copied_count = 0
    for f in os.listdir(SRC_FIGS_DIR):
        if f.endswith(('.png', '.pdf')):
            src_f = os.path.join(SRC_FIGS_DIR, f)
            dst_f = os.path.join(FIGS_DIR, f)
            shutil.copy2(src_f, dst_f)
            copied_count += 1
    print(f"Copied {copied_count} figure files (PNG & PDF) to {FIGS_DIR}")

print("Repository setup script completed successfully!")
