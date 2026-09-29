"""
SHAP Explainability and Feature Attribution Analysis
Explainable Machine Learning for Predicting Longitudinal Motor Function Outcomes
in Spinal Muscular Atrophy Type 1 and Type 2

Computes:
- Exact TreeExplainer Shapley values
- Global Mean Absolute SHAP Importance (Table VI in manuscript)
- Directional impact and top interactions
- Saves shap_global_ranking_table_VI.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sma_patient_cohort_N350.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "shap_global_ranking_table_VI.csv")

feature_name_map = {
    'Treatment_Delay_mo': 'Treatment Initiation Delay (mo)',
    'Baseline_CMAP_mV': 'Baseline CMAP Amplitude (mV)',
    'Normalized_Base_Score': 'Normalized Baseline Motor Score (%)',
    'SMN2_Copies': 'SMN2 Copy Number',
    'Treatment_Onasemnogene': 'Treatment: Onasemnogene Abeparvovec',
    'Age_Onset_mo': 'Age at Symptom Onset (mo)',
    'NIV_Hours_Daily': 'Daily Baseline NIV Usage (hrs)',
    'Scoliosis_Cobb_deg': 'Scoliosis Cobb Angle (deg)',
    'Nutritional_ZScore': 'Nutritional Z-Score',
    'SMA_Type_Type 2': 'Phenotypic Subtype: Type 2',
    'Treatment_Risdiplam': 'Treatment: Risdiplam',
    'Treatment_Untreated': 'Treatment: Untreated Control'
}

mechanistic_interpretations = {
    'Treatment Initiation Delay (mo)': 'Irreversible spinal motor neuron loss prior to rescue.',
    'Baseline CMAP Amplitude (mV)': 'Functional motor unit pool capable of terminal reinnervation.',
    'Normalized Baseline Motor Score (%)': 'Defines functional headroom; exhibits psychometric ceiling.',
    'SMN2 Copy Number': 'Primary genomic modifier governing endogenous SMN levels.',
    'Treatment: Onasemnogene Abeparvovec': 'Rapid, systemic SMN replacement drives early motor recovery.',
    'Age at Symptom Onset (mo)': 'Surrogate for biological aggressiveness and phenotype type.',
    'Daily Baseline NIV Usage (hrs)': 'Marker of respiratory diaphragmatic failure and debility.',
    'Scoliosis Cobb Angle (deg)': 'Biomechanical restriction impairing axial motor tasks.',
    'Nutritional Z-Score': 'Marker of bulbar feeding failure and systemic cachexia.',
    'Phenotypic Subtype: Type 2': 'Categorical baseline distinction between sitters and non-sitters.',
    'Treatment: Risdiplam': 'Systemic small-molecule SMN2 pre-mRNA splicing modifier.',
    'Treatment: Untreated Control': 'Natural history trajectory of unmitigated neurodegeneration.'
}

def run_shap_analysis():
    print("=" * 70)
    print("RUNNING SHAP GLOBAL ATTRIBUTION & RANKING (TABLE VI)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        'SMN2_Copies', 'Age_Onset_mo', 'Treatment_Delay_mo',
        'Baseline_CMAP_mV', 'Normalized_Base_Score', 'Nutritional_ZScore',
        'NIV_Hours_Daily', 'Scoliosis_Cobb_deg'
    ]
    X = pd.get_dummies(df[feature_cols + ['SMA_Type', 'Treatment']], drop_first=True)
    y = df['Delta_Score_24m'].values

    gbm_path = os.path.join(MODELS_DIR, "gbm_model.pkl")
    shap_path = os.path.join(MODELS_DIR, "shap_values.npy")

    if os.path.exists(gbm_path):
        with open(gbm_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.85, random_state=42)
        model.fit(X, y)

    if os.path.exists(shap_path):
        shap_vals = np.load(shap_path)
    else:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        np.save(shap_path, shap_vals)

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    records = []
    for col, val in zip(X.columns, mean_abs_shap):
        desc = feature_name_map.get(col, col)
        mech = mechanistic_interpretations.get(desc, "Biological covariate in neuromuscular recovery.")
        
        # Determine directional impact
        col_idx = list(X.columns).index(col)
        corr_val = np.corrcoef(X[col].values, shap_vals[:, col_idx])[0, 1]
        if np.isnan(corr_val):
            direction = "Non-linear / Categorical"
        elif corr_val > 0.3:
            direction = "Positive with Higher Values"
        elif corr_val < -0.3:
            direction = "Negative with Higher Values"
        else:
            direction = "Non-linear / Complex"

        records.append({
            'Feature Description': desc,
            'Mean |SHAP| (E[|phi|], pts)': round(val, 2),
            'Directional Impact Trend': direction,
            'Mechanistic Pathophysiological Interpretation': mech
        })

    df_shap = pd.DataFrame(records).sort_values(by='Mean |SHAP| (E[|phi|], pts)', ascending=False).reset_index(drop=True)
    df_shap.to_csv(OUTPUT_PATH, index=False)

    print("\nSHAP GLOBAL RANKING TABLE (TABLE VI):")
    print(df_shap.to_string(index=False))
    print(f"\nSaved SHAP ranking table to: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_shap_analysis()
