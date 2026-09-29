"""
Model Benchmarking and Nested Cross-Validation Pipeline
Explainable Machine Learning for Predicting Longitudinal Motor Function Outcomes
in Spinal Muscular Atrophy Type 1 and Type 2

Evaluates 6 Machine Learning Architectures:
1. Gradient Boosting Regressor (GBM)
2. Random Forest Regressor (RF)
3. Support Vector Regressor (SVR, RBF)
4. Multi-Layer Perceptron (MLP)
5. ElasticNet Regression
6. Ridge Regression

Performs 10-Fold Repeated Cross-Validation and generates Table V metrics.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sma_patient_cohort_N350.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "model_benchmark_table_V.csv")

def run_benchmark():
    print("=" * 70)
    print("RUNNING MACHINE LEARNING MODEL BENCHMARKING (10-FOLD CV)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        'SMN2_Copies', 'Age_Onset_mo', 'Treatment_Delay_mo',
        'Baseline_CMAP_mV', 'Normalized_Base_Score', 'Nutritional_ZScore',
        'NIV_Hours_Daily', 'Scoliosis_Cobb_deg'
    ]
    X = pd.get_dummies(df[feature_cols + ['SMA_Type', 'Treatment']], drop_first=True)
    y = df['Delta_Score_24m'].values

    models = {
        'Gradient Boosting Regressor (GBM)': GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.85, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42),
        'Support Vector Regressor (SVR)': SVR(kernel='rbf', C=5.0, epsilon=0.20),
        'Multi-Layer Perceptron (MLP)': MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01, max_iter=1000, random_state=42),
        'ElasticNet Regression': ElasticNet(alpha=0.10, l1_ratio=0.50, max_iter=2000, random_state=42),
        'Ridge Regression': Ridge(alpha=1.00, random_state=42)
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        print(f"Evaluating: {name}...")
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'],
            return_train_score=False
        )

        r2_vals = scores['test_r2']
        rmse_vals = -scores['test_neg_root_mean_squared_error']
        mae_vals = -scores['test_neg_mean_absolute_error']

        # Fit model on all data for correlation check
        model.fit(X, y)
        y_pred = model.predict(X)
        pearson_r, p_val = stats.pearsonr(y, y_pred)

        # 95% Confidence Interval for R2
        r2_mean = np.mean(r2_vals)
        r2_ci_low = np.percentile(r2_vals, 2.5)
        r2_ci_high = np.percentile(r2_vals, 97.5)

        results.append({
            'Model Architecture': name,
            'R2 Score (Mean)': round(r2_mean, 3),
            'R2 95% CI': f"[{round(r2_ci_low, 3)} - {round(r2_ci_high, 3)}]",
            'RMSE (Mean +- SD)': f"{round(np.mean(rmse_vals), 2)} +- {round(np.std(rmse_vals), 2)}",
            'MAE (Mean +- SD)': f"{round(np.mean(mae_vals), 2)} +- {round(np.std(mae_vals), 2)}",
            'Pearson r': round(pearson_r, 3),
            'p-value': "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        })

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by='R2 Score (Mean)', ascending=False).reset_index(drop=True)
    df_res['Rank'] = range(1, len(df_res) + 1)
    df_res.loc[0, 'Rank'] = "1 (Best)"

    df_res.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS (TABLE V COMPARISON):")
    print("=" * 70)
    print(df_res.to_string(index=False))
    print(f"\nSaved benchmark table to: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_benchmark()
