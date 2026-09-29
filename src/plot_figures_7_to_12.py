"""
Figure Generator: Figures 7 to 12
Explainable Machine Learning for Predicting Longitudinal Motor Function Outcomes
in Spinal Muscular Atrophy Type 1 and Type 2

Includes:
- Figure 7: Baseline Feature Importance for Longitudinal Motor Function Prediction
- Figure 8: Comparative Predictive Performance of Machine Learning Models
- Figure 9: Agreement Between Predicted and Observed Longitudinal Motor Function Outcomes
- Figure 10: Residual and Prediction Error Analysis of the Final Machine Learning Model
- Figure 11: Global Feature Importance Ranking Using SHAP Explanations
- Figure 12: SHAP Beeswarm Summary of Feature Effects on Longitudinal Motor Function
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sma_patient_cohort_N350.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df_cohort = pd.read_csv(DATA_PATH)

feature_cols = [
    'SMN2_Copies', 'Age_Onset_mo', 'Treatment_Delay_mo',
    'Baseline_CMAP_mV', 'Normalized_Base_Score', 'Nutritional_ZScore',
    'NIV_Hours_Daily', 'Scoliosis_Cobb_deg'
]
X_raw = pd.get_dummies(df_cohort[feature_cols + ['SMA_Type', 'Treatment']], drop_first=True)
y_delta = df_cohort['Delta_Score_24m'].values

# Load or fit GBM model and SHAP values
gbm_path = os.path.join(MODELS_DIR, "gbm_model.pkl")
shap_path = os.path.join(MODELS_DIR, "shap_values.npy")

if os.path.exists(gbm_path):
    with open(gbm_path, "rb") as f:
        gb_model = pickle.load(f)
else:
    gb_model = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.85, random_state=42)
    gb_model.fit(X_raw, y_delta)

if os.path.exists(shap_path):
    shap_values = np.load(shap_path)
    explainer = shap.TreeExplainer(gb_model)
else:
    explainer = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(X_raw)

# Calculate predicted values and residuals if not in dataframe
if 'Predicted_Delta_24m' not in df_cohort.columns:
    df_cohort['Predicted_Delta_24m'] = gb_model.predict(X_raw)
if 'Residual' not in df_cohort.columns:
    df_cohort['Residual'] = df_cohort['Delta_Score_24m'] - df_cohort['Predicted_Delta_24m']

# Visual styling
def apply_pub_style():
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans'],
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.5,
        'figure.titlesize': 14,
        'axes.linewidth': 1.1,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'grid.color': '#CBD5E1',
        'axes.edgecolor': '#334155',
        'figure.autolayout': False,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

apply_pub_style()

PALETTE = {
    'sma1': '#DC2626',
    'sma2': '#2563EB',
    'neutral': '#475569',
    'accent1': '#0D9488',
    'accent2': '#7C3AED',
    'accent3': '#EA580C',
    'nusinersen': '#3B82F6',
    'risdiplam': '#10B981',
    'gene_tx': '#8B5CF6',
    'untreated': '#94A3B8',
    'bg_light': '#F8FAFC',
    'card_edge': '#E2E8F0'
}

def save_figure(fig, fig_num, title):
    fname_png = os.path.join(OUTPUT_DIR, f"Figure_{fig_num:02d}_{title}.png")
    fname_pdf = os.path.join(OUTPUT_DIR, f"Figure_{fig_num:02d}_{title}.pdf")
    fig.savefig(fname_png, dpi=300, bbox_inches='tight')
    fig.savefig(fname_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: Figure {fig_num} -> {os.path.basename(fname_png)}")

feature_name_map = {
    'Treatment_Delay_mo': 'Treatment Delay (mo)',
    'Baseline_CMAP_mV': 'Baseline CMAP (mV)',
    'Normalized_Base_Score': 'Baseline Motor Score (%)',
    'SMN2_Copies': 'SMN2 Copies',
    'Treatment_Onasemnogene': 'Tx: Onasemnogene',
    'Age_Onset_mo': 'Age at Onset (mo)',
    'NIV_Hours_Daily': 'Daily NIV Usage (hrs)',
    'Scoliosis_Cobb_deg': 'Scoliosis Cobb Angle (deg)',
    'Nutritional_ZScore': 'Nutritional Z-Score',
    'SMA_Type_Type 2': 'Subtype: Type 2',
    'Treatment_Risdiplam': 'Tx: Risdiplam',
    'Treatment_Untreated': 'Tx: Untreated Natural History'
}

# -----------------------------------------------------------------------------
# FIGURE 7: Baseline Feature Importance (Permutation & Impurity)
# -----------------------------------------------------------------------------
def plot_figure_7():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Gini / MDI importance
    mdi_imp = gb_model.feature_importances_
    perm_res = permutation_importance(gb_model, X_raw, y_delta, n_repeats=15, random_state=42)
    perm_imp = perm_res.importances_mean

    feat_labels = [feature_name_map.get(col, col) for col in X_raw.columns]
    df_imp = pd.DataFrame({
        'Feature': feat_labels,
        'MDI': mdi_imp,
        'Permutation': perm_imp
    }).sort_values(by='Permutation', ascending=True)

    # Panel A: Permutation Importance
    axes[0].barh(df_imp['Feature'], df_imp['Permutation'], color='#3B82F6', alpha=0.85, edgecolor='none')
    axes[0].set_title("A. Permutation Feature Importance (GBM, 15 Repeats)", fontweight='bold')
    axes[0].set_xlabel("Mean Decrease in R2 Score")
    axes[0].grid(axis='x', ls='--', alpha=0.5)

    # Panel B: Gini / MDI Impurity Decrease
    axes[1].barh(df_imp['Feature'], df_imp['MDI'], color='#10B981', alpha=0.85, edgecolor='none')
    axes[1].set_title("B. Mean Decrease in Impurity (Gini/MDI)", fontweight='bold')
    axes[1].set_xlabel("Relative Importance Weight")
    axes[1].grid(axis='x', ls='--', alpha=0.5)

    save_figure(fig, 7, "Baseline_Feature_Importance")

# -----------------------------------------------------------------------------
# FIGURE 8: Comparative Predictive Performance
# -----------------------------------------------------------------------------
def plot_figure_8():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    models = {
        'GBM': GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.85, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42),
        'SVR (RBF)': SVR(C=5.0, epsilon=0.2),
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'Ridge': Ridge(alpha=1.0, random_state=42)
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    perf_data = []

    for name, m in models.items():
        scores = cross_validate(m, X_raw, y_delta, cv=cv,
                                scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'])
        for r2, rmse, mae in zip(scores['test_r2'], -scores['test_neg_root_mean_squared_error'], -scores['test_neg_mean_absolute_error']):
            perf_data.append({'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae})

    df_perf = pd.DataFrame(perf_data)

    model_colors = ['#DC2626', '#EA580C', '#2563EB', '#7C3AED', '#0D9488', '#64748B']

    # Panel A: R2 Boxplot
    sns.boxplot(data=df_perf, x='Model', y='R2', palette=model_colors, ax=axes[0])
    axes[0].set_title("A. Cross-Validated R2 Score", fontweight='bold')
    axes[0].set_ylabel("R2 Coefficient of Determination")
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].grid(axis='y', ls='--', alpha=0.5)

    # Panel B: RMSE Boxplot
    sns.boxplot(data=df_perf, x='Model', y='RMSE', palette=model_colors, ax=axes[1])
    axes[1].set_title("B. Root Mean Squared Error (RMSE)", fontweight='bold')
    axes[1].set_ylabel("RMSE (Points)")
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].grid(axis='y', ls='--', alpha=0.5)

    # Panel C: MAE Boxplot
    sns.boxplot(data=df_perf, x='Model', y='MAE', palette=model_colors, ax=axes[2])
    axes[2].set_title("C. Mean Absolute Error (MAE)", fontweight='bold')
    axes[2].set_ylabel("MAE (Points)")
    axes[2].tick_params(axis='x', rotation=25)
    axes[2].grid(axis='y', ls='--', alpha=0.5)

    save_figure(fig, 8, "Comparative_ML_Performance")

# -----------------------------------------------------------------------------
# FIGURE 9: Agreement Observed vs Predicted
# -----------------------------------------------------------------------------
def plot_figure_9():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    y_obs = df_cohort['Delta_Score_24m']
    y_pred = df_cohort['Predicted_Delta_24m']

    # Panel A: Scatter with 45 line of identity
    sns.scatterplot(data=df_cohort, x='Delta_Score_24m', y='Predicted_Delta_24m',
                    hue='SMA_Type', palette={'Type 1': PALETTE['sma1'], 'Type 2': PALETTE['sma2']},
                    s=65, alpha=0.85, ax=axes[0])
    lims = [-12, 22]
    axes[0].plot(lims, lims, color='#334155', ls='--', lw=1.8, label='Line of Identity (y = x)')
    m, b = np.polyfit(y_obs, y_pred, 1)
    axes[0].plot(y_obs, m * y_obs + b, color='#DC2626', lw=1.5, label=f'Fit: y = {m:.2f}x + {b:.2f}')
    r2_val = r2_score(y_obs, y_pred)
    pearson_r, _ = stats.pearsonr(y_obs, y_pred)
    axes[0].text(0.05, 0.85, f"R2 = {r2_val:.3f}\nPearson r = {pearson_r:.3f}\nRMSE = {np.sqrt(mean_squared_error(y_obs, y_pred)):.2f} pts",
                 transform=axes[0].transAxes, fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFFF', ec='#CBD5E1'))
    axes[0].set_title("A. Predicted vs. Observed 24-Month Score Gains", fontweight='bold')
    axes[0].set_xlabel("Observed 24-Month Delta Score (Points)")
    axes[0].set_ylabel("Model-Predicted 24-Month Delta Score (Points)")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].legend(loc='lower right')

    # Panel B: Bland-Altman Agreement Plot
    mean_val = (y_obs + y_pred) / 2.0
    diff_val = y_obs - y_pred
    mean_diff = np.mean(diff_val)
    std_diff = np.std(diff_val)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    axes[1].scatter(mean_val, diff_val, color='#475569', alpha=0.7, s=50)
    axes[1].axhline(y=mean_diff, color='#2563EB', lw=2.0, label=f'Mean Bias: {mean_diff:.2f}')
    axes[1].axhline(y=upper_loa, color='#DC2626', ls='--', lw=1.6, label=f'+1.96 SD: {upper_loa:.2f}')
    axes[1].axhline(y=lower_loa, color='#DC2626', ls='--', lw=1.6, label=f'-1.96 SD: {lower_loa:.2f}')
    axes[1].set_title("B. Bland-Altman Agreement Plot", fontweight='bold')
    axes[1].set_xlabel("Mean of Observed & Predicted Delta Scores")
    axes[1].set_ylabel("Difference (Observed - Predicted)")
    axes[1].legend(loc='upper right')
    axes[1].grid(ls='--', alpha=0.5)

    save_figure(fig, 9, "Agreement_Observed_vs_Predicted")

# -----------------------------------------------------------------------------
# FIGURE 10: Residual & Prediction Error Analysis
# -----------------------------------------------------------------------------
def plot_figure_10():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    res = df_cohort['Residual']

    # Panel A: Residual Distribution Histogram & KDE
    sns.histplot(res, kde=True, color='#0D9488', bins=25, ax=axes[0])
    axes[0].axvline(0, color='#DC2626', ls='--', lw=1.5)
    axes[0].set_title("A. Residual Error Distribution", fontweight='bold')
    axes[0].set_xlabel("Prediction Residual (Points)")
    axes[0].set_ylabel("Count")

    # Panel B: Residuals vs Predicted Score
    sns.scatterplot(data=df_cohort, x='Predicted_Delta_24m', y='Residual',
                    hue='SMA_Type', palette={'Type 1': PALETTE['sma1'], 'Type 2': PALETTE['sma2']},
                    alpha=0.75, s=55, ax=axes[1])
    axes[1].axhline(0, color='#DC2626', ls='--', lw=1.5)
    axes[1].set_title("B. Residuals vs. Fitted Values (Homoscedasticity)", fontweight='bold')
    axes[1].set_xlabel("Fitted 24-Month Delta Score")
    axes[1].set_ylabel("Residual (Points)")

    # Panel C: Normal Q-Q Plot
    stats.probplot(res, dist="norm", plot=axes[2])
    axes[2].set_title("C. Normal Q-Q Plot of Residuals", fontweight='bold')
    axes[2].grid(ls='--', alpha=0.5)

    save_figure(fig, 10, "Residual_and_Prediction_Error_Analysis")

# -----------------------------------------------------------------------------
# FIGURE 11: Global SHAP Feature Importance
# -----------------------------------------------------------------------------
def plot_figure_11():
    fig, ax = plt.subplots(figsize=(10, 7.5))
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feat_labels = [feature_name_map.get(col, col) for col in X_raw.columns]

    df_shap = pd.DataFrame({
        'Feature': feat_labels,
        'Mean_SHAP': mean_abs_shap
    }).sort_values(by='Mean_SHAP', ascending=True)

    y_pos = np.arange(len(df_shap))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(df_shap)))
    ax.barh(y_pos, df_shap['Mean_SHAP'], color=colors, edgecolor='none', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_shap['Feature'], fontsize=11)
    ax.set_xlabel("Mean |SHAP Value| (Impact on 24-Month Score Delta, Points)", fontsize=11, fontweight='bold')
    ax.set_title("Global Feature Importance Ranking via SHAP TreeExplainer", fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', ls='--', alpha=0.5)

    for i, v in enumerate(df_shap['Mean_SHAP']):
        ax.text(v + 0.04, i, f"{v:.2f} pts", va='center', fontsize=9.5, fontweight='bold', color='#1E293B')

    save_figure(fig, 11, "Global_SHAP_Feature_Importance")

# -----------------------------------------------------------------------------
# FIGURE 12: SHAP Beeswarm Summary Plot
# -----------------------------------------------------------------------------
def plot_figure_12():
    fig, ax = plt.subplots(figsize=(11, 8.5))
    # Rename columns for readable SHAP plot
    X_display = X_raw.rename(columns=feature_name_map)
    shap.summary_plot(shap_values, X_display, show=False, plot_size=None)
    plt.title("SHAP Beeswarm Summary: Feature Effects on 24-Month Motor Gains", fontsize=13, fontweight='bold', pad=14)
    plt.xlabel("SHAP Value (Impact on 24-Month Score Delta, Points)", fontsize=11, fontweight='bold')
    save_figure(plt.gcf(), 12, "SHAP_Beeswarm_Analysis")

if __name__ == "__main__":
    print("Running plot_figures_7_to_12.py...")
    plot_figure_7()
    plot_figure_8()
    plot_figure_9()
    plot_figure_10()
    plot_figure_11()
    plot_figure_12()
    print("Figures 7 to 12 generated successfully!")
