"""
Figure Generator: Figures 13 to 18
Explainable Machine Learning for Predicting Longitudinal Motor Function Outcomes
in Spinal Muscular Atrophy Type 1 and Type 2

Includes:
- Figure 13: SHAP Partial Dependence and Feature Interaction Landscapes
- Figure 14: Patient-Specific Waterfall Explanations for Divergent Clinical Trajectories
- Figure 15: Baseline Motor Function versus Longitudinal Outcomes Across Clinical Subgroups
- Figure 16: Model Performance and Calibration Across SMA Clinical Subgroups
- Figure 17: Treatment-Stratified Motor Function Trajectories and Model Predictions
- Figure 18: Cross-Validation Stability, Learning Curves, and Robustness Analysis
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, learning_curve, cross_val_score
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
# FIGURE 13: SHAP Partial Dependence & Feature Interactions
# -----------------------------------------------------------------------------
def plot_figure_13():
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    col_idx_delay = list(X_raw.columns).index('Treatment_Delay_mo')
    col_idx_cmap = list(X_raw.columns).index('Baseline_CMAP_mV')
    col_idx_base = list(X_raw.columns).index('Normalized_Base_Score')
    col_idx_smn2 = list(X_raw.columns).index('SMN2_Copies')

    # Panel A: Delay vs SHAP(Delay) colored by CMAP
    sc1 = axes[0, 0].scatter(X_raw['Treatment_Delay_mo'], shap_values[:, col_idx_delay],
                             c=X_raw['Baseline_CMAP_mV'], cmap='viridis', s=45, alpha=0.85)
    axes[0, 0].set_title("A. SHAP Dependence: Treatment Delay x Baseline CMAP", fontweight='bold')
    axes[0, 0].set_xlabel("Treatment Initiation Delay (months)")
    axes[0, 0].set_ylabel("SHAP Value for Treatment Delay")
    cbar1 = plt.colorbar(sc1, ax=axes[0, 0])
    cbar1.set_label("Baseline CMAP (mV)")

    # Panel B: Baseline Score vs SHAP colored by SMN2
    sc2 = axes[0, 1].scatter(X_raw['Normalized_Base_Score'], shap_values[:, col_idx_base],
                             c=X_raw['SMN2_Copies'], cmap='coolwarm', s=45, alpha=0.85)
    axes[0, 1].set_title("B. SHAP Dependence: Baseline Score x SMN2 Copies", fontweight='bold')
    axes[0, 1].set_xlabel("Normalized Baseline Score (%)")
    axes[0, 1].set_ylabel("SHAP Value for Baseline Score")
    cbar2 = plt.colorbar(sc2, ax=axes[0, 1])
    cbar2.set_label("SMN2 Copy Count")

    # Panel C: CMAP vs SHAP(CMAP) colored by Delay
    sc3 = axes[1, 0].scatter(X_raw['Baseline_CMAP_mV'], shap_values[:, col_idx_cmap],
                             c=X_raw['Treatment_Delay_mo'], cmap='plasma', s=45, alpha=0.85)
    axes[1, 0].set_title("C. SHAP Dependence: Baseline CMAP x Treatment Delay", fontweight='bold')
    axes[1, 0].set_xlabel("Baseline CMAP Amplitude (mV)")
    axes[1, 0].set_ylabel("SHAP Value for Baseline CMAP")
    cbar3 = plt.colorbar(sc3, ax=axes[1, 0])
    cbar3.set_label("Treatment Delay (mo)")

    # Panel D: SMN2 Copies vs SHAP(SMN2) colored by Onset Age
    sc4 = axes[1, 1].scatter(X_raw['SMN2_Copies'] + np.random.normal(0, 0.04, size=len(X_raw)),
                             shap_values[:, col_idx_smn2],
                             c=X_raw['Age_Onset_mo'], cmap='cividis', s=45, alpha=0.85)
    axes[1, 1].set_title("D. SHAP Dependence: SMN2 Copies x Age at Onset", fontweight='bold')
    axes[1, 1].set_xlabel("SMN2 Copy Count")
    axes[1, 1].set_ylabel("SHAP Value for SMN2 Copies")
    axes[1, 1].set_xticks([2, 3, 4])
    cbar4 = plt.colorbar(sc4, ax=axes[1, 1])
    cbar4.set_label("Onset Age (mo)")

    save_figure(fig, 13, "SHAP_Dependence_Analysis")

# -----------------------------------------------------------------------------
# FIGURE 14: Patient-Level Waterfall Explanations
# -----------------------------------------------------------------------------
def plot_figure_14():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    base_val = float(np.squeeze(explainer.expected_value))

    # Patient A: Exceptional Responder
    idx_resp = df_cohort[(df_cohort['SMA_Type'] == 'Type 1') & (df_cohort['Delta_Score_24m'] > 12.0)].index[0]
    # Patient B: Poor Responder (delayed treatment)
    idx_poor = df_cohort[(df_cohort['SMA_Type'] == 'Type 1') & (df_cohort['Delta_Score_24m'] < 2.0)].index[0]

    for ax, p_idx, title in zip(axes, [idx_resp, idx_poor],
                                ["A. Exceptional Responder (Early Tx, High CMAP)",
                                 "B. Suboptimal Responder (Delayed Tx, Low CMAP)"]):
        p_row = X_raw.iloc[p_idx]
        p_shap = shap_values[p_idx]
        pred_val = float(base_val + np.sum(p_shap))
        obs_val = float(y_delta[p_idx])

        sorted_idx = np.argsort(np.abs(p_shap))[-8:]
        top_names = [feature_name_map.get(X_raw.columns[i], X_raw.columns[i]) for i in sorted_idx]
        top_shap = p_shap[sorted_idx]
        top_vals = [f"={p_row.iloc[i]:.1f}" if isinstance(p_row.iloc[i], (int, float)) else "" for i in sorted_idx]
        labels = [f"{n} ({v})" if v else n for n, v in zip(top_names, top_vals)]

        bar_colors = ['#10B981' if v >= 0 else '#EF4444' for v in top_shap]
        ax.barh(labels, top_shap, color=bar_colors, edgecolor='none', height=0.65)
        ax.axvline(0, color='#334155', lw=1.2)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel("SHAP Value (Contribution to Delta Score, Points)")
        ax.text(0.04, 0.92, f"Base Value E[f(x)] = {base_val:.2f} pts\nPredicted Delta = {pred_val:.2f} pts\nObserved Delta = {obs_val:.2f} pts",
                transform=ax.transAxes, fontsize=10.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFFF', ec='#CBD5E1'))
        ax.grid(axis='x', ls='--', alpha=0.5)

    save_figure(fig, 14, "Patient_Level_SHAP_Waterfall")

# -----------------------------------------------------------------------------
# FIGURE 15: Baseline Motor vs Longitudinal Outcomes
# -----------------------------------------------------------------------------
def plot_figure_15():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel A: Type 1 Baseline vs 24m Gain
    df1 = df_cohort[df_cohort['SMA_Type'] == 'Type 1']
    sns.scatterplot(data=df1, x='Baseline_Motor_Score', y='Delta_Score_24m',
                    hue='Treatment', size='Baseline_CMAP_mV', sizes=(40, 160), alpha=0.85, ax=axes[0])
    axes[0].set_title("A. SMA Type 1: Baseline CHOP vs. 24-Month Gain", fontweight='bold')
    axes[0].set_xlabel("Baseline CHOP-INTEND Score (Points)")
    axes[0].set_ylabel("24-Month CHOP-INTEND Change (Points)")
    axes[0].axhline(4.0, color='#DC2626', ls='--', label='MCID (+4 pts)')
    axes[0].legend(loc='lower left', fontsize=8.5)

    # Panel B: Type 2 Baseline vs 24m Gain
    df2 = df_cohort[df_cohort['SMA_Type'] == 'Type 2']
    sns.scatterplot(data=df2, x='Baseline_Motor_Score', y='Delta_Score_24m',
                    hue='Treatment', size='Baseline_CMAP_mV', sizes=(40, 160), alpha=0.85, ax=axes[1])
    axes[1].set_title("B. SMA Type 2: Baseline HFMSE vs. 24-Month Gain", fontweight='bold')
    axes[1].set_xlabel("Baseline HFMSE Score (Points)")
    axes[1].set_ylabel("24-Month HFMSE Change (Points)")
    axes[1].axhline(3.0, color='#2563EB', ls='--', label='MCID (+3 pts)')
    axes[1].legend(loc='lower left', fontsize=8.5)

    save_figure(fig, 15, "Baseline_vs_Longitudinal_Outcomes")

# -----------------------------------------------------------------------------
# FIGURE 16: Subgroup Performance & Calibration
# -----------------------------------------------------------------------------
def plot_figure_16():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Subgroups: Type 1 vs Type 2, SMN2 Copies, Treatment
    subgroups = [
        ('Type 1', df_cohort['SMA_Type'] == 'Type 1'),
        ('Type 2', df_cohort['SMA_Type'] == 'Type 2'),
        ('SMN2: 2c', df_cohort['SMN2_Copies'] == 2),
        ('SMN2: 3c', df_cohort['SMN2_Copies'] == 3),
        ('SMN2: 4c', df_cohort['SMN2_Copies'] == 4),
        ('Nusinersen', df_cohort['Treatment'] == 'Nusinersen'),
        ('Risdiplam', df_cohort['Treatment'] == 'Risdiplam'),
        ('Gene Tx', df_cohort['Treatment'] == 'Onasemnogene')
    ]

    names, r2s, rmses, maes = [], [], [], []
    for name, mask in subgroups:
        sub_df = df_cohort[mask]
        obs = sub_df['Delta_Score_24m']
        pred = sub_df['Predicted_Delta_24m']
        names.append(name)
        r2s.append(r2_score(obs, pred))
        rmses.append(np.sqrt(mean_squared_error(obs, pred)))
        maes.append(mean_absolute_error(obs, pred))

    y_pos = np.arange(len(names))
    axes[0].barh(y_pos, r2s, color='#3B82F6', alpha=0.85)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names)
    axes[0].set_title("A. R2 Across Subgroups", fontweight='bold')
    axes[0].set_xlabel("R2 Score")
    axes[0].set_xlim(0, 1.0)
    axes[0].grid(axis='x', ls='--', alpha=0.5)

    axes[1].barh(y_pos, rmses, color='#EF4444', alpha=0.85)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    axes[1].set_title("B. RMSE Across Subgroups", fontweight='bold')
    axes[1].set_xlabel("RMSE (Points)")
    axes[1].grid(axis='x', ls='--', alpha=0.5)

    axes[2].barh(y_pos, maes, color='#10B981', alpha=0.85)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([])
    axes[2].set_title("C. MAE Across Subgroups", fontweight='bold')
    axes[2].set_xlabel("MAE (Points)")
    axes[2].grid(axis='x', ls='--', alpha=0.5)

    save_figure(fig, 16, "Subgroup_Performance")

# -----------------------------------------------------------------------------
# FIGURE 17: Treatment-Stratified Trajectories
# -----------------------------------------------------------------------------
def plot_figure_17():
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    tx_list = ['Onasemnogene', 'Nusinersen', 'Risdiplam', 'Untreated']
    tx_colors = {'Onasemnogene': '#8B5CF6', 'Nusinersen': '#3B82F6', 'Risdiplam': '#10B981', 'Untreated': '#94A3B8'}

    timepoints = [0, 6, 12, 18, 24]
    time_labels = ['0m', '6m', '12m', '18m', '24m']

    for i, tx in enumerate(tx_list):
        sub_df = df_cohort[df_cohort['Treatment'] == tx]
        mean_vals = [sub_df[f'Score_{m}m'].mean() for m in timepoints]
        std_vals = [sub_df[f'Score_{m}m'].std() for m in timepoints]

        axes[i].errorbar(timepoints, mean_vals, yerr=std_vals, color=tx_colors[tx],
                         fmt='-o', lw=2.5, markersize=7, capsize=4, label='Observed')
        axes[i].set_title(f"{tx} (n={len(sub_df)})", fontweight='bold', fontsize=12)
        axes[i].set_xlabel("Time (Months)")
        axes[i].set_xticks(timepoints)
        axes[i].set_xticklabels(time_labels)
        axes[i].grid(ls='--', alpha=0.5)
        if i == 0:
            axes[i].set_ylabel("Motor Score (Points)")

    plt.suptitle("Treatment-Stratified Longitudinal Trajectories (Mean +- SD)", fontsize=14, fontweight='bold', y=1.03)
    save_figure(fig, 17, "Treatment_Stratified_Analysis")

# -----------------------------------------------------------------------------
# FIGURE 18: Cross-Validation Stability & Learning Curves
# -----------------------------------------------------------------------------
def plot_figure_18():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Learning Curves
    train_sizes, train_scores, test_scores = learning_curve(
        gb_model, X_raw, y_delta, cv=5, train_sizes=np.linspace(0.15, 1.0, 8),
        scoring='r2', random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    axes[0].plot(train_sizes, train_mean, 'o-', color='#DC2626', lw=2.0, label='Training Score')
    axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='#DC2626', alpha=0.15)
    axes[0].plot(train_sizes, test_mean, 's-', color='#2563EB', lw=2.0, label='Validation Score (5-Fold CV)')
    axes[0].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='#2563EB', alpha=0.15)
    axes[0].set_title("A. Gradient Boosting Regressor Learning Curves", fontweight='bold')
    axes[0].set_xlabel("Training Sample Size")
    axes[0].set_ylabel("R2 Score")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend(loc='lower right')
    axes[0].grid(ls='--', alpha=0.5)

    # Panel B: 10-Fold CV Stability Across Random Seeds
    seeds = [7, 13, 21, 42, 88, 99, 123, 256, 512, 777]
    seed_scores = []
    for s in seeds:
        cv_s = KFold(n_splits=10, shuffle=True, random_state=s)
        scores = cross_val_score(gb_model, X_raw, y_delta, cv=cv_s, scoring='r2')
        seed_scores.append(scores)

    axes[1].boxplot(seed_scores, labels=[f"Seed {s}" for s in seeds])
    axes[1].set_title("B. 10-Fold CV Score Distribution Across 10 Random Seeds", fontweight='bold')
    axes[1].set_xlabel("Monte Carlo Random Split Seed")
    axes[1].set_ylabel("R2 Score")
    axes[1].tick_params(axis='x', rotation=35)
    axes[1].grid(ls='--', alpha=0.5)

    save_figure(fig, 18, "CV_Stability_and_Robustness")

if __name__ == "__main__":
    print("Running plot_figures_13_to_18.py...")
    plot_figure_13()
    plot_figure_14()
    plot_figure_15()
    plot_figure_16()
    plot_figure_17()
    plot_figure_18()
    print("Figures 13 to 18 generated successfully!")
