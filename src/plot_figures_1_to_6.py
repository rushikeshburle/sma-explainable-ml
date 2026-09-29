"""
Figure Generator: Figures 1 to 6
Explainable Machine Learning for Predicting Longitudinal Motor Function Outcomes
in Spinal Muscular Atrophy Type 1 and Type 2

Includes:
- Figure 1: Clinical and Pathophysiological Overview of SMA Type 1 and Type 2 (14pt headings, 12pt matter)
- Figure 2: Study Design and Explainable Machine Learning Workflow (14pt headings, 12pt matter)
- Figure 3: Baseline Clinical, Genetic, and Motor Function Characteristics
- Figure 4: Correlation Heatmap of Baseline Clinical and Genetic Features
- Figure 5: Longitudinal Trajectories of CHOP-INTEND and HFMSE Motor Function Scores
- Figure 6: Longitudinal Changes in Motor Function From Baseline and Responder Proportions
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sma_patient_cohort_N350.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df_cohort = pd.read_csv(DATA_PATH)

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

# -----------------------------------------------------------------------------
# FIGURE 1: 14pt Headings, 12pt Matter Premium Architecture
# -----------------------------------------------------------------------------
def plot_figure_1():
    fig = plt.figure(figsize=(24, 16.0), facecolor='#F8FAFC')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#F8FAFC')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 97.4, "Clinical and Pathophysiological Overview of Spinal Muscular Atrophy Type 1 and Type 2",
            ha='center', va='center', fontsize=20, fontweight='bold', color='#0F172A')
    ax.text(50, 95.5, "Genomic Basis, Splicing Mechanics, Molecular Motor Unit Degeneration, and Clinical Phenotypes",
            ha='center', va='center', fontsize=13, color='#475569', style='italic')

    # Panel A: Genomic Architecture
    card_a = patches.FancyBboxPatch((2, 50), 46, 43, boxstyle="round,pad=0.8",
                                     fc='#FFFFFF', ec='#CBD5E1', lw=1.8)
    ax.add_patch(card_a)
    head_a = patches.Rectangle((2, 88.5), 46, 4.5, fc='#EFF6FF', ec='none')
    ax.add_patch(head_a)
    ax.text(3.5, 90.7, "A. Genomic Architecture & Splicing Mechanics (SMN1 vs. SMN2)",
            fontsize=14, fontweight='bold', color='#1E3A8A', va='center')

    # SMN1 block
    smn1_box = patches.FancyBboxPatch((4, 70), 42, 16.5, boxstyle="round,pad=0.5", fc='#F8FAFC', ec='#93C5FD', lw=1.5)
    ax.add_patch(smn1_box)
    ax.text(5.5, 83.5, "Homozygous SMN1 Loss / Deletion (Telomeric SMN1)", fontsize=13, fontweight='bold', color='#1E40AF')
    ax.text(5.5, 80.5, "Exon 6", fontsize=12, fontweight='bold', bbox=dict(boxstyle='square,pad=0.3', fc='#DBEAFE', ec='#3B82F6'))
    ax.text(12.5, 80.5, "Intron 6", fontsize=11, color='#64748B')
    ax.text(20.0, 80.5, "Exon 7 (C at c.840)", fontsize=12, fontweight='bold', color='#15803D', bbox=dict(boxstyle='square,pad=0.3', fc='#DCFCE7', ec='#16A34A'))
    ax.text(35.0, 80.5, "Exon 8", fontsize=12, fontweight='bold', bbox=dict(boxstyle='square,pad=0.3', fc='#DBEAFE', ec='#3B82F6'))
    ax.text(5.5, 75.5, "c.840C creates ESE heptamer binding SF2/ASF -> 100% Full-Length SMN Transcript", fontsize=12, color='#1E293B')
    ax.text(5.5, 72.0, "Result: Stable, oligomerization-competent ~38 kDa Functional SMN Protein", fontsize=12, fontweight='bold', color='#16A34A')

    # SMN2 block
    smn2_box = patches.FancyBboxPatch((4, 51.5), 42, 17.0, boxstyle="round,pad=0.5", fc='#FEF2F2', ec='#FCA5A5', lw=1.5)
    ax.add_patch(smn2_box)
    ax.text(5.5, 65.5, "Centromeric SMN2 Gene Modifier (Centromeric SMN2)", fontsize=13, fontweight='bold', color='#991B1B')
    ax.text(5.5, 62.5, "Exon 6", fontsize=12, fontweight='bold', bbox=dict(boxstyle='square,pad=0.3', fc='#FEE2E2', ec='#EF4444'))
    ax.text(12.5, 62.5, "Intron 6", fontsize=11, color='#64748B')
    ax.text(20.0, 62.5, "Exon 7 (T at c.840)", fontsize=12, fontweight='bold', color='#B91C1C', bbox=dict(boxstyle='square,pad=0.3', fc='#FEE2E2', ec='#EF4444'))
    ax.text(35.0, 62.5, "Exon 8", fontsize=12, fontweight='bold', bbox=dict(boxstyle='square,pad=0.3', fc='#FEE2E2', ec='#EF4444'))
    ax.text(5.5, 57.5, "c.840C>T transition disrupts ESE and creates ESS -> 85-90% Exon 7 Skipping", fontsize=12, color='#1E293B')
    ax.text(5.5, 54.0, "Result: 85-90% Truncated SMN-delta7 (rapidly degraded via ubiquitin-proteasome); 10-15% Full-Length SMN", fontsize=12, fontweight='bold', color='#DC2626')

    # Panel B: Neuromuscular Junction Pathology
    card_b = patches.FancyBboxPatch((52, 50), 46, 43, boxstyle="round,pad=0.8", fc='#FFFFFF', ec='#CBD5E1', lw=1.8)
    ax.add_patch(card_b)
    head_b = patches.Rectangle((52, 88.5), 46, 4.5, fc='#FDF2F8', ec='none')
    ax.add_patch(head_b)
    ax.text(53.5, 90.7, "B. Neuromuscular Unit Pathology & Agrin-MuSK Denervation",
            fontsize=14, fontweight='bold', color='#831843', va='center')

    nmj_box = patches.FancyBboxPatch((54, 51.5), 42, 35.5, boxstyle="round,pad=0.5", fc='#F8FAFC', ec='#CBD5E1', lw=1.2)
    ax.add_patch(nmj_box)
    ax.text(55.5, 83.5, "1. Spinal Motor Neuron Perikaryon Degeneration:", fontsize=13, fontweight='bold', color='#1E293B')
    ax.text(57.0, 80.5, "Loss of SMN disrupts snRNP biogenesis and pre-mRNA splicing in anterior horn alpha-motor neurons.", fontsize=12, color='#334155')
    ax.text(55.5, 75.5, "2. Impaired Axonal Transport & Neuromuscular Junction Synaptogenesis:", fontsize=13, fontweight='bold', color='#1E293B')
    ax.text(57.0, 72.5, "Disrupted Agrin/MuSK/LRP4 signaling prevents acetylcholine receptor clustering at motor endplates.", fontsize=12, color='#334155')
    ax.text(55.5, 67.5, "3. Progressive Retrograde Denervation & Myofiber Atrophy:", fontsize=13, fontweight='bold', color='#1E293B')
    ax.text(57.0, 64.5, "Terminal axon retraction, muscle denervation, and secondary grouped neurogenic atrophy.", fontsize=12, color='#334155')
    ax.text(55.5, 59.5, "4. Electrophysiological Correlate:", fontsize=13, fontweight='bold', color='#0F766E')
    ax.text(57.0, 55.5, "Marked reduction in Compound Muscle Action Potential (CMAP) amplitude (Type 1: <1.0 mV; Type 2: 1.5-3.5 mV).", fontsize=12, fontweight='bold', color='#0F766E')

    # Panel C: Clinical Phenotype Comparison
    card_c = patches.FancyBboxPatch((2, 4), 46, 43, boxstyle="round,pad=0.8", fc='#FFFFFF', ec='#CBD5E1', lw=1.8)
    ax.add_patch(card_c)
    head_c = patches.Rectangle((2, 42.5), 46, 4.5, fc='#FEF3C7', ec='none')
    ax.add_patch(head_c)
    ax.text(3.5, 44.7, "C. Phenotypic Stratification: SMA Type 1 vs. SMA Type 2",
            fontsize=14, fontweight='bold', color='#92400E', va='center')

    # Type 1 card
    c1_box = patches.FancyBboxPatch((4, 24.5), 42, 16.5, boxstyle="round,pad=0.5", fc='#FEF2F2', ec='#EF4444', lw=1.5)
    ax.add_patch(c1_box)
    ax.text(5.5, 37.5, "SMA Type 1 (Werdnig-Hoffmann Disease) - Severe Non-Sitter", fontsize=13, fontweight='bold', color='#DC2626')
    ax.text(5.5, 34.0, "Onset: 0 to 6 months (mean: 2.4 months) | SMN2 copies: Typically 2 (rarely 3)", fontsize=12, color='#1E293B')
    ax.text(5.5, 30.5, "Milestones: Never achieve independent sitting; severe generalized hypotonia & head lag", fontsize=12, color='#1E293B')
    ax.text(5.5, 27.0, "Natural History: Median survival <2 years without mechanical ventilatory support", fontsize=12, fontweight='bold', color='#991B1B')

    # Type 2 card
    c2_box = patches.FancyBboxPatch((4, 5.5), 42, 17.5, boxstyle="round,pad=0.5", fc='#EFF6FF', ec='#3B82F6', lw=1.5)
    ax.add_patch(c2_box)
    ax.text(5.5, 19.5, "SMA Type 2 (Dubowitz Disease) - Intermediate Sitter", fontsize=13, fontweight='bold', color='#2563EB')
    ax.text(5.5, 16.0, "Onset: 7 to 18 months (mean: 11.1 months) | SMN2 copies: 3 to 4 copies", fontsize=12, color='#1E293B')
    ax.text(5.5, 12.5, "Milestones: Achieve independent sitting unsupported; never walk independently", fontsize=12, color='#1E293B')
    ax.text(5.5, 9.0, "Complications: Progressive scoliosis, joint contractures, restrictive pulmonary decline", fontsize=12, fontweight='bold', color='#1E40AF')

    # Panel D: Motor Scales & Treatment Horizons
    card_d = patches.FancyBboxPatch((52, 4), 46, 43, boxstyle="round,pad=0.8", fc='#FFFFFF', ec='#CBD5E1', lw=1.8)
    ax.add_patch(card_d)
    head_d = patches.Rectangle((52, 42.5), 46, 4.5, fc='#F0FDF4', ec='none')
    ax.add_patch(head_d)
    ax.text(53.5, 44.7, "D. Standardized Motor Scales & Therapeutic Intervention Timeline",
            fontsize=14, fontweight='bold', color='#166534', va='center')

    d_box = patches.FancyBboxPatch((54, 5.5), 42, 35.5, boxstyle="round,pad=0.5", fc='#F8FAFC', ec='#CBD5E1', lw=1.2)
    ax.add_patch(d_box)
    ax.text(55.5, 37.5, "1. Standardized Psychometric Motor Scales:", fontsize=13, fontweight='bold', color='#1E293B')
    ax.text(57.0, 34.0, "CHOP-INTEND (Type 1): 16 items, 0-64 pts (Head/trunk control, reach, kicking). MCID >= 4 pts.", fontsize=12, color='#334155')
    ax.text(57.0, 30.5, "HFMSE (Type 2): 33 items, 0-66 pts (Sitting, rolling, crawling, standing, squatting). MCID >= 3 pts.", fontsize=12, color='#334155')
    ax.text(55.5, 25.5, "2. Disease-Modifying Therapies (DMTs):", fontsize=13, fontweight='bold', color='#1E293B')
    ax.text(57.0, 22.0, "Nusinersen: Intrathecal ASO targeting ISS-N1 to promote SMN2 exon 7 inclusion.", fontsize=12, color='#1E40AF')
    ax.text(57.0, 18.5, "Risdiplam: Daily oral small-molecule SMN2 pre-mRNA splicing modifier.", fontsize=12, color='#047857')
    ax.text(57.0, 15.0, "Onasemnogene Abeparvovec: AAV9 intravenous gene replacement supplying functional SMN1 cDNA.", fontsize=12, color='#6D28D9')
    ax.text(55.5, 10.0, "Crucial Prognostic Window: Early treatment prior to irreversible motor unit loss dictates long-term response.", fontsize=12, fontweight='bold', color='#DC2626')

    save_figure(fig, 1, "Clinical_and_Pathophysiological_Overview")

# -----------------------------------------------------------------------------
# FIGURE 2: Study Design & XAI Workflow (14pt headings, 12pt matter)
# -----------------------------------------------------------------------------
def plot_figure_2():
    fig = plt.figure(figsize=(24, 15.0), facecolor='#F8FAFC')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#F8FAFC')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 97.4, "Study Design, Multimodal Data Harmonization, and Explainable ML Architecture",
            ha='center', va='center', fontsize=20, fontweight='bold', color='#0F172A')
    ax.text(50, 95.5, "From Multi-Center Registry Cohort to Nested Bayesian Validation and Bedside XAI Explanations",
            ha='center', va='center', fontsize=13, color='#475569', style='italic')

    # Step 1: Patient Cohort
    c1 = patches.FancyBboxPatch((2, 10), 22, 82, boxstyle="round,pad=0.8", fc='#FFFFFF', ec='#3B82F6', lw=2)
    ax.add_patch(c1)
    ax.text(13, 89.0, "Step 1: Patient Cohort", ha='center', fontsize=14, fontweight='bold', color='#1E40AF')
    ax.text(13, 85.5, "(N = 350 Patients)", ha='center', fontsize=13, fontweight='bold', color='#334155')

    ax.text(4, 80.0, "Inclusion Criteria:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(4, 76.5, "- Genetically confirmed 5q SMA\n  (homozygous SMN1 deletion)", fontsize=12, color='#334155')
    ax.text(4, 71.0, "- Clinical Type 1 (n = 154) or\n  Clinical Type 2 (n = 196)", fontsize=12, color='#334155')
    ax.text(4, 65.5, "- Documented SMN2 copy number\n  via MLPA (2, 3, or 4 copies)", fontsize=12, color='#334155')
    ax.text(4, 60.0, "- Baseline CMAP electrophysiology", fontsize=12, color='#334155')
    ax.text(4, 55.5, "- Complete 24-month longitudinal\n  motor function follow-up", fontsize=12, color='#334155')

    ax.text(4, 48.0, "Treatment Regimens:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(4, 44.5, "- Nusinersen: 158 (45.1%)", fontsize=12, color='#1E40AF')
    ax.text(4, 41.0, "- Risdiplam: 117 (33.4%)", fontsize=12, color='#047857')
    ax.text(4, 37.5, "- Onasemnogene: 57 (16.3%)", fontsize=12, color='#6D28D9')
    ax.text(4, 34.0, "- Untreated Controls: 18 (5.1%)", fontsize=12, color='#64748B')

    ax.text(4, 26.0, "Assessment Intervals:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(4, 22.0, "Baseline (0m), 6m, 12m,\n18m, and 24m follow-up.", fontsize=12, color='#334155')
    ax.text(4, 15.0, "Target: 24-Month Delta Score\nand MCID Attainment.", fontsize=12, fontweight='bold', color='#DC2626')

    # Step 2: Feature Space
    c2 = patches.FancyBboxPatch((26, 10), 22, 82, boxstyle="round,pad=0.8", fc='#FFFFFF', ec='#10B981', lw=2)
    ax.add_patch(c2)
    ax.text(37, 89.0, "Step 2: Feature Space", ha='center', fontsize=14, fontweight='bold', color='#065F46')
    ax.text(37, 85.5, "(11 Multimodal Variables)", ha='center', fontsize=13, fontweight='bold', color='#334155')

    ax.text(28, 80.0, "Genomic Predictor:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(28, 76.5, "- SMN2 Copy Number (2, 3, 4)", fontsize=12, color='#334155')

    ax.text(28, 71.0, "Chronological Predictors:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(28, 67.5, "- Age at Symptom Onset (mo)\n- Treatment Initiation Delay (mo)\n- Chronological Age at Tx (mo)", fontsize=12, color='#334155')

    ax.text(28, 57.0, "Neurophysiology:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(28, 53.5, "- Baseline Ulnar CMAP (mV)", fontsize=12, color='#334155')

    ax.text(28, 48.0, "Motor Function Scale:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(28, 44.5, "- Raw Baseline Score\n- Normalized Baseline Motor (%)\n  [CHOP-INTEND or HFMSE]", fontsize=12, color='#334155')

    ax.text(28, 35.0, "Anthropometric & Clinical:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(28, 31.5, "- Nutritional Z-Score (BMI/WFA)\n- Daily Baseline NIV (hrs/day)\n- Scoliosis Cobb Angle (deg)", fontsize=12, color='#334155')

    ax.text(28, 19.0, "Preprocessing Pipeline:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(28, 15.0, "Median/MICE imputation (<2.5%)\nRobust scaling & one-hot encoding.", fontsize=12, color='#334155')

    # Step 3: Machine Learning Models
    c3 = patches.FancyBboxPatch((50, 10), 22, 82, boxstyle="round,pad=0.8", fc='#FFFFFF', ec='#8B5CF6', lw=2)
    ax.add_patch(c3)
    ax.text(61, 89.0, "Step 3: ML Modeling", ha='center', fontsize=14, fontweight='bold', color='#5B21B6')
    ax.text(61, 85.5, "(Nested Cross-Validation)", ha='center', fontsize=13, fontweight='bold', color='#334155')

    ax.text(52, 80.0, "Candidate Architectures:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(52, 76.5, "- Gradient Boosting (GBM) [Best]\n- Random Forest (RF)\n- Support Vector Machine (SVR)\n- Multi-Layer Perceptron (MLP)\n- ElasticNet Regression\n- Ridge Regression", fontsize=12, color='#334155')

    ax.text(52, 53.0, "Validation Rigor:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(52, 49.5, "- 10-Fold Outer Loop (Eval)\n- 5-Fold Inner Loop (Bayesian\n  hyperparameter tuning)", fontsize=12, color='#334155')
    ax.text(52, 39.5, "- Stratification across SMA\n  Type 1 and Type 2 cohorts", fontsize=12, color='#334155')
    ax.text(52, 32.5, "- Permutation testing (n = 1000)\n  and Monte Carlo stability splits", fontsize=12, color='#334155')

    ax.text(52, 22.0, "Primary Benchmark:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(52, 17.0, "GBM achieves R2 = 0.840,\nRMSE = 1.77, Pearson r = 0.926.", fontsize=12, fontweight='bold', color='#16A34A')

    # Step 4: Explainability & Decision Support
    c4 = patches.FancyBboxPatch((74, 10), 24, 82, boxstyle="round,pad=0.8", fc='#FFFFFF', ec='#F59E0B', lw=2)
    ax.add_patch(c4)
    ax.text(86, 89.0, "Step 4: Explainable AI", ha='center', fontsize=14, fontweight='bold', color='#92400E')
    ax.text(86, 85.5, "(Game-Theoretic SHAP)", ha='center', fontsize=13, fontweight='bold', color='#334155')

    ax.text(76, 80.0, "Global Feature Attributions:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(76, 76.5, "- Exact TreeExplainer Shapley\n  values for all predictions", fontsize=12, color='#334155')
    ax.text(76, 69.5, "- Global ranking: Treatment Delay\n  (#1) and CMAP (#2) dominate", fontsize=12, color='#334155')
    ax.text(76, 62.5, "- Beeswarm directional plots", fontsize=12, color='#334155')

    ax.text(76, 54.0, "Feature Interactions:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(76, 50.5, "- Partial dependence curves with\n  second-order interaction effects\n- Critical therapeutic window:\n  Delay <= 3.5 mo drives +2.8 pts", fontsize=12, color='#334155')

    ax.text(76, 38.0, "Bedside Decision Support:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(76, 34.5, "- Personalized patient-level\n  waterfall decompositions\n- Transparent counterfactuals\n- FHIR / EHR integrable", fontsize=12, color='#334155')

    ax.text(76, 20.0, "Clinical Impact:", fontsize=13, fontweight='bold', color='#0F172A')
    ax.text(76, 15.0, "Objective prognostic counseling\nand optimized triage for early DMT.", fontsize=12, fontweight='bold', color='#1E40AF')

    save_figure(fig, 2, "Study_Design_and_Explainable_ML_Workflow")

# -----------------------------------------------------------------------------
# FIGURE 3: Baseline Characteristics
# -----------------------------------------------------------------------------
def plot_figure_3():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    # Panel A: Onset vs Delay
    sns.scatterplot(data=df_cohort, x='Age_Onset_mo', y='Treatment_Delay_mo',
                    hue='SMA_Type', palette={'Type 1': PALETTE['sma1'], 'Type 2': PALETTE['sma2']},
                    style='SMN2_Copies', s=65, alpha=0.85, ax=axes[0, 0])
    axes[0, 0].set_title("A. Symptom Onset vs. Treatment Delay", fontweight='bold')
    axes[0, 0].set_xlabel("Age at Symptom Onset (months)")
    axes[0, 0].set_ylabel("Treatment Initiation Delay (months)")

    # Panel B: Baseline CMAP Amplitude
    sns.boxplot(data=df_cohort, x='SMA_Type', y='Baseline_CMAP_mV', hue='SMN2_Copies',
                palette='Blues', ax=axes[0, 1])
    axes[0, 1].set_title("B. Baseline CMAP by Subtype & SMN2", fontweight='bold')
    axes[0, 1].set_xlabel("Clinical Subtype")
    axes[0, 1].set_ylabel("Baseline CMAP Amplitude (mV)")

    # Panel C: Normalized Motor Score
    sns.kdeplot(data=df_cohort[df_cohort['SMA_Type'] == 'Type 1'], x='Normalized_Base_Score',
                fill=True, color=PALETTE['sma1'], label='Type 1 (CHOP)', ax=axes[0, 2])
    sns.kdeplot(data=df_cohort[df_cohort['SMA_Type'] == 'Type 2'], x='Normalized_Base_Score',
                fill=True, color=PALETTE['sma2'], label='Type 2 (HFMSE)', ax=axes[0, 2])
    axes[0, 2].set_title("C. Normalized Baseline Motor Scores", fontweight='bold')
    axes[0, 2].set_xlabel("Normalized Score (%)")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].legend()

    # Panel D: Treatment Modality Distribution
    tx_counts = pd.crosstab(df_cohort['Treatment'], df_cohort['SMA_Type'])
    tx_counts.plot(kind='bar', stacked=True, color=[PALETTE['sma1'], PALETTE['sma2']], ax=axes[1, 0])
    axes[1, 0].set_title("D. Treatment Allocation Across Subtypes", fontweight='bold')
    axes[1, 0].set_xlabel("Disease-Modifying Therapy")
    axes[1, 0].set_ylabel("Patient Count (n)")
    axes[1, 0].tick_params(axis='x', rotation=25)

    # Panel E: Nutritional Z-Score vs NIV
    sns.scatterplot(data=df_cohort, x='Nutritional_ZScore', y='NIV_Hours_Daily',
                    hue='SMA_Type', palette={'Type 1': PALETTE['sma1'], 'Type 2': PALETTE['sma2']},
                    alpha=0.75, s=60, ax=axes[1, 1])
    axes[1, 1].set_title("E. Nutritional Z-Score vs. Daily NIV Hours", fontweight='bold')
    axes[1, 1].set_xlabel("Nutritional Z-Score (BMI/WFA)")
    axes[1, 1].set_ylabel("Daily NIV Usage (hours/day)")

    # Panel F: Scoliosis Cobb Angle
    sns.violinplot(data=df_cohort, x='SMA_Type', y='Scoliosis_Cobb_deg',
                   palette={'Type 1': PALETTE['sma1'], 'Type 2': PALETTE['sma2']}, ax=axes[1, 2], cut=0)
    axes[1, 2].set_title("F. Spinal Curvature Cobb Angle", fontweight='bold')
    axes[1, 2].set_xlabel("Clinical Subtype")
    axes[1, 2].set_ylabel("Cobb Angle (degrees)")

    save_figure(fig, 3, "Baseline_Characteristics")

# -----------------------------------------------------------------------------
# FIGURE 4: Correlation Heatmap
# -----------------------------------------------------------------------------
def plot_figure_4():
    numeric_cols = [
        'SMN2_Copies', 'Age_Onset_mo', 'Treatment_Delay_mo',
        'Age_Treatment_Start_mo', 'Baseline_CMAP_mV', 'Normalized_Base_Score',
        'Nutritional_ZScore', 'NIV_Hours_Daily', 'Scoliosis_Cobb_deg',
        'Delta_Score_24m'
    ]
    corr = df_cohort[numeric_cols].corr(method='spearman')

    col_labels = [
        'SMN2 Copies', 'Onset Age (mo)', 'Treatment Delay (mo)',
        'Tx Start Age (mo)', 'CMAP (mV)', 'Base Score (%)',
        'Nutritional Z', 'NIV Hours/day', 'Scoliosis Cobb',
        'Delta Score 24m'
    ]

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='vlag', center=0,
                vmin=-0.65, vmax=0.65, square=True, linewidths=0.75,
                xticklabels=col_labels, yticklabels=col_labels, cbar_kws={'label': "Spearman Rank Correlation (rho)"},
                ax=ax)
    ax.set_title("Correlation Heatmap of Baseline Predictors & 24-Month Motor Gains", fontweight='bold', fontsize=13, pad=12)
    save_figure(fig, 4, "Correlation_Heatmap")

# -----------------------------------------------------------------------------
# FIGURE 5: Longitudinal Trajectories
# -----------------------------------------------------------------------------
def plot_figure_5():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    timepoints = [0, 6, 12, 18, 24]
    time_labels = ['Baseline', '6m', '12m', '18m', '24m']

    # Panel A: Type 1 CHOP-INTEND
    df1 = df_cohort[df_cohort['SMA_Type'] == 'Type 1']
    for _, row in df1.sample(min(45, len(df1)), random_state=42).iterrows():
        scores = [row['Score_0m'], row['Score_6m'], row['Score_12m'], row['Score_18m'], row['Score_24m']]
        axes[0].plot(timepoints, scores, color='#FCA5A5', alpha=0.35, lw=1.1)

    mean_chop = [df1[f'Score_{m}m'].mean() for m in timepoints]
    std_chop = [df1[f'Score_{m}m'].std() for m in timepoints]
    axes[0].errorbar(timepoints, mean_chop, yerr=std_chop, color=PALETTE['sma1'], fmt='-o',
                     lw=2.8, markersize=8, capsize=5, label='Mean +- SD')
    axes[0].axhline(y=4.0 + df1['Score_0m'].mean(), color='#991B1B', ls='--', lw=1.5, label='MCID Threshold (+4 pts)')
    axes[0].set_title("A. SMA Type 1: Longitudinal CHOP-INTEND (n=154)", fontweight='bold')
    axes[0].set_xlabel("Follow-Up Evaluation Interval")
    axes[0].set_ylabel("CHOP-INTEND Score (0 - 64 points)")
    axes[0].set_xticks(timepoints)
    axes[0].set_xticklabels(time_labels)
    axes[0].set_ylim(0, 64)
    axes[0].legend(loc='lower right')

    # Panel B: Type 2 HFMSE
    df2 = df_cohort[df_cohort['SMA_Type'] == 'Type 2']
    for _, row in df2.sample(min(45, len(df2)), random_state=42).iterrows():
        scores = [row['Score_0m'], row['Score_6m'], row['Score_12m'], row['Score_18m'], row['Score_24m']]
        axes[1].plot(timepoints, scores, color='#93C5FD', alpha=0.35, lw=1.1)

    mean_hfmse = [df2[f'Score_{m}m'].mean() for m in timepoints]
    std_hfmse = [df2[f'Score_{m}m'].std() for m in timepoints]
    axes[1].errorbar(timepoints, mean_hfmse, yerr=std_hfmse, color=PALETTE['sma2'], fmt='-s',
                     lw=2.8, markersize=8, capsize=5, label='Mean +- SD')
    axes[1].axhline(y=3.0 + df2['Score_0m'].mean(), color='#1E40AF', ls='--', lw=1.5, label='MCID Threshold (+3 pts)')
    axes[1].set_title("B. SMA Type 2: Longitudinal HFMSE (n=196)", fontweight='bold')
    axes[1].set_xlabel("Follow-Up Evaluation Interval")
    axes[1].set_ylabel("HFMSE Score (0 - 66 points)")
    axes[1].set_xticks(timepoints)
    axes[1].set_xticklabels(time_labels)
    axes[1].set_ylim(0, 66)
    axes[1].legend(loc='lower right')

    save_figure(fig, 5, "Longitudinal_Trajectories")

# -----------------------------------------------------------------------------
# FIGURE 6: Longitudinal Changes & Responders
# -----------------------------------------------------------------------------
def plot_figure_6():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel A: Waterfall of 24m Changes
    df_sorted = df_cohort.sort_values(by='Delta_Score_24m', ascending=False).reset_index(drop=True)
    colors = [PALETTE['sma1'] if t == 'Type 1' else PALETTE['sma2'] for t in df_sorted['SMA_Type']]
    axes[0].bar(range(len(df_sorted)), df_sorted['Delta_Score_24m'], color=colors, width=1.0, edgecolor='none', alpha=0.85)
    axes[0].axhline(y=0, color='#334155', lw=1.0)
    axes[0].axhline(y=4.0, color='#DC2626', ls='--', lw=1.5, label='Type 1 MCID (+4 pts)')
    axes[0].axhline(y=3.0, color='#2563EB', ls=':', lw=1.5, label='Type 2 MCID (+3 pts)')
    axes[0].set_title("A. Cohort Waterfall: 24-Month Motor Score Gain (N=350)", fontweight='bold')
    axes[0].set_xlabel("Individual Patients (Ranked by Delta Score)")
    axes[0].set_ylabel("24-Month Score Change (Points)")
    axes[0].legend()

    # Panel B: Responder Proportions by SMN2 Copy Number
    resp_df = df_cohort.groupby(['SMA_Type', 'SMN2_Copies'])['Is_Responder'].mean().reset_index()
    resp_df['Percent'] = resp_df['Is_Responder'] * 100.0

    sns.barplot(data=resp_df, x='SMN2_Copies', y='Percent', hue='SMA_Type',
                palette={'Type 1': PALETTE['sma1'], 'Type 2': PALETTE['sma2']}, ax=axes[1])
    axes[1].set_title("B. MCID Responder Proportions by SMN2 Copy Count", fontweight='bold')
    axes[1].set_xlabel("SMN2 Gene Copy Number")
    axes[1].set_ylabel("MCID Responders (%)")
    axes[1].set_ylim(0, 100)
    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:
            axes[1].annotate(f"{height:.1f}%",
                             (p.get_x() + p.get_width() / 2., height / 2.),
                             ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    save_figure(fig, 6, "Longitudinal_Changes_and_Responders")

if __name__ == "__main__":
    print("Running plot_figures_1_to_6.py...")
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    print("Figures 1 to 6 generated successfully!")
