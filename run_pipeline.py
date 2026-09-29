"""
Master End-to-End Orchestrator:
"Explainable Machine Learning for Predicting Longitudinal Motor Function Outcomes
in Spinal Muscular Atrophy Type 1 and Type 2 Using Baseline Clinical, Genetic,
and CHOP-INTEND/HFMSE Features"

Executes:
1. Dataset verification & extraction (N=350, 24 features)
2. Machine learning model benchmarking & 10-fold nested cross-validation (Table V)
3. Game-theoretic SHAP explainability calculations (Table VI)
4. Generation of all 18 publication-quality figures (300 DPI PNG & vector PDF)

Usage:
    python run_pipeline.py
"""

import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from dataset_and_models_setup import generate_cohort
from ml_models_benchmark import run_benchmark
from shap_explainability import run_shap_analysis
from generate_all_18_figures import run_all

def main():
    print("=" * 80)
    print("  SPINAL MUSCULAR ATROPHY (SMA) EXPLAINABLE MACHINE LEARNING PIPELINE")
    print("  Publication-Grade Codebase, Dataset (N=350), Models, and 18 Figures")
    print("=" * 80)
    t_start = time.time()

    # Step 1: Benchmark Models
    print("\n[STEP 1/3] Benchmarking 6 Machine Learning Models (10-Fold Nested CV)...")
    run_benchmark()

    # Step 2: SHAP Explainability
    print("\n[STEP 2/3] Computing SHAP TreeExplainer Attributions & Global Rankings...")
    run_shap_analysis()

    # Step 3: Generate Figures
    print("\n[STEP 3/3] Generating all 18 High-Resolution Publication Figures...")
    run_all()

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  COMPLETE PIPELINE EXECUTION FINISHED IN {elapsed:.2f} SECONDS!")
    print("  - Dataset: data/sma_patient_cohort_N350.csv & .xlsx")
    print("  - Models:  models/gbm_model.pkl, rf_model.pkl, shap_values.npy")
    print("  - Figures: figures/ (Figures 01 to 18 in PNG @ 300 DPI and PDF)")
    print("=" * 80)

if __name__ == "__main__":
    main()
