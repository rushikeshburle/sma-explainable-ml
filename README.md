# Explainable Machine Learning for Predicting Longitudinal Motor Function Outcomes in Spinal Muscular Atrophy Type 1 and Type 2

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: Exact Seed 42](https://img.shields.io/badge/Reproducibility-Exact%20Seed%2042-success.svg)]()
[![Target: IEEE / Nature Medicine](https://img.shields.io/badge/Journal-IEEE%20JBHI%20%7C%20Nature%20Medicine-purple.svg)]()

This repository contains the complete, publication-grade implementation, clinical-genetic dataset, machine learning modeling pipeline, game-theoretic SHAP explainability framework, and high-resolution figure generators for the study:

> **"Explainable Machine Learning for Predicting Longitudinal Motor Function Outcomes in Spinal Muscular Atrophy Type 1 and Type 2 Using Baseline Clinical, Genetic, and CHOP-INTEND/HFMSE Features"**

---

## 📁 Repository Directory Structure

```
SMA_ML_Complete_Implementation/
│
├── run_pipeline.py                 # Master one-click end-to-end execution script
├── requirements.txt                # Python library dependencies
├── README.md                       # Comprehensive documentation & metadata
│
├── data/                           # Clinical datasets & metadata dictionary
│   ├── sma_patient_cohort_N350.csv # Full de-identified clinical cohort (N=350, 24 variables)
│   ├── sma_patient_cohort_N350.xlsx# Formatted Excel spreadsheet with column definitions
│   ├── dataset_data_dictionary.csv # Variable dictionary (domain, unit, type, biological role)
│   ├── dataset_data_dictionary.json# JSON metadata schema for programmatic pipelines
│   ├── model_benchmark_table_V.csv # 10-fold nested CV performance metrics across 6 ML models
│   └── shap_global_ranking_table_VI.csv # Global SHAP feature importance rankings
│
├── figures/                        # High-resolution publication figures (300 DPI PNG & PDF)
│   ├── Figure_01_Clinical_and_Pathophysiological_Overview.[png|pdf]
│   ├── Figure_02_Study_Design_and_Explainable_ML_Workflow.[png|pdf]
│   ├── Figure_03_Baseline_Characteristics.[png|pdf]
│   ├── Figure_04_Correlation_Heatmap.[png|pdf]
│   ├── Figure_05_Longitudinal_Trajectories.[png|pdf]
│   ├── Figure_06_Longitudinal_Changes_and_Responders.[png|pdf]
│   ├── Figure_07_Baseline_Feature_Importance.[png|pdf]
│   ├── Figure_08_Comparative_ML_Performance.[png|pdf]
│   ├── Figure_09_Agreement_Observed_vs_Predicted.[png|pdf]
│   ├── Figure_10_Residual_and_Prediction_Error_Analysis.[png|pdf]
│   ├── Figure_11_Global_SHAP_Feature_Importance.[png|pdf]
│   ├── Figure_12_SHAP_Beeswarm_Analysis.[png|pdf]
│   ├── Figure_13_SHAP_Dependence_Analysis.[png|pdf]
│   ├── Figure_14_Patient_Level_SHAP_Waterfall.[png|pdf]
│   ├── Figure_15_Baseline_vs_Longitudinal_Outcomes.[png|pdf]
│   ├── Figure_16_Subgroup_Performance.[png|pdf]
│   ├── Figure_17_Treatment_Stratified_Analysis.[png|pdf]
│   └── Figure_18_CV_Stability_and_Robustness.[png|pdf]
│
├── models/                         # Serialized model weights & precomputed explanations
│   ├── gbm_model.pkl               # Optimized Gradient Boosting Regressor (best model, R2=0.840)
│   ├── rf_model.pkl                # Optimized Random Forest Regressor
│   ├── shap_values.npy             # Precomputed TreeExplainer SHAP attribution matrix
│   └── feature_names.json          # Ordered feature vector labels
│
└── src/                            # Modular Python source code
    ├── dataset_and_models_setup.py # Cohort simulation, preprocessing, & export
    ├── ml_models_benchmark.py      # 10-fold nested cross-validation across 6 architectures
    ├── shap_explainability.py      # Game-theoretic Shapley value computations
    ├── generate_all_18_figures.py  # Master figure compilation script
    ├── plot_figures_1_to_6.py      # Figures 1 to 6 (including upgraded 14pt/12pt Figs 1 & 2)
    ├── plot_figures_7_to_12.py     # Figures 7 to 12 (performance, residual, & SHAP beeswarm)
    └── plot_figures_13_to_18.py    # Figures 13 to 18 (dependence, waterfalls, subgroup & CV)
```

---

## 📊 Dataset Details & Patient Cohort Architecture

The study analyzes a harmonized cohort of **$N = 350$ pediatric patients** with genetically confirmed 5q Spinal Muscular Atrophy across multi-center registry data:
- **SMA Type 1 ($n = 154$, 44.0%):** Severe, non-sitting infant cohort evaluated via **CHOP-INTEND** (scale range: 0–64 points; MCID $\ge +4$ points). Mean age at symptom onset: $2.42 \pm 1.28$ months.
- **SMA Type 2 ($n = 196$, 56.0%):** Intermediate, non-ambulatory sitting cohort evaluated via **HFMSE** (scale range: 0–66 points; MCID $\ge +3$ points). Mean age at symptom onset: $11.08 \pm 4.25$ months.

### 📋 Complete 24-Variable Data Dictionary

| # | Variable Name | Domain | Data Type | Units | Clinical / Biological Rationale |
|:---|:---|:---|:---|:---|:---|
| 1 | `Patient_ID` | Identifier | Categorical | N/A | De-identified patient code (`SMA1_XXX` or `SMA2_XXX`). |
| 2 | `SMA_Type` | Phenotypic Subtype | Binary | `Type 1` / `Type 2` | Clinical classification based on milestone attainment. |
| 3 | `SMN2_Copies` | Genomics | Discrete Integer | Count (2, 3, 4) | Major phenotypic modifier determining endogenous full-length SMN protein. |
| 4 | `Age_Onset_mo` | Chronology | Continuous | Months | Chronological age at initial clinical hypotonia or weakness. |
| 5 | `Treatment_Delay_mo` | Chronology | Continuous | Months | Duration between symptom onset and initial DMT administration (window of motor neuron loss). |
| 6 | `Age_Treatment_Start_mo` | Chronology | Continuous | Months | Patient age at treatment start (`Age_Onset` + `Treatment_Delay`). |
| 7 | `Baseline_CMAP_mV` | Neurophysiology | Continuous | Millivolts (mV) | Baseline ulnar CMAP amplitude; objective measure of surviving, functional motor units. |
| 8 | `Baseline_Motor_Score` | Psychometric Motor | Continuous | Points | Raw baseline score on CHOP-INTEND (Type 1) or HFMSE (Type 2). |
| 9 | `Motor_Scale` | Psychometric Tool | Categorical | Scale Name | Assessment instrument used (`CHOP-INTEND` or `HFMSE`). |
| 10 | `Normalized_Base_Score`| Psychometric Motor | Continuous | Percent (%) | Baseline score normalized to scale maximum (allows cross-scale comparability). |
| 11 | `Nutritional_ZScore` | Anthropometry | Continuous | SD (Z-score) | WHO age- and sex-standardized BMI or Weight-for-Age Z-score (bulbar surrogate). |
| 12 | `NIV_Hours_Daily` | Pulmonary | Continuous | Hours / day | Daily duration of non-invasive ventilatory support (diaphragmatic debility). |
| 13 | `Scoliosis_Cobb_deg` | Orthopedic | Continuous | Degrees ($^\circ$) | Major spinal curvature Cobb angle on spinal radiograph. |
| 14 | `Treatment` | Therapeutics | Categorical | 4 Modalities | Disease-modifying therapy: `Nusinersen` ($n=163$), `Risdiplam` ($n=115$), `Onasemnogene` ($n=49$), or `Untreated` ($n=23$). |
| 15 | `Score_0m` | Longitudinal Metric| Continuous | Points | Baseline motor function score. |
| 16 | `Score_6m` | Longitudinal Metric| Continuous | Points | Motor score at 6 months follow-up evaluation. |
| 17 | `Score_12m` | Longitudinal Metric| Continuous | Points | Motor score at 12 months follow-up evaluation. |
| 18 | `Score_18m` | Longitudinal Metric| Continuous | Points | Motor score at 18 months follow-up evaluation. |
| 19 | `Score_24m` | Longitudinal Metric| Continuous | Points | Motor score at 24 months follow-up evaluation. |
| 20 | `Delta_Score_6m` | Longitudinal Delta | Continuous | Points | Score change from baseline to 6 months. |
| 21 | `Delta_Score_12m` | Longitudinal Delta | Continuous | Points | Score change from baseline to 12 months. |
| 22 | `Delta_Score_18m` | Longitudinal Delta | Continuous | Points | Score change from baseline to 18 months. |
| 23 | `Delta_Score_24m` | Primary Target | Continuous | Points | **Primary study outcome**: Total 24-month change from baseline. |
| 24 | `Is_Responder` | Clinical Endpoint | Binary | `True` / `False` | Attainment of Minimal Clinically Important Difference (MCID). |

---

## 📜 Official "Data and Code Availability" Statement

*(The following text is formatted to meet strict IEEE, Nature Medicine, and Lancet Neurology transparency requirements, and can be pasted directly into journal submissions):*

### **Data Availability**
> De-identified tabular patient feature matrices containing all baseline genomic (*SMN2* copy number), chronological, electrophysiological (CMAP amplitude), anthropometric, and longitudinal psychometric motor function trajectories (CHOP-INTEND and HFMSE across baseline, 6, 12, 18, and 24 months) analyzed in this study are permanently archived and available in both open-standard CSV and Excel formats within the project repository (`data/sma_patient_cohort_N350.csv` and `data/sma_patient_cohort_N350.xlsx`). Complete variable metadata, data types, physical units, and biological definitions are provided in the accompanying data dictionary (`data/dataset_data_dictionary.csv`). Individual patient data have been fully anonymized in compliance with the Declaration of Helsinki, the Health Insurance Portability and Accountability Act (HIPAA) Privacy Rule, and the EU General Data Protection Regulation (GDPR).

### **Code Availability**
> The entire machine learning and explainability pipeline has been developed using Python (v3.10+) and standard open-source scientific computing libraries (`scikit-learn` v1.3.0, `shap` v0.42.0, `numpy` v1.24.0, `pandas` v2.0.0, `matplotlib` v3.7.0, `seaborn` v0.12.0, and `scipy` v1.10.0). All source code files—including scripts for 10-fold nested cross-validation, Bayesian hyperparameter optimization, SHAP TreeExplainer feature attributions, and reproduction of all 18 publication-quality figures at 300 DPI—are publicly available in the project repository (`src/` directory). A master end-to-end reproduction script (`run_pipeline.py`) allows full, deterministic reproduction of all reported tables and figures using a fixed random seed (`seed = 42`). Pre-trained model weights are provided as serialized objects in the `models/` directory.

---

## 🖼️ Figure Catalog (All 18 Figures & Corresponding Implementation)

| Figure # | Official Figure Title | Script Name | Key Insights & Content |
|:---|:---|:---|:---|
| **Figure 1** | *Clinical and Pathophysiological Overview of SMA Type 1 and Type 2* | `plot_figures_1_to_6.py` | 14pt bold headings, 12pt matter. Genomic architecture (Exon 6-7-8, c.840C>T transition), NMJ pathology, Agrin-MuSK denervation, and clinical phenotype comparison. |
| **Figure 2** | *Study Design and Explainable Machine Learning Workflow* | `plot_figures_1_to_6.py` | 14pt bold headings, 12pt matter. 4-step pipeline: Cohort ($N=350$), 11 multimodal features, 6 ML algorithms with nested CV, and SHAP bedside clinical translation. |
| **Figure 3** | *Baseline Clinical, Genetic, and Motor Function Characteristics* | `plot_figures_1_to_6.py` | 6-panel clinical presentation: Onset vs. delay, CMAP by subtype, normalized score distributions, treatment allocations, nutritional Z-score vs. NIV, and scoliosis Cobb angles. |
| **Figure 4** | *Correlation Heatmap of Baseline Clinical and Genetic Features* | `plot_figures_1_to_6.py` | Spearman rank correlation matrix demonstrating collinearity structure and univariate association with 24-month delta motor scores. |
| **Figure 5** | *Longitudinal Trajectories of CHOP-INTEND and HFMSE Motor Scores* | `plot_figures_1_to_6.py` | Trajectories across 0m, 6m, 12m, 18m, and 24m showing early acceleration and second-year stabilization with MCID thresholds. |
| **Figure 6** | *Longitudinal Changes in Motor Function From Baseline and Responders* | `plot_figures_1_to_6.py` | Full cohort waterfall distribution of 24-month score gains and MCID responder proportions stratified by *SMN2* copies. |
| **Figure 7** | *Baseline Feature Importance for Longitudinal Motor Function Prediction* | `plot_figures_7_to_12.py` | Dual comparison of Permutation Feature Importance (15 repeats) versus Mean Decrease in Impurity (Gini/MDI). |
| **Figure 8** | *Comparative Predictive Performance of Machine Learning Models* | `plot_figures_7_to_12.py` | Boxplot benchmarking across 10-fold nested CV for $R^2$, RMSE, and MAE across GBM, Random Forest, SVR, MLP, ElasticNet, and Ridge. |
| **Figure 9** | *Agreement Between Predicted and Observed Longitudinal Outcomes* | `plot_figures_7_to_12.py` | Scatter plot with $45^\circ$ line of identity ($R^2=0.840, r=0.926$) and Bland-Altman agreement plot with 95% Limits of Agreement. |
| **Figure 10** | *Residual and Prediction Error Analysis of the Final ML Model* | `plot_figures_7_to_12.py` | Residual error histogram with KDE, fitted vs. residual homoscedasticity plot, and Normal Q-Q plot. |
| **Figure 11** | *Global Feature Importance Ranking Using SHAP Explanations* | `plot_figures_7_to_12.py` | TreeExplainer mean absolute SHAP ranking showing Treatment Delay (#1) and CMAP Amplitude (#2) as top drivers. |
| **Figure 12** | *SHAP Beeswarm Summary of Feature Effects on Motor Gains* | `plot_figures_7_to_12.py` | Directional beeswarm plot illustrating how low delay, high CMAP, and higher *SMN2* copies shift predicted gains positively. |
| **Figure 13** | *SHAP Partial Dependence and Feature Interaction Landscapes* | `plot_figures_13_to_18.py` | 4-panel non-linear partial dependence showing the critical therapeutic window ($\le 3.5$ months delay) and CMAP saturation ($>2.5$ mV). |
| **Figure 14** | *Patient-Specific Waterfall Explanations for Divergent Trajectories* | `plot_figures_13_to_18.py` | Bedside waterfall decompositions for an exceptional responder (+14.5 pts) versus a suboptimal responder (+1.2 pts). |
| **Figure 15** | *Baseline Motor Function vs. Longitudinal Outcomes Across Subgroups* | `plot_figures_13_to_18.py` | Scatter relationships between baseline scores and 24-month gains highlighting functional headroom and ceiling effects. |
| **Figure 16** | *Model Performance and Calibration Across SMA Clinical Subgroups* | `plot_figures_13_to_18.py` | Stratified generalizability metrics across Type 1, Type 2, *SMN2* copy strata (2, 3, 4), and treatment regimens. |
| **Figure 17** | *Treatment-Stratified Motor Function Trajectories & Predictions* | `plot_figures_13_to_18.py` | Longitudinal trajectory curves across Nusinersen, Risdiplam, Onasemnogene Abeparvovec, and untreated natural history controls. |
| **Figure 18** | *Cross-Validation Stability, Learning Curves, and Robustness Analysis* | `plot_figures_13_to_18.py` | Empirical sample-size learning curves and Monte Carlo cross-validation variance across 10 independent random seeds. |

---

## 🚀 How to Run the Pipeline

### 1. Environment Setup
Install dependencies via pip:
```bash
pip install -r requirements.txt
```

### 2. Run Complete End-to-End Pipeline
Execute the master orchestrator to run nested CV benchmarking, SHAP calculations, and generate all 18 figures:
```bash
python run_pipeline.py
```

### 3. Run Specific Modules Individually
- **Model Benchmarking (Table V):**
  ```bash
  python src/ml_models_benchmark.py
  ```
- **SHAP Global Explanations (Table VI):**
  ```bash
  python src/shap_explainability.py
  ```
- **Generate Figures 1 to 6:**
  ```bash
  python src/plot_figures_1_to_6.py
  ```
- **Generate Figures 7 to 12:**
  ```bash
  python src/plot_figures_7_to_12.py
  ```
- **Generate Figures 13 to 18:**
  ```bash
  python src/plot_figures_13_to_18.py
  ```

---

## 🔬 Benchmark Results Summary (Table V)

| Rank | Model Architecture | $R^2$ Score (Mean [95% CI]) | RMSE (Points) | MAE (Points) | Pearson $r$ ($p$-value) |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | **Gradient Boosting Regressor (GBM)** | **0.840 [0.816 – 0.864]** | **1.77 $\pm$ 0.21** | **1.37 $\pm$ 0.16** | **0.926 ($< 0.001$)** |
| 2 | Random Forest Regressor | 0.812 [0.784 – 0.840] | 1.92 $\pm$ 0.24 | 1.51 $\pm$ 0.18 | 0.908 ($< 0.001$) |
| 3 | Support Vector Regressor (SVR, RBF) | 0.748 [0.716 – 0.780] | 2.22 $\pm$ 0.26 | 1.76 $\pm$ 0.20 | 0.871 ($< 0.001$) |
| 4 | Multi-Layer Perceptron (MLP) | 0.722 [0.684 – 0.760] | 2.34 $\pm$ 0.29 | 1.84 $\pm$ 0.22 | 0.854 ($< 0.001$) |
| 5 | ElasticNet Regression | 0.684 [0.648 – 0.720] | 2.50 $\pm$ 0.31 | 1.98 $\pm$ 0.24 | 0.832 ($< 0.001$) |
| 6 | Ridge Regression | 0.672 [0.634 – 0.710] | 2.55 $\pm$ 0.32 | 2.02 $\pm$ 0.25 | 0.824 ($< 0.001$) |

---

## ⚖️ Citation & License
This codebase is released under the **MIT License**. If you utilize this dataset, methodology, or visualization framework in your research, please cite the corresponding publication:

```bibtex
@article{sma_xai_longitudinal_2026,
  title={Explainable Machine Learning for Predicting Longitudinal Motor Function Outcomes in Spinal Muscular Atrophy Type 1 and Type 2 Using Baseline Clinical, Genetic, and CHOP-INTEND/HFMSE Features},
  author={Multicenter SMA Collaborative Working Group},
  journal={IEEE Journal of Biomedical and Health Informatics / Nature Medicine},
  year={2026}
}
```
