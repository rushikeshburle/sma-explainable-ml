"""
Master Script: Generate All 18 Publication-Grade Figures
"Explainable Machine Learning for Predicting Longitudinal Motor Function Outcomes
in Spinal Muscular Atrophy Type 1 and Type 2 Using Baseline Clinical, Genetic,
and CHOP-INTEND/HFMSE Features"

Executes:
- Figures 1 to 6 (plot_figures_1_to_6.py)
- Figures 7 to 12 (plot_figures_7_to_12.py)
- Figures 13 to 18 (plot_figures_13_to_18.py)

Outputs all 18 figures at 300 DPI in PNG and vector PDF format into figures/ directory.
"""

import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from plot_figures_1_to_6 import (
    plot_figure_1, plot_figure_2, plot_figure_3,
    plot_figure_4, plot_figure_5, plot_figure_6
)
from plot_figures_7_to_12 import (
    plot_figure_7, plot_figure_8, plot_figure_9,
    plot_figure_10, plot_figure_11, plot_figure_12
)
from plot_figures_13_to_18 import (
    plot_figure_13, plot_figure_14, plot_figure_15,
    plot_figure_16, plot_figure_17, plot_figure_18
)

def run_all():
    print("=" * 75)
    print("STARTING FULL REPRODUCTION OF ALL 18 FIGURES")
    print("Project: Explainable ML for Longitudinal SMA Outcomes (N=350)")
    print("=" * 75)
    start_time = time.time()

    print("\n--- Generating Part 1: Figures 1 to 6 ---")
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()

    print("\n--- Generating Part 2: Figures 7 to 12 ---")
    plot_figure_7()
    plot_figure_8()
    plot_figure_9()
    plot_figure_10()
    plot_figure_11()
    plot_figure_12()

    print("\n--- Generating Part 3: Figures 13 to 18 ---")
    plot_figure_13()
    plot_figure_14()
    plot_figure_15()
    plot_figure_16()
    plot_figure_17()
    plot_figure_18()

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"ALL 18 FIGURES GENERATED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("Output directory: figures/ (PNG @ 300 DPI and Vector PDF)")
    print("=" * 75)

if __name__ == "__main__":
    run_all()
