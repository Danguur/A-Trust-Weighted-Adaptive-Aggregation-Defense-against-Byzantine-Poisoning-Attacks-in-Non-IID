#!/bin/sh
python small_scale/advanced_experiments.py
python dev_v2.py
python small_scale/make_small_figures.py --out paper_assets/figures
python small_scale/make_small_tables.py --out paper_assets/tables --dev_csv dev_v2.csv
python small_scale/make_diagrams.py --out paper_assets/figures
