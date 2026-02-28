# CMJ → Bat Speed Transfer Efficiency Model

This project builds a repeatable modeling pipeline to evaluate transfer efficiency between CMJ jump height and bat speed across competition levels.

## Features

- Automated data cleaning
  - Removes bat speeds < 40 mph
  - Removes ±3 SD outliers
- Level-stratified regression modeling
  - High School
  - College
  - Pro
- Residual-based transfer efficiency analysis
- Automated HTML + PDF report generation

## Example Findings (Filtered)

High School: r ≈ 0.56  
College: r ≈ 0.45  
Pro: r ≈ 0.21  

Predictive slope decreases as level increases, suggesting raw force production becomes less explanatory at elite levels.

## Run the Report

```bash
python driveline_transfer_report.py --input data/hp_obp.csv