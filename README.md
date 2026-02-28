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

## Data Source

This project utilizes anonymized elite-level athlete performance data from:

Wasserberger, K.W., Brady, A.C., Besky, D.M., Jones, B.R., & Boddy, K.J. (2022).  
*The OpenBiomechanics Project: The open source initiative for anonymized, elite-level athletic motion capture data.*  
Available at: https://github.com/drivelineresearch/openbiomechanics

This repository is an independent analytical implementation and is not affiliated with Driveline Baseball.

## Run the Report

```bash
python driveline_transfer_report.py --input data/hp_obp.csv
