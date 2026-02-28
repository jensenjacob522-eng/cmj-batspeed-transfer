CMJ → Bat Speed Transfer Modeling

A level-stratified regression framework analyzing the relationship between countermovement jump (CMJ) height and bat speed across competitive levels.

This project investigates how lower-body force production transfers to rotational bat speed, and how that relationship changes from High School to Professional athletes.

Objective
to quantify how much variance in bat speed can be explained by CMJ height at different competitive levels, and to separate:
Raw force production capacity
Transfer efficiency (technical/kinematic contribution)

Modeling Approach
Pipeline includes:
Automated data cleaning
Removes bat speeds < 40 mph
Removes ±3 SD outliers
Level-stratified linear regression models
High School
College
Professional
Residual-based transfer efficiency analysis
Residual = Actual − Predicted
Identifies over- and under-performers relative to force capacity
Automated HTML + PDF report generation

Example Findings (Filtered Dataset)
High School: r ≈ 0.56
College: r ≈ 0.45
Pro: r ≈ 0.21

Predictive slope decreases as level increases, suggesting:
Raw force production explains more variance at lower levels
Transfer efficiency and technical sequencing become more dominant at elite levels

Practical Application
This framework helps distinguish:
Athletes limited by force production
Athletes limited by transfer efficiency

Coaching implications:
HS: Emphasize force development
College: Blend force + sequencing
Pro: Emphasize transfer efficiency and mechanics

Data Source

This project utilizes anonymized elite-level athlete performance data from:

Wasserberger, K.W., Brady, A.C., Besky, D.M., Jones, B.R., & Boddy, K.J. (2022).
The OpenBiomechanics Project: The open source initiative for anonymized, elite-level athletic motion capture data.
Available at: https://github.com/drivelineresearch/openbiomechanics

This repository is an independent analytical implementation and is not affiliated with Driveline Baseball.

## Run the Report

```bash
python driveline_transfer_report.py --input data/hp_obp.csv
