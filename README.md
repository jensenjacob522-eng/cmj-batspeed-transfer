# CMJ → Bat Speed Transfer Modeling

A level-stratified regression framework analyzing the relationship between countermovement jump (CMJ) height and bat speed across competitive levels.

This project investigates how lower-body force production transfers to rotational bat speed — and how that relationship changes from High School to Professional athletes.

It also includes a reproducible CLI-based projection tool for individual athlete evaluation.

---

## Objective

To:

- Quantify how much variance in bat speed can be explained by CMJ height across competitive levels
- Separate raw force production capacity from transfer efficiency
- Build a reproducible projection tool for athlete-specific bat speed estimation

This framework allows evaluation of whether performance limitations are primarily strength-based or transfer/technical in nature.

---

## Modeling Approach

### 1️⃣ Automated Data Cleaning

- Removes bat speeds < 40 mph  
- Removes ±3 SD outliers  
- Handles missing values programmatically  

### 2️⃣ Level-Stratified Linear Regression

Separate regression models are fit for:

- High School
- College
- Professional

This allows slope and correlation comparisons across levels.

### 3️⃣ Residual-Based Transfer Efficiency Analysis

Residual = Actual − Predicted

Used to identify:
- Over-performers (efficient transfer)
- Under-performers (force production not translating)

### 4️⃣ Athlete Projection Tool

The `predict_batspeed.py` script:

- Fits a level-filtered regression model
- Predicts bat speed from CMJ height
- Computes bootstrap 95% confidence intervals
- Automatically generates HTML + PDF reports

---

## Example Findings (Filtered Dataset)

High School: r ≈ 0.56  
College: r ≈ 0.45  
Professional: r ≈ 0.21  

The predictive slope decreases as level increases, suggesting:

- Raw force production explains more variance at lower levels
- Transfer efficiency and sequencing become more dominant at elite levels

---

## Practical Application

This framework helps distinguish:

- Athletes limited by force production
- Athletes limited by transfer efficiency

### Coaching Implications

High School  
→ Emphasize force development

College  
→ Blend force development + sequencing refinement

Professional  
→ Emphasize transfer efficiency, mechanics, and kinetic chain timing

---

# Quick Start (Reproducible Demo)

Clone the repository and run the projection tool using the included demo dataset:

```bash
git clone https://github.com/jensenjacob522-eng/cmj-batspeed-transfer
cd cmj-batspeed-transfer

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python predict_batspeed.py --input data/demo_data.csv --level College --athlete_cmj 35

## Development Notes

Built using an agent-assisted workflow:
- AI-assisted scaffolding of analysis pipeline
- Iterative debugging and refactoring
- Automated filtering logic
- Modularized metric and plotting functions
- Reproducible environment via requirements.txt
