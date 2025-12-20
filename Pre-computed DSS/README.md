# NumericAg: A Pre-computed, Similarity-based Decision Support System for Nitrogen Management

This repository contains the core implementation of **NumericAg**, a modular and scalable decision support system (DSS) designed to deliver **real-time, uncertainty-aware nitrogen recommendations** in precision agriculture.  

The system adopts a **pre-computation architecture**, in which computationally intensive tasks—such as similarity matching, quadratic–plateau (QP) yield modeling, and probabilistic profit estimation—are executed offline and stored in indexed databases. Online user queries are resolved through efficient lookup and aggregation, enabling rapid response times even when operating on millions of historical records.

---

## 1. System Overview

NumericAg integrates agronomic similarity analysis, yield response modeling, and economic uncertainty into a unified DSS framework. The system is specifically designed to support **field-level nitrogen decision-making under data sparsity and uncertainty**.

### Key Characteristics
- Similarity-based matching between user inputs and historical field records
- Quadratic–Plateau (QP) yield response modeling with probabilistic error representation
- Economic risk analysis under fertilizer and crop price uncertainty
- Fully pre-computed backend enabling sub-second to second-level query response times
- Modular architecture allowing future model substitution or extension

---

## 2. Computational Pipeline

### Step 1 – Similarity Pre-computation  
**File:** `Step1_Similarity_Calculation.py`

All possible combinations of discretized user inputs (bins) are generated for continuous and categorical agronomic features. For each user-input combination and each historical record, a weighted similarity score is computed and stored.

- Continuous features: `ACLAY`, `SOM`, `CHU`, `AWDR`
- Categorical features: Previous crop type, Tillage system
- Similarity formulation: normalized distance with power parameter `q`
- Each user-input combination is encoded as a hashed `bin_key`

**Output table:** `mode4`  
This table forms the backbone of all downstream computations.

---

### Step 2 – Yield Modeling and Error Probability Estimation  
**File:** `Step2_Errorprob_Calculation.py`

For each user-input group, a **Quadratic–Plateau (QP)** yield response model is fitted using similarity-weighted historical observations.

- Candidate QP parameters are evaluated via brute-force search
- The optimal parameter set maximizes near-zero error probability
- Prediction errors are aggregated into probabilistic distributions
- A checkpoint mechanism allows safe resume of long-running computations

**Output tables:**
- `ErrorProb` – probabilistic yield error representation  
- `QP_Checkpoint` – execution tracking and recovery

---

### Step 3 – Economic Profit and Risk Assessment  
**File:** `Step3_profit_Calculation.py`

Expected economic outcomes are computed by combining yield uncertainty with fertilizer and crop price uncertainty.

- Net Return over Cost of Fertilization (NRCF)
- Expected Fertilization Benefit (EFB)
- Probability of exceeding economic thresholds (e.g., \$1000–\$2500 ha⁻¹)

Results are stored in a lookup table indexed by nitrogen rate and user-input group.

**Output table:** `ExpectedProfitLookup_Rebuilt`

---

### Step 4 – Backend API  
**File:** `Step4_Backend.py`

A FastAPI backend exposes the pre-computed DSS through REST endpoints.

**Core functions:**
- Conversion of real-valued user inputs into bin indices
- Retrieval of best-matching pre-computed scenarios
- Identification of economically optimal nitrogen rate (EONR)
- Generation of tables, charts, and optional email summaries

---

### Step 5 – Frontend Interface  
**File:** `Step5_Front.txt`

A React-based frontend provides an interactive interface for end users.

- Map-based field selection (Leaflet)
- Dynamic agronomic input sliders
- Visualization of profit curves, uncertainty, and EONR
- Comparison of multiple decision scenarios

---

### Step 6 – Sensitivity and Robustness Analysis  
**File:** `Step6_sensitivityAnalysis.py`

This module evaluates DSS robustness and interpretability.

- Monte Carlo resampling under correlated feature distributions
- Top-K similarity aggregation analysis
- Convergence diagnostics
- Surrogate modeling and SHAP-based explainability

---

## 3. Repository Structure

```text
Pre_computed DSS/
│
├── README.md
├── LICENSE
├── requirements.txt
│
│
├── precomputation/
│   ├── Step1_Similarity_Calculation.py
│   ├── Step2_Errorprob_Calculation.py
│   ├── Step3_profit_Calculation.py
│
├── backend/
│   ├── Step4_Backend.py
│   └── api_models.py
│
├── frontend/
│   └── Step5_Front.txt
│
├── analysis/
  └── Step6_sensitivityAnalysis.py

