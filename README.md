# IBOVESPA Direction Forecasting — Time Series Analysis & Machine Learning

<a href="https://colab.research.google.com/github/dressasys/TechChallenge2_Grupo87_PosTech/blob/main/TechChallenge2%20_v3.ipynb" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)
![License](https://img.shields.io/badge/License-MIT-green)

> 🇧🇷 [Versão em português disponível aqui](README_PT.md)

> Predicting next-day IBOVESPA market direction (up/down) using classical time series models and machine learning classifiers trained on six correlated macroeconomic indicators from the Brazilian and global markets (2022–2025).

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Academic Context](#academic-context)

---

## Problem Statement

The IBOVESPA (Índice Bovespa) is Brazil's primary stock market benchmark, widely used by institutional and retail investors to gauge economic sentiment. Anticipating whether the index will rise or fall the following trading day — even with modest accuracy above 55–60% — has direct implications for portfolio allocation and risk management.

Manual interpretation of correlated macroeconomic variables (exchange rates, commodity prices, foreign indices, interest rates) is time-consuming and prone to cognitive bias. This project frames the problem as a **binary classification task**: given today's market data, predict whether tomorrow's IBOVESPA closing price will be higher than today's.

---

## Methodology

The project follows a structured ML pipeline with a strong emphasis on **temporal integrity** — no future data leaks into training.

```
Raw CSVs → Preprocessing → Feature Engineering → EDA → Model Training → Temporal Evaluation
```

### 1. Data Collection & Preprocessing
- Six financial time series loaded from CSV files (2022–2025)
- Date parsing, encoding normalization, and column standardization
- NaN imputation using the mean of adjacent trading days (interpolation)
- Volume column parsed from string (e.g., `1.5B`) to float

### 2. Target Variable
```python
ibov['ibov_alta'] = (next_day_close > today_close).astype(int)
# 1 → market up tomorrow | 0 → market flat or down
```

### 3. Temporal Train/Validation/Test Split
To avoid look-ahead bias, data is **never shuffled**:
- **Test set**: last 30 trading days (most recent)
- **Validation set**: 10% of remaining data (immediately preceding test)
- **Training set**: everything before that

> Note: The period 2025-10-22 to 2025-11-11 was identified as a structural outlier — the largest consecutive IBOVESPA rally in 15 years — and was removed to avoid distorting the training signal.

---

## Feature Engineering

Over **40 features** were engineered across four categories:

| Category | Examples |
|---|---|
| Technical Indicators | MACD, MACD Signal, SMA-5, SMA-10, SMA-100, EMA-100 |
| Lag Returns (1–10 days) | `ibov_ret_lag_1`, `brent_ret_lag_2`, `dolar_ret_lag_1`, `vale_ret_lag_1` |
| Rolling Volatility | 5-day rolling std for IBOV, Brent, Vale, Dollar |
| Cross-asset Ratios | `brent_dolar_ratio`, `ma_brent_10`, intraday range |
| Raw OHLCV | Open, High, Low, Close, Volume |

All lag and rolling features were computed **before the train/test split** to preserve temporal causality.

---

## Models Evaluated

| Model | Type | Purpose | Split Strategy |
|---|---|---|---|
| **SARIMAX** | Classical TS | Trend & seasonality decomposition | In-sample fit |
| **Prophet** | Additive TS | Changepoint & seasonality detection | In-sample fit |
| **Random Forest** | ML Classifier | Direction prediction baseline | Temporal (no shuffle) |
| **XGBoost** | ML Classifier | Direction prediction (final model) | Temporal (no shuffle) |

> The alternative notebook (`Modelagem_Series_Temporais_IBOV.ipynb`) explores a purely technical approach using only OHLCV features from Yahoo Finance (2015–2025) as a comparative baseline.

---

## Results

### XGBoost — Final Model (Temporal Split)

| Split | Accuracy | AUC-ROC |
|---|---|---|
| Validation | **60.00%** | **65.69%** |
| Test (last 30 days) | 36.67% | 37.95% |

**Validation Classification Report**

```
              precision    recall  f1-score   support
           0       0.61      0.64      0.62        36
           1       0.59      0.56      0.58        34
    accuracy                           0.60        70
```

The test set performance degradation reflects a **market regime shift**: the test window coincided with an abnormal volatility period, demonstrating how out-of-distribution events challenge even well-trained models.

### Top Features by Importance (XGBoost)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `ma_brent_10` | 0.028 |
| 2 | `vale_ret_lag_1` | 0.028 |
| 3 | `range_diario` | 0.028 |
| 4 | `brent_dolar_ratio` | 0.027 |
| 5 | `Abertura` (Open) | 0.027 |

Cross-asset features (Brent/Dollar ratio, Vale lags) outperformed pure IBOVESPA technical indicators, supporting the thesis that the Brazilian market is highly sensitive to commodity prices and FX movements.

---

## Key Findings

- **Cross-asset signals matter more than pure technicals**: Brent crude, Vale3 returns, and USD/BRL lags consistently ranked among the top predictors, reflecting Brazil's commodity-driven economy.
- **Temporal evaluation is critical in financial ML**: A shuffled train/test split inflates accuracy (yielding ~75%+) by allowing the model to learn from future data — a methodological flaw that produces misleading results in real-world applications.
- **Structural outliers require deliberate handling**: The Oct–Nov 2025 IBOVESPA anomaly (unprecedented 15-year rally) was identified through EDA and removed to prevent it from distorting the model's generalization.
- **Decomposition revealed clear seasonality**: STL decomposition confirmed a long-term trend reversal in 2023 and intra-month cyclical patterns driven by macro events.
- **AUC > Accuracy as evaluation metric**: With a near-balanced target class, AUC-ROC (65.7% on validation) provides a more reliable performance signal than raw accuracy.

---

## Dataset

All data covers the period **January 2022 – October 2025**.

| File | Variable | Source |
|---|---|---|
| `Ibovespa_2022_2025.csv` | IBOVESPA OHLCV + Variation | B3 / Investing.com |
| `Dolar_2022_2025.csv` | USD/BRL exchange rate | Investing.com |
| `Petroleo_brent_2022_2025.csv` | Brent crude oil (USD/barrel) | Investing.com |
| `SeP500_2022_2025.csv` | S&P 500 index | Investing.com |
| `Selic_2022_2025.csv` | Brazilian base interest rate | Banco Central do Brasil |
| `Vale3_2022_2025.csv` | Vale S.A. (VALE3) stock price | B3 / Investing.com |

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Statistical Models | `statsmodels` (SARIMAX, seasonal_decompose, ACF) |
| Forecasting | `prophet` |
| Technical Indicators | `pandas_ta` (MACD, SMA, EMA) |
| Machine Learning | `scikit-learn` (RandomForest, metrics, model_selection) |
| Gradient Boosting | `xgboost` (XGBClassifier) |
| Data Acquisition | `yfinance` (alternative notebook) |
| Environment | Google Colab / Jupyter Notebook |

---

## Repository Structure

```
├── TechChallenge2 _v3.ipynb           # Final notebook — full pipeline (recommended)
├── TechChallenge2_v2.ipynb            # Intermediate version
├── TechChallenge2.ipynb               # Initial exploration
├── Modelagem_Series_Temporais_IBOV.ipynb  # Alternative: OHLCV-only approach (yfinance)
├── Ibovespa_2022_2025.csv
├── Dolar_2022_2025.csv
├── Petroleo_brent_2022_2025.csv
├── SeP500_2022_2025.csv
├── Selic_2022_2025.csv
└── Vale3_2022_2025.csv
```

---

## How to Run

### Option 1 — Google Colab (Recommended)

Click the badge at the top of this README to open `TechChallenge2_v3.ipynb` directly in Colab. All CSV files are loaded from the repository path, so clone or mount the repo first:

```python
# In the first Colab cell, if needed:
!git clone https://github.com/dressasys/TechChallenge2_Grupo87_PosTech.git
%cd TechChallenge2_Grupo87_PosTech
```

### Option 2 — Local Jupyter

```bash
# 1. Clone the repository
git clone https://github.com/dressasys/TechChallenge2_Grupo87_PosTech.git
cd TechChallenge2_Grupo87_PosTech

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels prophet pandas_ta yfinance

# 3. Launch Jupyter
jupyter notebook "TechChallenge2 _v3.ipynb"
```

> All CSV data files are included in the repository. No additional data download is required.

---

## Academic Context

**Program**: FIAP PosTech — Business Analytics & Data-Driven Decision Making  
**Phase**: 2 — Tech Challenge  
**Group**: 87  

This project was developed as a practical application of time series analysis and machine learning techniques in a financial domain, covering the full data science lifecycle from raw data ingestion to model evaluation under realistic conditions.

---

*Developed by Group 87 — FIAP PosTech · 2025*
