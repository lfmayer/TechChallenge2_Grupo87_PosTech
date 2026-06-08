# IBOVESPA Time Series Forecasting · Machine Learning

Predictive modeling of the IBOVESPA index using time series analysis
and machine learning. The model forecasts market direction based on
macroeconomic indicators including USD/BRL exchange rate, Brent crude
oil, S&P 500, Selic interest rate, and Vale3 stock data (2022–2025).

---

## Problem

Financial institutions and investors need reliable tools to anticipate
market movements. Manual analysis of correlated macroeconomic variables
is time-consuming and prone to cognitive bias.

## Approach

- Collected and processed historical data for 6 financial indicators
(IBOVESPA, USD/BRL, Brent oil, S&P 500, Selic rate, Vale3)
- Performed temporal EDA to identify trends, seasonality, and
correlations between variables
- Built time series forecasting models to predict IBOVESPA direction
- Evaluated models targeting 75%+ predictive accuracy

## Data Sources

| Dataset | Period | Description |
|---|---|---|
| Ibovespa_2022_2025.csv | 2022–2025 | Brazilian stock market index |
| Dolar_2022_2025.csv | 2022–2025 | USD/BRL exchange rate |
| Petroleo_brent_2022_2025.csv | 2022–2025 | Brent crude oil price |
| SeP500_2022_2025.csv | 2022–2025 | US stock market index |
| Selic_2022_2025.csv | 2022–2025 | Brazilian interest rate |
| Vale3_2022_2025.csv | 2022–2025 | Vale S.A. stock price |

## Tech Stack

- **Python** · pandas · numpy · matplotlib · seaborn · scikit-learn
- **Time Series:** statsmodels · temporal feature engineering
- **Environment:** Google Colab / Jupyter Notebook

## Project Structure

```
├── TechChallenge2_v3.ipynb         # Final notebook (recommended)
├── TechChallenge2_v2.ipynb         # Previous version
├── TechChallenge2.ipynb            # Initial version
├── Ibovespa_2022_2025.csv
├── Dolar_2022_2025.csv
├── Petroleo_brent_2022_2025.csv
├── SeP500_2022_2025.csv
├── Selic_2022_2025.csv
└── Vale3_2022_2025.csv
```

## How to Run

```bash
# Open TechChallenge2_v3.ipynb in Google Colab or Jupyter
# All data files are included in the repository
```

---

**FIAP PosTech** · Business Analytics & Data-Driven Decision
Phase 2 Tech Challenge · Group 87
