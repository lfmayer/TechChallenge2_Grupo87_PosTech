# Previsão da Direção do IBOVESPA — Análise de Séries Temporais e Machine Learning

<a href="https://colab.research.google.com/github/dressasys/TechChallenge2_Grupo87_PosTech/blob/main/TechChallenge2%20_v3.ipynb" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir no Colab"/>
</a>
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)
![License](https://img.shields.io/badge/License-MIT-green)

> 🇺🇸 [English version available here](README.md)

> Previsão da direção do IBOVESPA no dia seguinte (alta/baixa) utilizando modelos clássicos de séries temporais e classificadores de machine learning treinados em seis indicadores macroeconômicos correlacionados dos mercados brasileiro e global (2022–2025).

---

## Sumário

- [Problema](#problema)
- [Metodologia](#metodologia)
- [Engenharia de Features](#engenharia-de-features)
- [Modelos Avaliados](#modelos-avaliados)
- [Resultados](#resultados)
- [Principais Conclusões](#principais-conclusões)
- [Dataset](#dataset)
- [Stack Tecnológica](#stack-tecnológica)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Como Executar](#como-executar)
- [Contexto Acadêmico](#contexto-acadêmico)

---

## Problema

O IBOVESPA (Índice Bovespa) é o principal benchmark do mercado de ações brasileiro, amplamente utilizado por investidores institucionais e individuais para mensurar o sentimento econômico. Antecipar se o índice subirá ou cairá no próximo pregão — mesmo com uma acurácia modesta acima de 55–60% — tem implicações diretas para alocação de portfólio e gestão de risco.

A interpretação manual de variáveis macroeconômicas correlacionadas (câmbio, preços de commodities, índices estrangeiros, taxas de juros) é trabalhosa e sujeita a viés cognitivo. Este projeto enquadra o problema como uma **tarefa de classificação binária**: dado os dados de mercado de hoje, prever se o preço de fechamento do IBOVESPA amanhã será maior do que o de hoje.

---

## Metodologia

O projeto segue um pipeline estruturado de ML com forte ênfase em **integridade temporal** — nenhum dado futuro vaza para o treinamento.

```
CSVs Brutos → Pré-processamento → Engenharia de Features → AED → Treinamento → Avaliação Temporal
```

### 1. Coleta e Pré-processamento dos Dados
- Seis séries temporais financeiras carregadas de arquivos CSV (2022–2025)
- Parsing de datas, normalização de encoding e padronização de colunas
- Imputação de NaN pela média dos dias de negociação adjacentes (interpolação)
- Coluna de volume convertida de string (ex: `1.5B`) para float

### 2. Variável-Alvo
```python
ibov['ibov_alta'] = (fechamento_dia_seguinte > fechamento_hoje).astype(int)
# 1 → mercado em alta amanhã | 0 → mercado estável ou em queda
```

### 3. Divisão Temporal Treino/Validação/Teste
Para evitar viés de antecipação (look-ahead bias), os dados **nunca são embaralhados**:
- **Conjunto de teste**: últimos 30 pregões (os mais recentes)
- **Conjunto de validação**: 10% dos dados restantes (imediatamente anterior ao teste)
- **Conjunto de treinamento**: tudo antes disso

> Nota: O período de 22/10/2025 a 11/11/2025 foi identificado como outlier estrutural — a maior alta consecutiva do IBOVESPA em 15 anos — e foi removido para evitar distorção no sinal de treinamento.

---

## Engenharia de Features

Mais de **40 features** foram construídas em quatro categorias:

| Categoria | Exemplos |
|---|---|
| Indicadores Técnicos | MACD, Sinal MACD, SMA-5, SMA-10, SMA-100, EMA-100 |
| Retornos Defasados (1–10 dias) | `ibov_ret_lag_1`, `brent_ret_lag_2`, `dolar_ret_lag_1`, `vale_ret_lag_1` |
| Volatilidade Móvel | Desvio padrão móvel de 5 dias para IBOV, Brent, Vale, Dólar |
| Razões entre Ativos | `brent_dolar_ratio`, `ma_brent_10`, amplitude intradiária |
| OHLCV Bruto | Abertura, Máxima, Mínima, Fechamento, Volume |

Todas as features defasadas e móveis foram calculadas **antes da divisão treino/teste** para preservar a causalidade temporal.

---

## Modelos Avaliados

| Modelo | Tipo | Finalidade | Estratégia de Divisão |
|---|---|---|---|
| **SARIMAX** | ST Clássica | Decomposição de tendência e sazonalidade | Ajuste in-sample |
| **Prophet** | ST Aditiva | Detecção de pontos de mudança e sazonalidade | Ajuste in-sample |
| **Random Forest** | Classificador ML | Baseline de previsão de direção | Temporal (sem embaralhamento) |
| **XGBoost** | Classificador ML | Previsão de direção (modelo final) | Temporal (sem embaralhamento) |

> O notebook alternativo (`Modelagem_Series_Temporais_IBOV.ipynb`) explora uma abordagem puramente técnica utilizando apenas features OHLCV do Yahoo Finance (2015–2025) como baseline comparativo.

---

## Resultados

### XGBoost — Modelo Final (Divisão Temporal)

| Conjunto | Acurácia | AUC-ROC |
|---|---|---|
| Validação | **60,00%** | **65,69%** |
| Teste (últimos 30 dias) | 36,67% | 37,95% |

**Relatório de Classificação — Validação**

```
              precision    recall  f1-score   support
           0       0.61      0.64      0.62        36
           1       0.59      0.56      0.58        34
    accuracy                           0.60        70
```

A queda de desempenho no conjunto de teste reflete uma **mudança de regime de mercado**: a janela de teste coincidiu com um período de volatilidade anormal, demonstrando como eventos fora da distribuição desafiam até modelos bem treinados.

### Top Features por Importância (XGBoost)

| Rank | Feature | Importância |
|---|---|---|
| 1 | `ma_brent_10` | 0,028 |
| 2 | `vale_ret_lag_1` | 0,028 |
| 3 | `range_diario` | 0,028 |
| 4 | `brent_dolar_ratio` | 0,027 |
| 5 | `Abertura` | 0,027 |

Features entre ativos (razão Brent/Dólar, defasagens da Vale) superaram indicadores técnicos puramente do IBOVESPA, corroborando a tese de que o mercado brasileiro é altamente sensível a preços de commodities e variações cambiais.

---

## Principais Conclusões

- **Sinais entre ativos importam mais do que puros indicadores técnicos**: Petróleo Brent, retornos da Vale3 e defasagens do USD/BRL se destacaram consistentemente entre os melhores preditores, refletindo a economia brasileira orientada a commodities.
- **Avaliação temporal é crítica em ML financeiro**: Uma divisão treino/teste embaralhada infla a acurácia (alcançando ~75%+) ao permitir que o modelo aprenda com dados futuros — uma falha metodológica que produz resultados enganosos em aplicações reais.
- **Outliers estruturais exigem tratamento deliberado**: A anomalia do IBOVESPA em out–nov/2025 (maior alta em 15 anos) foi identificada via AED e removida para evitar distorção na generalização do modelo.
- **Decomposição revelou sazonalidade clara**: A decomposição STL confirmou uma reversão de tendência de longo prazo em 2023 e padrões cíclicos intramensais impulsionados por eventos macroeconômicos.
- **AUC superior à Acurácia como métrica de avaliação**: Com uma classe-alvo quase balanceada, o AUC-ROC (65,7% na validação) fornece um sinal de desempenho mais confiável do que a acurácia bruta.

---

## Dataset

Todos os dados cobrem o período de **janeiro de 2022 a outubro de 2025**.

| Arquivo | Variável | Fonte |
|---|---|---|
| `Ibovespa_2022_2025.csv` | IBOVESPA OHLCV + Variação | B3 / Investing.com |
| `Dolar_2022_2025.csv` | Taxa de câmbio USD/BRL | Investing.com |
| `Petroleo_brent_2022_2025.csv` | Petróleo Brent (USD/barril) | Investing.com |
| `SeP500_2022_2025.csv` | Índice S&P 500 | Investing.com |
| `Selic_2022_2025.csv` | Taxa básica de juros brasileira | Banco Central do Brasil |
| `Vale3_2022_2025.csv` | Preço da ação Vale S.A. (VALE3) | B3 / Investing.com |

---

## Stack Tecnológica

| Camada | Bibliotecas |
|---|---|
| Manipulação de Dados | `pandas`, `numpy` |
| Visualização | `matplotlib`, `seaborn` |
| Modelos Estatísticos | `statsmodels` (SARIMAX, seasonal_decompose, ACF) |
| Previsão | `prophet` |
| Indicadores Técnicos | `pandas_ta` (MACD, SMA, EMA) |
| Machine Learning | `scikit-learn` (RandomForest, metrics, model_selection) |
| Gradient Boosting | `xgboost` (XGBClassifier) |
| Aquisição de Dados | `yfinance` (notebook alternativo) |
| Ambiente | Google Colab / Jupyter Notebook |

---

## Estrutura do Repositório

```
├── TechChallenge2 _v3.ipynb           # Notebook final — pipeline completo (recomendado)
├── TechChallenge2_v2.ipynb            # Versão intermediária
├── TechChallenge2.ipynb               # Exploração inicial
├── Modelagem_Series_Temporais_IBOV.ipynb  # Alternativo: abordagem apenas com OHLCV (yfinance)
├── Ibovespa_2022_2025.csv
├── Dolar_2022_2025.csv
├── Petroleo_brent_2022_2025.csv
├── SeP500_2022_2025.csv
├── Selic_2022_2025.csv
└── Vale3_2022_2025.csv
```

---

## Como Executar

### Opção 1 — Google Colab (Recomendado)

Clique no badge no topo deste README para abrir o `TechChallenge2_v3.ipynb` diretamente no Colab. Todos os arquivos CSV são carregados a partir do caminho do repositório, portanto clone ou monte o repositório primeiro:

```python
# Na primeira célula do Colab, se necessário:
!git clone https://github.com/dressasys/TechChallenge2_Grupo87_PosTech.git
%cd TechChallenge2_Grupo87_PosTech
```

### Opção 2 — Jupyter Local

```bash
# 1. Clone o repositório
git clone https://github.com/dressasys/TechChallenge2_Grupo87_PosTech.git
cd TechChallenge2_Grupo87_PosTech

# 2. Instale as dependências
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels prophet pandas_ta yfinance

# 3. Inicie o Jupyter
jupyter notebook "TechChallenge2 _v3.ipynb"
```

> Todos os arquivos CSV de dados estão incluídos no repositório. Nenhum download adicional de dados é necessário.

---

## Contexto Acadêmico

**Programa**: FIAP PosTech — Business Analytics & Data-Driven Decision Making  
**Fase**: 2 — Tech Challenge  
**Grupo**: 87  

Este projeto foi desenvolvido como aplicação prática de técnicas de análise de séries temporais e machine learning no domínio financeiro, cobrindo o ciclo completo de ciência de dados — da ingestão de dados brutos à avaliação de modelos em condições realistas.

---

*Desenvolvido pelo Grupo 87 — FIAP PosTech · 2025*
