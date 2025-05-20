# Quant Lab – ML-Driven Trading Simulator

Quant Lab is a web-based application for training machine learning models on historical stock data and simulating trading strategies based on model predictions. It bridges financial data analysis, predictive modeling, and trading performance evaluation in a single interactive interface.

---

## Features

- Search any stock ticker (e.g., AAPL, MSFT) and load up to 5 years of price history
- Train machine learning models: Linear Regression, Random Forest, and XGBoost
- Tune hyperparameters interactively (tree depth, learning rate, estimators)
- Visualize predicted vs. actual prices
- Backtest a trading strategy using model-based buy/sell signals
- Evaluate performance with metrics such as Sharpe Ratio, Drawdown, and Annualized Return

---

## Live Demo

- [Launch the app on Streamlit](https://ameur-quantlab.streamlit.app/)
- [View the source code on GitHub](https://github.com/nassim747/QuantLab)

## Preview

![Quant Lab Screenshot](capture.png)
_A sample view of the model training and strategy backtest interface._

---

## Installation

```bash
# Clone this repository
git clone https://github.com/nassim747/QuantLab.git
cd QuantLab

# Install required packages
pip install -r requirements.txt

# Run the application
python -m streamlit run app.py
