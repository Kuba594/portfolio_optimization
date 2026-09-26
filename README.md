# Dynamic Portfolio Optimization

Models realized asset covariances and predicts future covariance matrices
for portfolio construction, evaluating performance and risk against
predicted covariance structures.

## Overview
This project forecasts asset covariance matrices using machine learning
methods and uses these predictions to construct and evaluate investment
portfolios, comparing performance and risk outcomes to the underlying
covariance structure.

## Features
- Realized covariance modeling from historical asset data
- ML-based covariance forecasting
- Portfolio construction based on predicted covariances
- Performance and risk evaluation:
  - RMSE
  - QLIKE
  - Return
  - Volatility
  - Sharpe Ratio
  - Maximum Drawdown
  - Transaction Costs
- Models used:
  - HAR
  - DCC-GARCH
  - DCC-RV
  - LSTM
  - GRU

## Tech stack
Python, R, PyTorch
