# Data (not included in this repository)

This repository does not include the raw market CSV datasets used for backtesting.

## Expected format
Input data should be provided as OHLC time series with (at minimum) the following columns:
- Timestamp (or Date)
- Open
- High
- Low
- Close
- Volume

## How to prepare data
Export market data from your preferred source as CSV and place prepared files under `data/`.
This folder is ignored by Git except for this README.
