# Firms Industry Sector Classification from Quarterly Financial Fundamentals

Hi! This is my final project for DATA1030.

The project involves using Quarterly Financial Fundamentals for Firms Industry Sector Classification. This is a time-series multi-class classification problem. The full dataset contains more than 2.7M rows and 150 columns after feature engineering.


Compustat global dataset link (requires Brown email access): 
https://drive.google.com/file/d/17PVUrG5wFJmN6V9GFcdPPnzVmvfw6jqn/view?usp=sharing

To recreate the environment, install the conda environment from the `environment.yml` file.

Packages used:
	- numpy=1.24.4
	- pandas=2.0.3
	- matplotlib=3.7.2
	- seaborn=0.12.2
	- scipy=1.11.2
	- scikit-learn=1.3.0
	- scikit-optimize=0.9.0 (requires manually fixing a numpy bug in the package at transformer.py)
	- xgboost=2.0.2 (requires pip installation)