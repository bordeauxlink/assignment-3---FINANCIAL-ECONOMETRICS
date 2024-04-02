# a
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bonds = pd.read_parquet('UK_bonds.parquet')
bonds['mean'] = bonds.mean(axis=1)
sp500 = pd.read_csv('S_P 500 Historical Data.csv')

# descriptive statistics
print("UK Bonds Descriptive Statistics:")
print(bonds.describe())

#sp500
sp500['Price'] = pd.to_numeric(sp500['Price'].str.replace(',', ''))
sp500['Date'] = pd.to_datetime(sp500['Date'])
sp500 = sp500.sort_values('Date')
print(sp500)
# Generate descriptive statistics for 'Price'
print("S&P 500 'Price' Descriptive Statistics:")
print(sp500['Price'].describe())

# Plot bond time-series graphs
plt.figure(figsize=(12, 6))
sns.lineplot(data=bonds)
plt.title('UK Bonds Yields Over Time')
plt.xlabel('Date')
plt.ylabel('Yield')
plt.show()

# 'Date' as the index
sp500.set_index('Date', inplace=True)

# Plot sp500 time-series graph
plt.figure(figsize=(12, 6))
sns.lineplot(data=sp500['Price'])
plt.title('S&P 500 Index Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#b
from statsmodels.tsa.stattools import adfuller

def calculate_adf_statistics(s, p):
    # ADF test for HA: stationary
    result_stationary = adfuller(s, maxlag=p)
    adf_t_stat_stationary = result_stationary[0]
    adf_delta_stat_stationary = len(s) * result_stationary[1]

    # ADF test for HA: trend-stationary
    result_trend_stationary = adfuller(s, maxlag=p, regression='ct')
    adf_t_stat_trend_stationary = result_trend_stationary[0]
    adf_delta_stat_trend_stationary = len(s) * result_trend_stationary[1]

    return adf_t_stat_stationary, adf_t_stat_trend_stationary, adf_delta_stat_stationary, adf_delta_stat_trend_stationary

#sp500
adf_stats_sp500 = calculate_adf_statistics(sp500['Price'], p=1)
print("ADF Statistics for S&P 500:")
print("ADF t statistic (stationary):", adf_stats_sp500[0])
print("ADF t statistic (trend-stationary):", adf_stats_sp500[1])
print("ADF δ statistic (stationary):", adf_stats_sp500[2])
print("ADF δ statistic (trend-stationary):", adf_stats_sp500[3])

# bonds_mean
bonds_mean_series = bonds.mean(axis=1)
adf_stats_bonds_mean = calculate_adf_statistics(bonds_mean_series, p=1)

print("ADF Statistics for Mean of Bonds:")
print("ADF t statistic (stationary):", adf_stats_bonds_mean[0])
print("ADF t statistic (trend-stationary):", adf_stats_bonds_mean[1])
print("ADF δ statistic (stationary):", adf_stats_bonds_mean[2])
print("ADF δ statistic (trend-stationary):", adf_stats_bonds_mean[3])

#c
