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
    critical_values_stationary = result_stationary[4]

    # ADF test for HA: trend-stationary
    result_trend_stationary = adfuller(s, maxlag=p, regression='ct')
    adf_t_stat_trend_stationary = result_trend_stationary[0]
    adf_delta_stat_trend_stationary = len(s) * result_trend_stationary[1]
    critical_values_trend_stationary = result_trend_stationary[4]

    return (adf_t_stat_stationary, adf_t_stat_trend_stationary, 
            adf_delta_stat_stationary, adf_delta_stat_trend_stationary, 
            critical_values_stationary, critical_values_trend_stationary)
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
def simulate_and_test(s, N, T, p):
    # Calculate the difference of s
    delta_s = s.diff().dropna()

    # Initialize a DataFrame to store the results
    results = pd.DataFrame(0, index=['1%', '5%', '10%'], columns=['ADF t (stationary)', 'ADF t (trend-stationary)', 'ADF δ (stationary)', 'ADF δ (trend-stationary)'])

    # Simulate N random walks and run the ADF test
    for i in range(N):
        # Bootstrap from delta_s to generate a random walk
        random_walk = delta_s.sample(T, replace=True).cumsum()

        # Run the ADF test
        adf_t_stat_stationary, adf_t_stat_trend_stationary, adf_delta_stat_stationary, adf_delta_stat_trend_stationary, critical_values_stationary, critical_values_trend_stationary = calculate_adf_statistics(random_walk, p)

        # Compare the ADF statistics with the critical values
        for level in ['1%', '5%', '10%']:
            results.loc[level, 'ADF t (stationary)'] += adf_t_stat_stationary < critical_values_stationary[level]
            results.loc[level, 'ADF t (trend-stationary)'] += adf_t_stat_trend_stationary < critical_values_trend_stationary[level]
            results.loc[level, 'ADF δ (stationary)'] += adf_delta_stat_stationary < critical_values_stationary[level]
            results.loc[level, 'ADF δ (trend-stationary)'] += adf_delta_stat_trend_stationary < critical_values_trend_stationary[level]

    # Calculate the average of the results
    results = results.div(N)

    return results

#bond
results_bonds = simulate_and_test(bonds_mean_series, N=1000, T=252, p=1)
print("Results for Bonds:")
print(results_bonds)

#sp500
results_sp500 = simulate_and_test(sp500['Price'], N=1000, T=252, p=1)
print("Results for S&P 500:")
print(results_sp500)
