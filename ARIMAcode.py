import pandas as pd
from pandas import read_csv
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error
# Load dataset for crypto weekly marketcap
series = read_csv('/Users/amitjain/Desktop/Aashvi-SeniorProject/New-Tesla-Stock-Prices - Sheet1.csv', header=0, parse_dates=[0], index_col=0)

# Convert index to datetime if not already in datetime format
series.index = pd.to_datetime(series.index)

# Use asfreq to set the frequency and handle missing dates
series = series.asfreq('D')

# Forward fill missing values (assuming missing dates have the same value as the last available date)
series = series.ffill()
# Define the prediction start and end dates
prediction_start_date = '2023-09-01'
prediction_end_date = '2024-02-29'

# Generate the prediction index with daily frequency
prediction_index = pd.date_range(prediction_start_date, prediction_end_date, freq='D')

# Using auto_arima to find value of P, D, and Q
stepwise_model = auto_arima(series, start_p=1, start_q=1,
                            max_p=3, max_q=3, m=100,
                            start_P=0, seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True,
                            exogenous=series.index)

print(stepwise_model.summary())

# Fit the model using the dataset index as the datetime index
stepwise_model.fit(series)

# Make predictions using the datetime prediction index
future_forecast = stepwise_model.predict(n_periods=len(prediction_index), index=prediction_index)
print("Forecasted Values:", future_forecast)

# Create a DataFrame for the predictions with the prediction index
future_forecast_df = pd.DataFrame(future_forecast, index=prediction_index, columns=['Forecast'])


# Plot the forecasted values
plt.plot(series, label='Observed')
plt.plot(future_forecast_df, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title ('Tesla Technology Forecasted Stock Prices')
plt.legend()
plt.show()

# Print the DataFrame containing the forecasted values
print(future_forecast_df)

# Load testing data from a separate dataset
testing_data = read_csv('/Users/amitjain/Desktop/Aashvi-SeniorProject/Tesla-Test-Data-Modified - Sheet1.csv', header=0, parse_dates=[0], index_col=0)

# Define actual values from testing dataset
actual_values = testing_data['Open']

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(actual_values, future_forecast_df)

# Print MAE and MPE
print("Mean Absolute Error (MAE):", mae)

import numpy as np

# Calculate Mean Percentage Error (MPE)
def mean_percentage_error(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    return np.mean((forecast - actual) / actual) * 100

mpe = mean_percentage_error(actual_values, future_forecast_df)
print("Mean Percentage Error (MPE): {:.2f}%".format(mpe))

# Plotting the Graph
plt.plot(series, label='Training Data (Observed)')
plt.plot(future_forecast_df, label='ARIMA Forecast')
plt.plot(testing_data.index, testing_data['Open'], label='Testing Data (Actual)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Tesla Technologies Stock Prices Forecast')
plt.legend()
plt.show()


future_forecast_df.to_csv('/Users/amitjain/Desktop/Aashvi-SeniorProject/Tesla Forecasted Stock Prices - Sheet1.csv')  
