import pandas as pd

# Read the original dataset
original_df = pd.read_csv('/Users/aashvi/Desktop/Senior Project/Sharps-Test-Values - STSS (1).csv')

# Convert the 'Date' column to datetime format
original_df['Date'] = pd.to_datetime(original_df['Date'])

# Set 'Date' column as the index
original_df.set_index('Date', inplace=True)

# Resample the DataFrame to fill in missing dates and forward fill missing values
resampled_df = original_df.resample('D').ffill()

# Truncate stock prices to two decimal points
resampled_df['Open'] = resampled_df['Open'].round(2)

# Write the resampled DataFrame to a new CSV file
resampled_df.to_csv('/Users/aashvi/Desktop/Senior Project/Sharps-Test-Data-Modified - Sheet1.csv')

print("New dataset created successfully.")




