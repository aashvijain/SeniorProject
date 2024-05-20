import pandas as pd

# Load the datasets
forecasted_df = pd.read_csv('/Users/aashvi/Desktop/Senior Project/TeslaForecastedPrices - Tesla Forecasted Stock Prices - Sheet1.csv')
actual_df = pd.read_csv('/Users/aashvi/Desktop/Senior Project/Tesla-Test-Data-Modified - Sheet1.csv')
tone_df = pd.read_csv('/Users/aashvi/Desktop/Senior Project/Tesla Sentiment Scores - Sheet1.csv')

# Ensure the Date columns are in datetime format and set as index
forecasted_df['Date'] = pd.to_datetime(forecasted_df['Date'])
actual_df['Date'] = pd.to_datetime(actual_df['Date'])
tone_df['Date'] = pd.to_datetime(tone_df['Date'])

forecasted_df.set_index('Date', inplace=True)
actual_df.set_index('Date', inplace=True)
tone_df.set_index('Date', inplace=True)

# Calculate trends for forecasted and actual prices
forecasted_df['Trend'] = (forecasted_df['Forecast'].diff() >= 0).astype(int)  # 1 if up/same, 0 if down
actual_df['Trend'] = (actual_df['Open'].diff() >= 0).astype(int)         # 1 if up/same, 0 if down

# Join all datasets into a single DataFrame
combined_df = forecasted_df.join(actual_df, rsuffix='_actual').join(tone_df)

# Define a function to check tone
def tone_matches(row):
    # Assume 'Tone' column has values 'positive', 'neutral', 'negative'
    # Checking if negative tone corresponds to forecasted up and actual down
    if row['Sentiment'] == 'Negative' and row['Trend'] == 1 and row['Trend_actual'] == 0:
        return 1
    elif row['Sentiment'] == 'Positive' and row['Trend'] == 0 and row['Trend_actual'] == 1:
        return 1
    return 0

# Apply function
combined_df['Tone_Matches'] = combined_df.apply(tone_matches, axis=1)

# Calculate the percentage of matching cases
matching_percentage = (combined_df['Tone_Matches'].sum() / len(combined_df)) * 100

print(f"The tone of the news articles accounts for the differences between the forecasted and real stock prices {matching_percentage:.2f}% of the time.")

# Optional: Save the results to a new CSV for further analysis
#combined_df.to_csv('analysis_results.csv')
