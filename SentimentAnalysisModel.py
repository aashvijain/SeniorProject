import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Load the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to calculate sentiment score for a given text
def calculate_sentiment(text):
    scores = sid.polarity_scores(text)
    return scores['compound']

# Function to calculate sentiment category for a given text
def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Read the dataset from the CSV file
df = pd.read_csv('/Users/aashvi/Desktop/Senior Project/Meta News Articles - Sheet1.csv')

# Calculate sentiment score for each news article
df['Sentiment Score'] = df['News Article Title'].apply(calculate_sentiment)

# Categorize sentiment score
df['Sentiment'] = df['Sentiment Score'].apply(categorize_sentiment)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Calculate average sentiment score for each date
average_sentiment = df.groupby('Date')['Sentiment'].apply(lambda x: x.value_counts().index[0]).reset_index()

# Print the new dataset
print(average_sentiment)

# Save the results to a new CSV file
average_sentiment.to_csv('/Users/aashvi/Desktop/Senior Project/Meta Sentiment Score - Sheet1.csv', index=False)
