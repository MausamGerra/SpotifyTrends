import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

#---------------------------------------------Top_1000---------------------------------------------------
df = pd.read_csv('spotify_top_1000.csv')

#Here, since we need only few selected columns to determine whether the song will be popular or not, we are dropping these columns
df.drop(columns=['speechiness', 'track_id', 'mode', 'valence', 'tempo', 'time_signature', 'instrumentalness'], inplace=True)
print(df.head())
selected_columns = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'acousticness', 'liveness', 'duration_ms', 'streaming_count']

means = df[selected_columns].mean()

df.fillna(means, inplace=True)
print(df)
df['duration_minutes'] = df['duration_ms'] / 60000

scaler = StandardScaler()
df[['acousticness', 'danceability', 'energy', 'loudness', 'liveness']] = scaler.fit_transform(df[['acousticness', 'danceability', 'energy', 'loudness', 'liveness']])

df.to_csv('spotify_top_1000_update.csv', index=False)

# Assume 'df' is your DataFrame and 'target' is the column you're trying to predict
X = df.drop(columns=['popularity'])  # Features
y = df['popularity']  # Target variable

model = LinearRegression()
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X = df.drop(columns=['track_name', 'artist_name'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

# Predictions on both training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate on training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Evaluate on test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"Testing MSE: {test_mse:.4f}, R²: {test_r2:.4f}")


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")

residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

feature_importance = pd.Series(model.coef_, index=X_train.columns)
feature_importance.plot(kind='bar')
plt.title("Feature Importance")
plt.show()

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

sns.pairplot(df[['popularity', 'energy', 'danceability', 'acousticness', 'loudness']])
plt.title("Pair Plot of Key Features")
plt.show()

#---------------------------------------------UPDATED---------------------------------------------------

df = pd.read_csv('spotify_top_1000_update.csv')

# Parameters
num_weeks = 16  # Number of weeks to distribute stream counts into
np.random.seed(42)  # For reproducibility (optional)

# Function to generate weekly stream counts with exponential decay
def generate_weekly_stream_counts(total_stream_count, num_weeks=16, decay_rate=0.1):
    # Create an exponential decay factor for each week
    weeks = np.arange(1, num_weeks + 1)
    decay_factors = np.exp(-decay_rate * weeks)  # Exponential decay function
    
    # Normalize the decay factors to ensure the total stream count is preserved
    decay_factors_normalized = decay_factors / decay_factors.sum()
    
    # Calculate the weekly stream counts based on the normalized decay factors
    weekly_stream_counts = decay_factors_normalized * total_stream_count
    
    return weekly_stream_counts

# Example usage for generating weekly stream counts with exponential decay
num_weeks = 16  # Number of weeks to distribute stream counts into
decay_rate = 0.05  # Decay rate (can be adjusted for steeper or slower decay)

weekly_stream_counts = []

# Apply the exponential decay function to each song's total stream count
for _, row in df.iterrows():
    song_stream_count = row['streaming_count']
    weekly_counts = generate_weekly_stream_counts(song_stream_count, num_weeks, decay_rate)
    weekly_stream_counts.append(weekly_counts)

# Add the weekly stream counts to the DataFrame
for i in range(num_weeks):
    df[f'Week_{i+1}'] = [weekly_counts[i] for weekly_counts in weekly_stream_counts]

# Calculate the mean stream count for each song and add it to a new column
df['Weekly Mean'] = df[[f'Week_{i+1}' for i in range(num_weeks)]].mean(axis=1)


# Save the updated DataFrame back to a new CSV
df.to_csv('spotify_top_1000_weekly.csv', index=False)

# Show the updated DataFrame
print(df.head())

df = pd.read_csv('spotify_top_1000_weekly.csv')

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
selected_columns=['popularity', 'danceability', 'energy', 'key', 'loudness', 'acousticness', 'liveness', 'duration_ms', 'streaming_count']
sns.heatmap(df[selected_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

sns.histplot(df['popularity'], kde=True)
plt.title('Popularity Distribution')
plt.show()

sns.scatterplot(x='energy', y='popularity', data=df)
plt.title('Energy vs Popularity')
plt.show()

#---------------------------------------------WEEKLY---------------------------------------------------

# Step 1: Load the data and prepare it for analysis
df = pd.read_csv('spotify_top_1000_weekly.csv')

# Step 2: Filter data for the specific song and extract the relevant columns (streaming counts for 16 weeks)
song_data = df[df['track_name'] == 'Taste'][['Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_5', 
                                                'Week_6', 'Week_7', 'Week_8', 'Week_9', 'Week_10', 
                                                'Week_11', 'Week_12', 'Week_13', 'Week_14', 'Week_15', 
                                                'Week_16', 'popularity']].reset_index(drop=True)

# Step 3: Reshape the data into a time series format (1 column for streaming counts, 1 column for popularity)
streaming_counts = song_data.iloc[0, :-1].values  # Get the streaming counts (first row)
popularity = song_data['popularity'].iloc[0]  # Get the popularity value (same for all weeks in this example)

# Create a time series index (weeks 1-16)
weeks = range(1, 17)

# Step 4: Prepare the exogenous variable (popularity)
# Since popularity is constant, we'll just replicate it for each of the 16 weeks
popularity_data = [popularity] * 16

# Convert to DataFrame for easy handling
ts_data = pd.DataFrame({
    'week': weeks,
    'streaming_count': streaming_counts,
    'popularity': popularity_data
})

# Step 5: Fit SARIMAX model (with popularity as exogenous variable)
# We will use (p, d, q) = (1, 1, 1) and seasonal order of (1, 1, 1, 4) since we have 4 quarters in a year or weekly seasonality
model = SARIMAX(ts_data['streaming_count'], exog=ts_data['popularity'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
model_fit = model.fit(disp=False)

# Step 6: Forecast for the next 4 weeks
steps_ahead = 10

# For simplicity, let's assume the popularity stays the same for the next 4 weeks
future_popularity = [popularity] * steps_ahead  # Use the same popularity for forecast

# Forecast the next 4 weeks using the SARIMAX model
forecast = model_fit.forecast(steps=steps_ahead, exog=future_popularity)
print(f"Predicted stream counts for the next {steps_ahead} weeks: {forecast.tolist()}")

# Step 7: Plot historical and forecasted data
historical_index = ts_data.index
future_index = range(len(ts_data), len(ts_data) + steps_ahead)

plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(historical_index, ts_data['streaming_count'], label='Historical Data', color='blue', linewidth=2)

# Plot forecasted data
plt.plot(future_index, forecast, label='Forecasted Data', color='red', linestyle='dashed', marker='o')

# Add a vertical line to separate historical and forecasted data
plt.axvline(x=len(ts_data)-1, color='gray', linestyle='dotted', linewidth=1.5, label='Forecast Start')

# Formatting the plot
plt.title('Stream Count: Historical vs. Forecasted Data')
plt.xlabel('Weeks')
plt.ylabel('Stream Count')
plt.legend()
plt.grid(True)
plt.show()

# Print the model summary for diagnostics
print(model_fit.summary())





