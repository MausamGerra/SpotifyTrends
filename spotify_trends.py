import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SpotifyDataAnalyzer:
    def __init__(self, file_path, log_level=logging.INFO):
        """
        Initialize the Spotify Data Analyzer

        Args:
            file_path (str): Path to the Spotify CSV file
            log_level (int): Logging level
        """
        self.logger = self._setup_logger(log_level)
        self.file_path = file_path
        self.df = None

    def _setup_logger(self, log_level):
        """
        Setup logging configuration

        Args:
            log_level (int): Logging level

        Returns:
            logging.Logger: Configured logger
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)

    def load_data(self, drop_columns=None):
        """
        Load and preprocess Spotify data

        Args:
            drop_columns (list, optional): Columns to drop from the dataset
        """
        try:
            self.df = pd.read_csv(self.file_path)

            if drop_columns:
                self.df.drop(columns=drop_columns, inplace=True)

            # Convert duration to minutes
            self.df['duration_minutes'] = self.df['duration_ms'] / 60000

            self.logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.file_path}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("The CSV file is empty.")
            raise

    def normalize_features(self, features):
        """
        Normalize specified features using StandardScaler

        Args:
            features (list): List of feature column names
        """
        scaler = StandardScaler()
        self.df[features] = scaler.fit_transform(self.df[features])
        self.logger.info(f"Normalized features: {features}")

    def build_regression_models(self, target='popularity', test_size=0.2):
        """
        Build and evaluate multiple regression models

        Args:
            target (str): Target variable name
            test_size (float): Proportion of test data

        Returns:
            dict: Performance metrics for different models
        """
        # Prepare features and target
        X = self.df.drop(columns=[target, 'track_name', 'artist_name'])
        y = self.df[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1)
        }

        results = {}
        for name, model in models.items():
            # Create a pipeline with polynomial features and model
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('regressor', model)
            ])

            # Fit model
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred = pipeline.predict(X_test)

            # Evaluate
            results[name] = {
                'MSE': mean_squared_error(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'MAPE': mean_absolute_percentage_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }

            # Cross-validation score
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
            results[name]['CV_MSE'] = -cv_scores.mean()

        return results

    def generate_weekly_streams(self, num_weeks=16, decay_rate=0.05):
        """
        Generate weekly stream counts with exponential decay

        Args:
            num_weeks (int): Number of weeks to distribute stream counts
            decay_rate (float): Rate of exponential decay

        Returns:
            pd.DataFrame: DataFrame with weekly stream counts
        """

        def _generate_weekly_counts(total_stream_count):
            weeks = np.arange(1, num_weeks + 1)
            decay_factors = np.exp(-decay_rate * weeks)
            decay_factors_normalized = decay_factors / decay_factors.sum()
            return decay_factors_normalized * total_stream_count

        weekly_stream_counts = self.df['streaming_count'].apply(_generate_weekly_counts)

        # Add weekly columns
        for i in range(num_weeks):
            self.df[f'Week_{i + 1}'] = weekly_stream_counts.apply(lambda x: x[i])

        self.df['Weekly_Mean'] = self.df[[f'Week_{i + 1}' for i in range(num_weeks)]].mean(axis=1)

        return self.df

    def forecast_streams(self, track_name, steps_ahead=10):
        """
        Forecast stream counts for a specific track

        Args:
            track_name (str): Name of the track to forecast
            steps_ahead (int): Number of weeks to forecast

        Returns:
            np.ndarray: Forecasted stream counts
        """
        song_data = self.df[self.df['track_name'] == track_name]

        if len(song_data) == 0:
            self.logger.error(f"Track '{track_name}' not found in the dataset.")
            return None

        streaming_counts = song_data.iloc[0, song_data.columns.str.contains('Week_')].values
        popularity = song_data['popularity'].iloc[0]

        # Prepare time series data
        ts_data = pd.DataFrame({
            'week': range(1, len(streaming_counts) + 1),
            'streaming_count': streaming_counts,
            'popularity': [popularity] * len(streaming_counts)
        })

        # Fit SARIMAX model
        model = SARIMAX(
            ts_data['streaming_count'],
            exog=ts_data['popularity'],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 4)
        )
        model_fit = model.fit(disp=False)

        # Forecast
        future_popularity = [popularity] * steps_ahead
        forecast = model_fit.forecast(steps=steps_ahead, exog=future_popularity)

        return forecast

    def visualize_results(self, output_dir='results'):
        """
        Generate various visualizations

        Args:
            output_dir (str): Directory to save visualization plots
        """
        os.makedirs(output_dir, exist_ok=True)

        # Correlation Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.close()

        # Popularity Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['popularity'], kde=True)
        plt.title('Popularity Distribution')
        plt.savefig(os.path.join(output_dir, 'popularity_distribution.png'))
        plt.close()

        # Scatter: Energy vs Popularity
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='energy', y='popularity', data=self.df)
        plt.title('Energy vs Popularity')
        plt.savefig(os.path.join(output_dir, 'energy_vs_popularity.png'))
        plt.close()


def main():
    # Example usage
    analyzer = SpotifyDataAnalyzer('spotify_top_1000.csv')

    # Load and preprocess data
    drop_columns = ['speechiness', 'track_id', 'mode', 'valence', 'tempo', 'time_signature', 'instrumentalness']
    analyzer.load_data(drop_columns=drop_columns)

    # Normalize features
    norm_features = ['acousticness', 'danceability', 'energy', 'loudness', 'liveness']
    analyzer.normalize_features(norm_features)

    # Run regression models
    regression_results = analyzer.build_regression_models()
    for model, metrics in regression_results.items():
        print(f"\n{model} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    # Generate weekly streams
    weekly_data = analyzer.generate_weekly_streams()
    weekly_data.to_csv('spotify_top_1000_weekly.csv', index=False)

    # Forecast for a specific track
    track_forecast = analyzer.forecast_streams('Taste')
    if track_forecast is not None:
        print("\nForecasted Stream Counts for 'Taste':")
        for week, streams in enumerate(track_forecast, 1):
            print(f"Week {week}: {streams}")

    # Generate visualizations
    analyzer.visualize_results()


if __name__ == "__main__":
    main()