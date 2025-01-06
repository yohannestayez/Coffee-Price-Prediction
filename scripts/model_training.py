import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib



def load_and_clean_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # Replace NaN and infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)  # Drop rows with NaN value
    data = data[~data.index.duplicated(keep='first')]
    
    return data


def scale_data(df):
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(df)
    
    # Convert the scaled data back to a DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    return scaled_df

# Function to perform ADF test
def adf_test(series):
    # Perform the ADF test
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    return result[1] > 0.05  # Returns True if the series is non-stationary

# Define the validate_data_quality function
def validate_data_quality(df):
    quality_checks = {
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }
    return quality_checks

def fit_var_model(train_data, maxlags=3, forecast_steps=1000):
    model = VAR(train_data)
    try:
        lag_order = model.select_order(maxlags=maxlags)
        optimal_lag = lag_order.aic
        var_model = model.fit(optimal_lag)

        # Access the fitted values (predicted values) and assign the original column names
        fitted_values = pd.DataFrame(var_model.fittedvalues, columns=train_data.columns)

        # Forecast future values
        forecast = var_model.forecast(train_data.values[-optimal_lag:], steps=forecast_steps)

        # Create a date range for the forecasted values
        forecast_index = pd.date_range(start=train_data.index[-1], periods=forecast_steps + 1, freq='D')[1:]

        # Create forecast DataFrame with column names and date index
        forecast_df = pd.DataFrame(forecast, columns=train_data.columns, index=forecast_index)

        return fitted_values, forecast_df

    except np.linalg.LinAlgError:
        print("Handling LinAlgError by refitting with a lower lag or regularization.")
        return None, None
    

def plot_actual_vs_forecast(train_data, forecast_df, variables):
    num_vars = len(variables)
    fig, axes = plt.subplots(num_vars, 1, figsize=(15, num_vars * 3), sharex=True)

    for i, var in enumerate(variables):
        # Plot actual values
        axes[i].plot(train_data.index, train_data[var], label=f'Actual {var.capitalize()}', color='blue')

        # Plot forecasted values (shifted to match the forecast period)
        forecast_index = pd.date_range(start=train_data.index[-1], periods=len(forecast_df)+1, freq='B')[1:]  # Assuming business day frequency
        axes[i].plot(forecast_index, forecast_df[var], label=f'Forecast {var.capitalize()}', linestyle='--', color='orange')

        # Add legend and labels
        axes[i].legend(loc='upper left')
        axes[i].set_title(f'{var.capitalize()} - Actual vs Forecast')
        axes[i].set_ylabel(var.capitalize())

    # Set a limit for the x-axis if needed to zoom in on recent data
    axes[-1].set_xlim(pd.Timestamp('2015-01-01'), train_data.index[-1])

    # Improve spacing and layout
    fig.tight_layout()
    fig.suptitle("VAR Model Forecast vs Actual", fontsize=16)
    fig.subplots_adjust(top=0.95)  # Adjust to fit the suptitle

    # Show the plot
    plt.show()


# Plotting actual vs predicted values for model evaluation
def plot_forecast_vs_actual(train, test, forecast_df, columns):
    plt.figure(figsize=(20, 10))
    for col in columns:
        plt.plot(test.index, test[col], label=f'Actual {col}')
        forecast_index = pd.date_range(start=train.index[-1], periods=len(forecast_df)+1, freq='B')[1:]
        plt.plot(forecast_index, forecast_df[col], label=f'Forecast {col}', linestyle='--')
    plt.legend()
    plt.title("VAR Model Forecast vs Actual")
    plt.show()

# Model evaluation function with extended metrics
def evaluate_model(test, forecast_df, recency_window=5):
    evaluation_metrics = {}
    forecast_df = forecast_df.iloc[:len(test)]

    for col in test.columns:
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mean_squared_error(test[col].iloc[:len(forecast_df)], forecast_df[col]))
        
        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(test[col].iloc[:len(forecast_df)], forecast_df[col])
        
        # Mean Absolute Percentage Error (MAPE)
        mape = mean_absolute_percentage_error(test[col].iloc[:len(forecast_df)], forecast_df[col])
        
        # Recency RMSE (last 'recency_window' points)
        recent_rmse = np.sqrt(mean_squared_error(test[col].iloc[-recency_window:], forecast_df[col].iloc[-recency_window:]))

        evaluation_metrics[col] = {
            'Root Mean Squared Error ': rmse,
            'Mean Absolute Error ': mae,
            'Mean Absolute Percentage Error': mape,
            'Recency RMSE': recent_rmse
        }
        
        # Print the metrics
        print(f"Metrics for {col}:")
        print(f"  Root Mean Squared Error: {rmse}")
        print(f"  Mean Absolute Error: {mae}")
        print(f"  Mean Absolute Percentage Error: {mape}")
        print(f"  Recency RMSE (last {recency_window} values): {recent_rmse}\n")
        
    return evaluation_metrics