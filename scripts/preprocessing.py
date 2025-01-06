import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from arch import arch_model
from ruptures import Pelt
import warnings
warnings.filterwarnings('ignore')


def load_xlsx_files(directory):

  # Get a list of all XLSX files in the specified directory
  file_list = glob.glob(os.path.join(directory, '*.xlsx'))

  # Create an empty list to store DataFrames
  df_list = []

  # Check if there are any XLSX files found
  if not file_list:
    print("No XLSX files found in the specified directory.")
  else:
    # Read each XLSX file into a DataFrame and append to the list (if not empty)
    for file in file_list:
      try:
        df = pd.read_excel(file)
        df_list.append(df)
      except FileNotFoundError:
        print(f"File not found: {file}")

  # Only proceed with concatenation if there are DataFrames in the list
  if df_list:
    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)
    return df
  else:
    return pd.DataFrame()  # Return an empty DataFrame if no files were loaded


def preprocess_data(df):
    # Reset the index to make 'Date' a column again
    df.reset_index(inplace=True)

    # Convert columns to float where possible
    for column in df.columns:
        try:
            df[column] = df[column].str.replace(',', '').astype(float)
        except (ValueError, AttributeError):
            continue

    # Convert 'Trade Date' to datetime and clean the data
    df['Trade Date'] = pd.to_datetime(df['Trade Date'], errors='coerce')
    df = df.dropna(subset=['Trade Date'])
    df = df.sort_values(by='Trade Date')
    df = df.rename(columns={'Trade Date': 'Date'})
    df.reset_index(drop=True, inplace=True)

    # Ensure there are no missing values
    df = df.dropna()

    # Set 'Date' as the index again
    df.set_index('Date', inplace=True)
    
    return df



def plot_time_series(df):
    
    plt.figure(figsize=(20, 15))
    
    # Plot Price
    plt.subplot(4, 1, 1)
    plt.yticks(np.arange(0, 3000, step=300))
    plt.plot(df.index, df['Opening Price'], color='blue', label='Coffee Price')
    plt.title("Coffee Price Time Series")
    plt.ylabel("Opening Price")
    plt.legend()

    # Plot Closing price
    plt.subplot(4, 1, 2)
    plt.yticks(np.arange(0, 3000, step=300))
    plt.plot(df.index, df['Closing Price'], color='purple', label='Coffee Price')
    plt.ylabel("Closing Price")
    plt.legend()
    
    # Plot High
    plt.subplot(4, 1, 3)
    plt.yticks(np.arange(0, 3000, step=300))
    plt.plot(df.index, df['High'], color='green', label='Coffee Price')
    plt.xlabel("Date")
    plt.ylabel("High")
    plt.legend()

        # Plot Low
    plt.subplot(4, 1, 4)
    plt.yticks(np.arange(0, 3000, step=300))
    plt.plot(df.index, df['Low'], color='green', label='Coffee Price')
    plt.xlabel("Date")
    plt.ylabel("Low")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_acf_pacf(df, column='Opening Price'):
    # Create a figure with 2 subplots for ACF and PACF
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot the Autocorrelation Function (ACF)
    plot_acf(df[column], ax=ax[0])
    ax[0].set_title("Autocorrelation (ACF)")
    
    # Plot the Partial Autocorrelation Function (PACF)
    plot_pacf(df[column], ax=ax[1])
    ax[1].set_title("Partial Autocorrelation (PACF)")
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def correlation_analysis(df):
    # Select only float columns
    float_df = df.select_dtypes(include=['float64'])
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(float_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Engineered Features")
    plt.show()


# Distribution Analysis
def distribution_analysis(df):
    
    plt.figure(figsize=(20, 12))
    
    # Opening Price Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['Opening Price'], bins=50, color='blue', kde=True)
    plt.title("Opening Price Distribution")

    # Closing Price Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['Closing Price'], bins=50, color='purple', kde=True)
    plt.title("Closing Price Distribution")
    
    # High Price Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(df['High'], bins=50, color='green', kde=True)
    plt.title("High Price Distribution")
    
    # Low Price Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(df['Low'], bins=50, color='red', kde=True)
    plt.title("Low Price Distribution")
    
    plt.tight_layout()
    plt.show()


# Seasonality Analysis
def seasonality_analysis(df):
   
    decomposition = seasonal_decompose(df['Opening Price'], model='multiplicative', period=365)

    # Set up a wider figure
    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
    
    # Plot each component with wider axes
    decomposition.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    
    decomposition.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    
    decomposition.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    
    decomposition.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')
    
    plt.tight_layout()
    plt.show()


# Risk Metrics Calculation
def calculate_risk_metrics(df, confidence_level=0.05):
    var = np.percentile(df['Returns'].dropna(), 100 * confidence_level)
    es = df['Returns'][df['Returns'] <= var].mean()
    
    print(f"Value at Risk (VaR) at {confidence_level*100}% confidence level: {var:.4f}")
    print(f"Expected Shortfall (ES) at {confidence_level*100}% confidence level: {es:.4f}")


# Rolling Correlation
def rolling_correlation(df, window=21):
    df['Rolling_Corr'] = df['Volatility'].rolling(window).corr(df['Returns'])
    
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Rolling_Corr'], color='red', label=f'Rolling Correlation (window={window})')
    plt.title("Rolling Correlation between Volatility and Returns")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    plt.show()


def check_stationarity(df, column='Closing Price'):
    result = adfuller(df[column])
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] < 0.05:
        print("The data is stationary.")
    else:
        print("The data is non-stationary. Differencing may be needed.")


def plot_with_events(df, events_df):
    # Ensure 'date' column in events_df is in datetime format
    events_df['date'] = pd.to_datetime(events_df['date'])
    
    plt.figure(figsize=(20, 16))  # Increase the height to make the graph taller
    plt.plot(df.index, df['Opening Price'], label='Coffee Opening Price', color='blue')
    
    y_max = df['Opening Price'].max()
    y_min = df['Opening Price'].min()

    # Add padding to the y-axis for space above and below
    y_padding = (y_max - y_min) * 0.3  # 30% padding for more vertical space
    plt.ylim(y_min - y_padding, y_max + y_padding)

    for _, row in events_df.iterrows():
        event_date = row['date']
        event_text = row['event']
        
        # Add a vertical line at the event date
        plt.axvline(event_date, color='green', linestyle='--', alpha=0.6)

        # Add horizontal line connecting text to the event line
        text_y = y_max + y_padding * 0.4  # Place the text above the graph
        plt.plot([event_date, event_date], [y_min, text_y], color='green', linestyle=':', alpha=0.5)
        
        # Place event text slightly above the graph
        plt.text(event_date, text_y, event_text, rotation=90, verticalalignment='bottom', fontsize=12, color='black')
    
    plt.title("Coffee Opening Prices with Significant Events (2012-2019)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD per Ton)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Event Impact on Volatility and Returns
def event_impact_analysis(df, events_df):
    impacts = []
    for _, event in events_df.iterrows():
        date = event['date']
        # Define a window around the event date
        before = df.loc[date - pd.Timedelta(days=5):date - pd.Timedelta(days=1)]
        after = df.loc[date:date + pd.Timedelta(days=5)]
        
        # Calculate average volatility and returns before and after the event
        before_volatility = before['Volatility'].mean()
        after_volatility = after['Volatility'].mean()
        before_return = before['Returns'].mean()
        after_return = after['Returns'].mean()
        
        # Calculate average prices before and after the event
        before_opening_price = before['Opening Price'].mean()
        after_opening_price = after['Opening Price'].mean()
        before_closing_price = before['Closing Price'].mean()
        after_closing_price = after['Closing Price'].mean()
        before_high_price = before['High'].mean()
        after_high_price = after['High'].mean()
        
        impacts.append({
            "Event": event['event'],
            "Date": date,
            "Before_Opening_Price": before_opening_price,
            "After_Opening_Price": after_opening_price,
            "Before_Closing_Price": before_closing_price,
            "After_Closing_Price": after_closing_price,
            "Before_High_Price": before_high_price,
            "After_High_Price": after_high_price,
            "Before_Volatility": before_volatility,
            "After_Volatility": after_volatility,
            "Before_Return": before_return,
            "After_Return": after_return,
        })
    
    impact_df = pd.DataFrame(impacts)
    return impact_df
