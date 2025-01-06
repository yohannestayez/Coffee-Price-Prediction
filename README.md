# Ethiopian Coffee Stock Price Analysis and Prediction  

## Overview  
This notebook focuses on analyzing Ethiopian coffee stock prices from 2012 to 2019. The primary goal is to preprocess the dataset, conduct exploratory data analysis (EDA), and develop models for predicting coffee prices. It incorporates various statistical and machine-learning techniques to clean, analyze, and model the data effectively.  

The analysis includes time-series analysis, distribution exploration, event impact evaluation, and seasonality decomposition. The insights derived from this study are intended to support decision-making and forecasting in the Ethiopian coffee market.  



## Notebooks 

### Data_preproccessing.ipynb

#### 1. **Data Loading**  
- **Purpose**: Load multiple Excel files containing Ethiopian coffee stock price data into a single DataFrame.  
- **Key Functions**: 
    - `load_xlsx_files(directory)`: Combines data from all `.xlsx` files in the specified directory into a unified dataset.  

#### 2. **Data Preprocessing**  
- **Purpose**: Clean and structure the data for analysis.  
- **Steps Performed**:  
  - Convert columns to numeric data types.  
  - Handle missing values and ensure date columns are properly formatted.  
  - Set `Date` as the index for time-series operations.  
- **Key Function**:  
  - `preprocess_data(df)`: Handles all preprocessing tasks for the loaded dataset.  

#### 3. **Exploratory Data Analysis (EDA)**  
- **Purpose**: Gain insights into the data through visualization and statistical analysis.  
- **Steps Performed**:  
  - Visualize time series of opening, closing, high, and low prices.  
  - Plot ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function).  
  - Generate a correlation heatmap and histograms of price distributions.  
- **Key Functions**:  
  - `plot_time_series(df)`: Visualizes the time series for different price metrics.  
  - `plot_acf_pacf(df)`: Displays autocorrelation and partial autocorrelation.  
  - `correlation_analysis(df)`: Examines correlations between features.  
  - `distribution_analysis(df)`: Explores price distributions using histograms.  

#### 4. **Seasonality and Trend Analysis**  
- **Purpose**: Decompose the time series to identify trends, seasonality, and residuals.  
- **Key Function**:  
  - `seasonality_analysis(df)`: Applies seasonal decomposition to the opening price data.  

#### 5. **Risk and Volatility Metrics**  
- **Purpose**: Assess the risk and volatility of coffee prices.  
- **Key Functions**:  
  - `calculate_risk_metrics(df, confidence_level=0.05)`: Calculates Value at Risk (VaR) and Expected Shortfall (ES).  
  - `rolling_correlation(df, window=21)`: Computes rolling correlations for price volatility.  

#### 6. **Stationarity Testing**  
- **Purpose**: Check the stationarity of price data to prepare for modeling.  
- **Key Function**:  
  - `check_stationarity(df, column)`: Uses the Augmented Dickey-Fuller test to assess stationarity.  

#### 7. **Event Impact Analysis**  
- **Purpose**: Evaluate how significant events impacted coffee prices, volatility, and returns.  
- **Key Functions**:  
  - `plot_with_events(df, events_df)`: Annotates price trends with significant events.  
  - `event_impact_analysis(df, events_df)`: Calculates pre- and post-event price changes, volatility, and returns.  



