# Ethiopian Coffee Stock Price Analysis and Prediction  

## Overview  

This project analyzes Ethiopian coffee stock prices (2012–2019) using time-series techniques to uncover patterns, trends, and interdependencies. It preprocesses the data, conducts EDA, and employs a Vector Autoregression (VAR) model to forecast prices based on historical data, including opening, closing, high, and low prices. Insights aim to enhance decision-making and forecasting in the Ethiopian coffee market.  

## Folder Structure
.
├── .github                     # Contains GitHub-specific configurations, such as workflows or issue templates.
├── data                        # Directory for storing raw and processed data files.
├── notebooks                   
│   ├── Data_Preprocessing.ipynb        # Notebook for data cleaning, transformation, and preprocessing.
│   └── Training_model_with_VAR.ipynb   # Notebook for training and evaluating the VAR model for forecasting.
├── scripts                     # Contains Python scripts for modularized functions and reusable code.
│   ├── __init__.py                     # Marks the folder as a package for importable modules.
│   ├── model_training.py               # Script for training the VAR model and generating forecasts.
│   └── preprocessing.py                # Script for preprocessing raw data into a usable format.
├── src                        
│   └── __init__.py                    # Marks the folder as a package for importable modules.
├── tests                      
│   └── __init__.py                    # Marks the folder as a package for testing modules.
├── .gitignore                 # Specifies intentionally untracked files to ignore in the Git repository.
├── README.md                  # Project documentation providing an overview, setup instructions, and usage details.
└── requirements.txt           # File listing Python dependencies required for the project.


## Dataset Description
- **Source:** Ethiopian coffee stock prices (2012–2019).
- **Fields:**
  - **Date:** The trading date of the record.
  - **Opening Price:** Price at the start of the trading session.
  - **Closing Price:** Price at the end of the trading session.
  - **High:** Highest price achieved during the session.
  - **Low:** Lowest price achieved during the session.

## Notebooks 

### 1. Data_preproccessing.ipynb

  1. **Data Loading**  
  - **Purpose**: Load multiple Excel files containing Ethiopian coffee stock price data into a single DataFrame.  
  - **Key Functions**: 
      - `load_xlsx_files(directory)`: Combines data from all `.xlsx` files in the specified directory into a unified dataset.  

   2. **Data Preprocessing**  
  - **Purpose**: Clean and structure the data for analysis.  
  - **Steps Performed**:  
    - Convert columns to numeric data types.  
    - Handle missing values and ensure date columns are properly formatted.  
    - Set `Date` as the index for time-series operations.  
  - **Key Function**:  
    - `preprocess_data(df)`: Handles all preprocessing tasks for the loaded dataset.  

  3. **Exploratory Data Analysis (EDA)**  
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

  4. **Seasonality and Trend Analysis**  
  - **Purpose**: Decompose the time series to identify trends, seasonality, and residuals.  
  - **Key Function**:  
    - `seasonality_analysis(df)`: Applies seasonal decomposition to the opening price data.  

  5. **Risk and Volatility Metrics**  
  - **Purpose**: Assess the risk and volatility of coffee prices.  
  - **Key Functions**:  
    - `calculate_risk_metrics(df, confidence_level=0.05)`: Calculates Value at Risk (VaR) and Expected Shortfall (ES).  
    - `rolling_correlation(df, window=21)`: Computes rolling correlations for price volatility.  

  6. **Stationarity Testing**  
  - **Purpose**: Check the stationarity of price data to prepare for modeling.  
  - **Key Function**:  
    - `check_stationarity(df, column)`: Uses the Augmented Dickey-Fuller test to assess stationarity.  

  7. **Event Impact Analysis**  
  - **Purpose**: Evaluate how significant events impacted coffee prices, volatility, and returns.  
  - **Key Functions**:  
    - `plot_with_events(df, events_df)`: Annotates price trends with significant events.  
    - `event_impact_analysis(df, events_df)`: Calculates pre- and post-event price changes, volatility, and returns.  


### Training_model_with_VAR.ipynb

  1. **Setup and Imports**
  - Imported necessary libraries such as `pandas`, `numpy`, `matplotlib`, and `seaborn` for data manipulation and visualization.
  - Imported `statsmodels` for building and training the VAR model.

  2. **Loading Preprocessed Data**
  - Preprocessed coffee stock prices data is loaded into a DataFrame.
  - Verified the integrity of the dataset to ensure readiness for analysis.
  3. **Exploratory Data Analysis (EDA)**
  - Visualized time-series data to identify trends, seasonality, and potential anomalies.
  - Confirmed stationarity of data using the Augmented Dickey-Fuller (ADF) test.

  4. **Train-Test Split**
  - Divided the dataset into training and testing sets for model validation.
  - Ensured temporal integrity by using earlier data for training and later data for testing.

  5. **Vector Autoregression (VAR) Model Training**
  - **Model Selection:**
    - Determined optimal lag order using the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).
    - Iteratively tested different lag values and selected the best-performing model.

  - **Training the VAR Model:**
    - Trained the model using the training dataset.
    - Fitted the model to capture interdependencies between variables like opening price, closing price, high, and low prices.

  6. **Forecasting with VAR**
  - Generated multi-step forecasts for coffee prices based on the testing dataset.
  - Used the forecast to predict future values for all variables (e.g., opening price, closing price).

  7. **Model Evaluation**
  - Compared predicted values with actual values from the testing dataset.
  - Metrics used for evaluation:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)

  - Visualized the accuracy of forecasts using plots, including:
    - Actual vs. Predicted Values.
    - Residual Analysis to check for model bias.

  8. **Conclusion and Insights**
  - Highlighted the model's performance in forecasting coffee prices.
  - Identified the strengths and limitations of using VAR for this dataset.
  - Discussed potential improvements, including testing alternative models or refining preprocessing techniques.


## Key Outcomes
- **Stationarity:** Confirmed through transformations and testing.
- **Optimal Lag Selection:** Identified through AIC/BIC minimization.
- **Forecast Accuracy:** Achieved reasonable predictions for coffee prices with minor deviations.
- **Interdependencies:** Captured relationships between opening price, closing price, high, and low variables effectively.

---

## How to Use
1. **Dependencies:** Ensure the following libraries are installed:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - statsmodels

2. **Preprocessed Data:** open and run the first notebook(Data_Preprocessing.ipynb) to get the preprocessed data.

3. **Execution:** Run the second notebook(Training_model_with_VAR.ipynb) to:
   - Train the VAR model.
   - Generate predictions.
   - Evaluate model performance.

---

## Future Work
- Experiment with advanced time-series models like LSTM or Prophet for comparison.
- Incorporate external variables (e.g., global coffee prices, weather data) to improve forecasting accuracy.
- Extend the analysis to include causal inference between coffee price variables.

---

For further inquiries or issues, feel free to reach out.

