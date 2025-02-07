# Tesla Stock Price Prediction using GRU

## Overview
This project implements an end-to-end time series forecasting solution to predict Tesla's closing stock price. The work was developed and executed in Google Colab using Python, with the data sourced from a CSV file containing historical Tesla stock data.

## Data
- **Dataset:** Tesla stock data (CSV format).
- **Features Included:** Open, High, Low, Close, Volume.
- **Preprocessing Steps:**  
  - Renaming and formatting columns.
  - Converting date strings to datetime objects.
  - Handling missing values.
  - Exploratory Data Analysis (plots, candlestick charts, monthly averages, and correlation heatmaps).

## Feature Engineering
- **Technical Indicators:**
  - **RSI (Relative Strength Index):** Measures momentum.
  - **MACD (Moving Average Convergence Divergence):** Captures trend changes.
  - **Bollinger Bands:** Assesses volatility.
- Additional features derived from the data were used to enhance model performance.

## Model Development
- **Approach:** Time-series forecasting using a GRU (Gated Recurrent Unit) based deep learning model.
- **Data Preparation:**  
  - Created a sliding window of 15 days from the features (including technical indicators) to predict the next day’s closing price.
  - Normalized features and target variables separately.
- **Model Architecture:**
  - **Layer 1:** GRU with 32 units, `return_sequences=True`
  - **Layer 2:** GRU with 32 units, `return_sequences=True`
  - **Layer 3:** GRU with 32 units
  - **Dropout:** 20% dropout rate to reduce overfitting.
  - **Output Layer:** Dense layer with 1 neuron for the prediction.
- **Training Parameters:**
  - **Optimizer:** Adam
  - **Loss Function:** Mean Squared Error (MSE)
  - **Batch Size:** 32
  - **Epochs:** 200

## Evaluation
- **Metrics Used:**  
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Explained Variance
  - R2 Score
  - Mean Gamma Deviance
  - Mean Poisson Deviance
- **Results:**  
  - **Train RMSE:** ~27.37  
  - **Test RMSE:** ~18.98  
  - Additional metrics (MAE, R2, etc.) further validate the model's strong performance.

## How to Run
1. **Open the Notebook:** Launch the provided notebook in Google Colab.
2. **Data Setup:** Upload the Tesla stock CSV file to the working directory.
3. **Execute Cells:** Run the notebook cells sequentially to perform data preprocessing, feature engineering, model training, and evaluation.
4. **Visualizations:** The notebook automatically generates visualizations including technical indicator plots, training curves, and future forecasts.

## Conclusion
This project demonstrates a robust methodology for predicting Tesla's stock closing price using a GRU-based model. By integrating detailed feature engineering (with technical indicators like RSI, MACD, and Bollinger Bands) and carefully tuned hyperparameters, the solution achieves high predictive accuracy as shown by the evaluation metrics.
