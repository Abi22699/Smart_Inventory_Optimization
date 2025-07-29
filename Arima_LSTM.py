import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv('/content/GlobalTech_Inc_SalesData.csv')

# Inspect the dataset
print(data.head())

# Handle missing values
data = data.dropna()

# Convert date column to datetime if applicable
if 'Order Date' in data.columns:
    data['Order Date'] = pd.to_datetime(data['Order Date'])
else:
    raise ValueError("The dataset must contain an 'Order Date' column.")

# Ensure numeric columns are correctly formatted
if 'Total Sales' in data.columns:
    data['Total Sales'] = pd.to_numeric(data['Total Sales'], errors='coerce')
else:
    raise ValueError("The dataset must contain a 'Total Sales' column.")

# Check for duplicates and drop if necessary
data = data.drop_duplicates()

# Sort the data by date for time-series models
data = data.sort_values(by='Order Date')

# Summarize sales by product
if 'Product Name' in data.columns:
    product_sales = data.groupby('Product Name')['Total Sales'].sum().reset_index()
    top_products = product_sales.sort_values(by='Total Sales', ascending=False)
    print(top_products.head())
else:
    raise ValueError("The dataset must contain a 'Product Name' column.")

# Define function to classify data complexity
def is_complex(data_series):
    variance = np.var(data_series)
    return variance > 1_000_000  # Adjust threshold based on data insights

# Define function to create dataset for LSTM
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Choose model based on complexity
if is_complex(data['Total Sales']):
    print("Using LSTM for prediction (Complex Data)")

    # Preprocess data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Total Sales']])

    # Prepare LSTM dataset
    time_step = 10
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train LSTM model
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    # Predict with LSTM
    lstm_predictions = lstm_model.predict(X)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    forecast_dates = pd.date_range(start=data['Order Date'].iloc[-1], periods=len(lstm_predictions), freq='D')
    final_forecast = pd.DataFrame({'Date': forecast_dates, 'Forecasted_Sales': lstm_predictions.flatten()})

else:
    print("Using ARIMA for prediction (Simple Data)")

    # Train ARIMA model
    arima_model = ARIMA(data['Total Sales'], order=(5, 1, 0))
    arima_result = arima_model.fit()

    # Forecast with ARIMA
    forecast_steps = 30
    arima_forecast = arima_result.forecast(steps=forecast_steps)

    forecast_dates = pd.date_range(start=data['Order Date'].iloc[-1], periods=forecast_steps, freq='D')
    final_forecast = pd.DataFrame({'Date': forecast_dates, 'Forecasted_Sales': arima_forecast})

# Merge forecasted sales with the original dataset
data = pd.merge(data, final_forecast, left_on='Order Date', right_on='Date', how='left')

# Summarize forecasted sales by product
if 'Forecasted_Sales' in data.columns:
    product_sales_forecast = data.groupby('Product Name')['Forecasted_Sales'].sum().reset_index()
    product_sales_forecast = product_sales_forecast.sort_values(by='Forecasted_Sales', ascending=False)

    # Plot forecasted sales by product
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Product Name', y='Forecasted_Sales', data=product_sales_forecast, palette='Blues_d')
    plt.title('Forecasted Sales by Product')
    plt.xlabel('Product Name')
    plt.ylabel('Forecasted Sales')
    plt.xticks(rotation=90)
    plt.show()

    # Plot forecasted sales trend
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Forecasted_Sales', data=final_forecast, marker='o')
    plt.title('Forecasted Sales Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Forecasted Sales')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
else:
    print("Error: 'Forecasted_Sales' column not found in the dataset.")
