# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 15-10-2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("Sunspots.csv")

# Clean up and rename columns
data = data.rename(columns={
    'Date': 'date',
    'Monthly Mean Total Sunspot Number': 'sunspots'
})

# Convert date to datetime and sort
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna(subset=['date', 'sunspots'])
data = data.sort_values('date')

# --- Plot 1: Time Series ---
plt.figure(figsize=(10,5))
plt.plot(data['date'], data['sunspots'], label='Sunspots')
plt.xlabel('Date')
plt.ylabel('Monthly Mean Total Sunspot Number')
plt.title('Sunspots Time Series')
plt.legend()
plt.show()

# --- Stationarity Test ---
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('\nADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['sunspots'])

# --- Plot 2: ACF ---
plot_acf(data['sunspots'], lags=30)
plt.title('Autocorrelation Function (ACF)')
plt.show()

# --- Plot 3: PACF ---
plot_pacf(data['sunspots'], lags=30, method='ywm')  # avoid warning
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# --- Train-Test Split ---
train_size = int(len(data) * 0.8)
train, test = data['sunspots'][:train_size], data['sunspots'][train_size:]

# --- SARIMA Model ---
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_result = sarima_model.fit(disp=False)

# --- Forecast ---
predictions = sarima_result.predict(start=len(train), end=len(train)+len(test)-1)

# --- Evaluate ---
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('\nRMSE:', rmse)

# --- Plot 4: Actual vs Predicted ---
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Index')
plt.ylabel('Monthly Mean Total Sunspot Number')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```
### OUTPUT:
<img width="850" height="468" alt="image" src="https://github.com/user-attachments/assets/6b9791ad-92d5-465a-9373-e685d0a8ae3f" />

<img width="568" height="433" alt="image" src="https://github.com/user-attachments/assets/47e922d7-d9e9-4f1d-aad7-9d06c7a1bef8" />

<img width="568" height="433" alt="image" src="https://github.com/user-attachments/assets/4b3a439b-3cb0-4c9a-b597-232ba77f74f0" />

<img width="850" height="468" alt="image" src="https://github.com/user-attachments/assets/7a71eec5-f2f3-4226-b883-e299f1dc00fa" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
