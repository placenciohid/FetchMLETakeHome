import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Import and Preprocessing
df = pd.read_csv('data_daily.csv')
df['# Date'] = pd.to_datetime(df['# Date'])
df.rename(columns={'# Date': 'Date'}, inplace=True)
df.set_index('Date', inplace=True)

# ARIMA Model Functions
def fit_arima(history, p, d, q):
    ar_params = np.array([0.5])  # Example parameter, adjust as needed
    ma_params = np.array([0.5])  # Example parameter, adjust as needed
    residuals = np.zeros(q)
    return ar_params, ma_params, residuals

def forecast_arima(history, ar_params, ma_params, residuals, p, d, q):
    diff = np.diff(history, n=d)
    ar_component = np.dot(ar_params, diff[-p:][::-1])
    ma_component = np.dot(ma_params, residuals)
    forecast = history[-1] + ar_component + ma_component
    return forecast

# STL Model Functions
def fit_STL(history, p, d, q):
    trend = np.array([0.5])  # Example parameter, adjust as needed
    seasonal = np.array([0.5])  # Example parameter, adjust as needed
    residuals = np.zeros(q)
    return trend, seasonal, residuals

def forecast_STL(history, trend, seasonal, residuals, p, d, q):
    diff = np.diff(history, n=d)
    trend_component = np.dot(trend, diff[-p:][::-1])
    seasonal_component = np.dot(seasonal, residuals)
    forecast = history[-1] + trend_component + seasonal_component
    return forecast

# Data Splitting
test_length = 65  # 22% of data for testing
train = df['Receipt_Count'][:-test_length]
test = df['Receipt_Count'][-test_length:]

# Forecasting
arima_forecasts = expanding_window_arima(train, test_length)  # Add expanding_window_arima definition
stl_forecasts = expanding_window_STL(train, test_length, p=1, d=1, q=1)  # Add expanding_window_STL definition

# Stacking and Evaluating Forecasts
final_forecasts = stack_forecasts(arima_forecasts, stl_forecasts)  # Add stack_forecasts definition

mae, rmse = evaluate_forecasts(test, final_forecasts)  # Add evaluate_forecasts definition
print(f'The MAE and RMSE for the test set are {mae:.2f} and {rmse:.2f}, respectively.')

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(df.index[-test_length:], df['Receipt_Count'][-test_length:], label='Actual')
plt.plot(df.index[-test_length:], final_forecasts, label='Predicted')
plt.title('Actual vs Predicted in Test Set - Last 65 Days')
plt.legend()
plt.savefig('actual_vs_predicted.png')
plt.show()



# Prediction for 2022
steps_ahead = 365  
arima_forecasts_2022 = expanding_window_arima(df['Receipt_Count'], steps_ahead)  
stl_forecasts_2022 = expanding_window_STL(df['Receipt_Count'], steps_ahead, p=1, d=1, q=1)  

final_forecasts_2022 = stack_forecasts(arima_forecasts_2022, stl_forecasts_2022) 

plt.figure(figsize=(15, 5))
plt.plot(pd.date_range('2022-01-01', periods=steps_ahead, freq='D'), final_forecasts_2022, label='Predicted')
plt.title('Predicted Receipt Count for 2022')
plt.savefig('Predicted.png')
plt.legend()
plt.show()


final_forecasts_2022 = pd.Series(final_forecasts_2022, index=pd.date_range('2022-01-01', periods=steps_ahead, freq='D'))
final_forecasts_2022.to_csv('final_forecasts.csv')
