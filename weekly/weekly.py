import pandas as pd
import numpy as np
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DB_FILE = "weekly_forecast.db"

data = pd.read_csv('final_sales_data.csv')
data['DATE'] = pd.to_datetime(data['DATE'])
data['NetAmount'] = data['NetAmount'].abs()

def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weekly_forecast (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            SalesPersonCode TEXT,
            Week TEXT,
            Predicted_NetAmount REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def save_forecast_to_db(sales_code, forecast_df):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for _, row in forecast_df.iterrows():
        try:
            week_str = row['Week'].strftime('%Y-%m-%d') if isinstance(row['Week'], pd.Timestamp) else str(row['Week'])
            
            cursor.execute('''INSERT OR IGNORE INTO weekly_forecast (SalesPersonCode, Week, Predicted_NetAmount) 
                              VALUES (?, ?, ?)''', (sales_code, week_str, row['Predicted NetAmount']))
        
        except Exception as e:
            print(f"Failed to insert row: {e}")

    conn.commit()
    conn.close()

def predict_weekly_sales(sales_code, last_date):
    person_data = data[data['SalesPersonCode'] == sales_code]

    if person_data.empty:
        print(f"No data found for SalesPerson {sales_code}.")
        return None

    forecast_week_start = last_date + pd.Timedelta(days=1)
    forecast_week_end = forecast_week_start + pd.Timedelta(days=6)

    weekly_sales = person_data.groupby(pd.Grouper(key='DATE', freq='W'))['NetAmount'].sum().reset_index()

    if len(weekly_sales) < 5:
        print(f"Not enough data for SalesPerson {sales_code}.")
        return None

    try:
        weekly_sales_series = weekly_sales.set_index('DATE')['NetAmount']

        model = ARIMA(weekly_sales_series, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)  

        results_df = pd.DataFrame({
            'Week': [forecast_week_start],
            'Predicted NetAmount': forecast.values
        })

        save_forecast_to_db(sales_code, results_df)
        return results_df

    except Exception as e:
        print(f"ARIMA failed for {sales_code} due to {e}. Switching to mean-based prediction...")
        
        mean_forecast = [weekly_sales['NetAmount'].mean()]

        results_df = pd.DataFrame({
            'Week': [forecast_week_start],
            'Predicted NetAmount': mean_forecast
        })

        save_forecast_to_db(sales_code, results_df)
        return results_df

def generate_weekly_forecasts(sales_codes):
    setup_database()
    forecasts = {}
    
    for sales_code in sales_codes:
        last_date = data[data['SalesPersonCode'] == sales_code]['DATE'].max()
        
        forecast = predict_weekly_sales(sales_code, last_date)
        if forecast is not None:
            forecasts[sales_code] = forecast
            print(f"\nWeekly Forecast for Sales Person {sales_code}:")
            print(f"Prediction Week: {forecast['Week'].dt.strftime('%Y-%m-%d').values[0]} to {(forecast['Week'] + pd.Timedelta(days=6)).dt.strftime('%Y-%m-%d').values[0]}")
            print(f"Predicted Net Amount: {forecast['Predicted NetAmount'].values[0]:,.2f}")
    
    return forecasts

if __name__ == "__main__":
    sales_codes = ['265', '430', '544', '525', '254']
    generate_weekly_forecasts(sales_codes)