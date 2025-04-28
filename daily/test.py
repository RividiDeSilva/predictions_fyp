import pandas as pd
import numpy as np
import sqlite3
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

DB_FILE = "daily_forecast.db"

data = pd.read_csv('2023-2024-Horana-cleaned.csv')
data['DATE'] = pd.to_datetime(data['DATE'])
data['NetAmount'] = data['NetAmount'].abs()

daily_sales = data.groupby(['DATE', 'SalesPersonCode'])['NetAmount'].sum().reset_index()

def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_forecast (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            SalesPersonCode TEXT,
            DATE TEXT,
            Predicted_NetAmount REAL
        )
    ''')

    
    conn.commit()
    conn.close()

def check_existing_predictions(sales_code):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM daily_forecast WHERE SalesPersonCode = ?", (sales_code,))
    count = cursor.fetchone()[0]
    
    conn.close()
    return count > 0  

def save_forecast_to_db(sales_code, forecast_df):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for _, row in forecast_df.iterrows():
        try:
            date_str = row['DATE'].strftime('%Y-%m-%d') if isinstance(row['DATE'], pd.Timestamp) else str(row['DATE'])
            
            cursor.execute('''INSERT OR IGNORE INTO daily_forecast (SalesPersonCode, DATE, Predicted_NetAmount) 
                              VALUES (?, ?, ?)''', (sales_code, date_str, row['Predicted NetAmount']))
        
        except Exception as e:
            print(f"Failed to insert row: {e}")

    conn.commit()
    conn.close()

def predict_sales(sales_code):
    # if check_existing_predictions(sales_code):
    #     print(f"SalesPersonCode {sales_code} already has predictions in the database.")
    #     return None

    person_data = daily_sales[daily_sales['SalesPersonCode'] == sales_code]

    if person_data.empty:
        print(f"No data found for SalesPerson {sales_code}.")
        return None

    person_data = person_data.set_index('DATE').asfreq('D').fillna(0)

    if len(person_data) < 5:
        print(f"Not enough data for SalesPerson {sales_code}.")
        return None

    try:
        model = ARIMA(person_data['NetAmount'], order=(7, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)

        last_date = person_data.index[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]

        results_df = pd.DataFrame({
            'DATE': forecast_dates,
            'Predicted NetAmount': forecast.values
        })

        save_forecast_to_db(sales_code, results_df)
        return results_df

    except Exception as e:
        print(f"ARIMA failed for {sales_code} due to {e}. Switching to ETS...")
        return predict_ets(sales_code)

def predict_ets(sales_code):
    person_data = daily_sales[daily_sales['SalesPersonCode'] == sales_code]

    if len(person_data) < 3:
        print(f"Not enough data for {sales_code}, using mean-based prediction.")
        mean_forecast = [person_data['NetAmount'].mean()] * 7
    else:
        model = ExponentialSmoothing(person_data['NetAmount'], trend="add", seasonal=None)
        model_fit = model.fit()
        mean_forecast = model_fit.forecast(steps=7)

    last_date = person_data['DATE'].max()
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]

    results_df = pd.DataFrame({
        'DATE': forecast_dates,
        'Predicted NetAmount': mean_forecast
    })

    save_forecast_to_db(sales_code, results_df)
    return results_df

def explain_predictions(sales_code):
    setup_database()
    for sale_code in sales_code:

        forecast_df = predict_sales(sale_code)
        print(forecast_df)
        

if __name__ == "__main__":

    sales_code = ['265', '430', '544', '525', '254']
    forecast_df = explain_predictions(sales_code)


