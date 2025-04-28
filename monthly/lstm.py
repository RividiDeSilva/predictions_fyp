import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.preprocessing import MinMaxScaler
import os
import sys
from statsmodels.tsa.stattools import adfuller
import sqlite3

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping
    except ImportError:
        print("ERROR: Could not import TensorFlow/Keras")
        sys.exit(1)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("WARNING: SHAP not available")
    SHAP_AVAILABLE = False

DB_PATH = 'monthly_forecast.db'

def create_forecast_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monthly_forecast (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            SalesPersonCode TEXT,
            Month TEXT,
            Predicted_NetAmount REAL,
            SHAP_Report TEXT
        );
    """)
    conn.commit()
    conn.close()

def store_prediction(sales_person_code, month, predicted_sales, shap_report):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO monthly_forecast (SalesPersonCode, Month, Predicted_NetAmount, SHAP_Report)
        VALUES (?, ?, ?, ?)
    """, (sales_person_code, month, predicted_sales, shap_report))
    conn.commit()
    conn.close()

def predict_sales(sales_person_codes):

    create_forecast_table()
    if isinstance(sales_person_codes, str):
        sales_person_codes = [sales_person_codes]

    for sales_person_code in sales_person_codes:
        print("\n" + "="*50)
        print(f"Analyzing sales data for SalesPersonCode: {sales_person_code}")
        print("="*50)
        
        try:
            data = pd.read_csv('2023-2024-Horana-cleaned.csv')
            data['DATE'] = pd.to_datetime(data['DATE'])
            data['NetAmount'] = data['NetAmount'].abs()
        except FileNotFoundError:
            print("ERROR: Could not find sales data file '2023-2024-Horana-cleaned.csv'")
            continue
        except Exception as e:
            print(f"ERROR loading sales data: {str(e)}")
            continue

        try:
            kpi_data = pd.read_csv('KPI.csv')
            kpi_data['Year-Month'] = pd.to_datetime(kpi_data['Year-Month'], format='%Y-%m')
            kpi_data['Person Code'] = kpi_data['Person Code'].astype(str)
        except FileNotFoundError:
            print("ERROR: Could not find KPI data file 'KPI.csv'")
            continue
        except Exception as e:
            print(f"ERROR loading KPI data: {str(e)}")
            continue

        sales_filtered = data[data['SalesPersonCode'] == sales_person_code]
        
        if len(sales_filtered) == 0:
            print(f"No data found for SalesPersonCode {sales_person_code}")
            continue
        
        sales_monthly = sales_filtered.resample('ME', on='DATE')['NetAmount'].sum().reset_index()
        sales_monthly['Year-Month'] = sales_monthly['DATE'].dt.to_period('M').dt.to_timestamp()

        merged_data = sales_monthly.merge(kpi_data[kpi_data['Person Code'] == sales_person_code], 
                                         on=['Year-Month'], how='left')
        
        if len(merged_data) < 15:  
            print(f"Not enough data points for SalesPersonCode {sales_person_code} after merging with KPI data")
            print(f"Found only {len(merged_data)} months of data. Need at least 15 months.")
            continue

        features = ['NetAmount', 'Punctuality', 'Absenteeism', 'Maintaining Env', 'Sales Perf', 'Cust. Sat']
        
        if merged_data[features].isna().any().any():
            print("Warning: Missing values found in the data. Filling with forward fill then backward fill.")
            merged_data[features] = merged_data[features].fillna(method='ffill').fillna(method='bfill')
        
        merged_data[features] = merged_data[features].apply(pd.to_numeric)

        adf_result = adfuller(merged_data['NetAmount'].values)
        p_value = adf_result[1]
        
        print(f"ADF Test for SalesPersonCode {sales_person_code}:")
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Critical Values: {adf_result[4]}")
        
        data_for_model = merged_data.copy()
        differenced = False
        
        if p_value > 0.05:
            print("Data is not stationary. Applying first differencing...")
            
            data_diff = data_for_model.copy()
            
            data_diff['NetAmount'] = data_diff['NetAmount'].diff().fillna(0)
            
            data_for_model = data_diff
            differenced = True
            
            adf_result_diff = adfuller(data_for_model['NetAmount'].dropna().values)
            print(f"ADF Test after differencing:")
            print(f"ADF Statistic: {adf_result_diff[0]:.4f}")
            print(f"p-value: {adf_result_diff[1]:.4f}")
        else:
            print("Data is already stationary. No differencing needed.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_for_model[features])

        time_step = 12
        X, y = create_dataset(scaled_data, time_step)
        
        if len(X) == 0:
            print(f"Not enough data points to create sequences with time_step={time_step}")
            continue
            
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

        train_size = int(len(X) * 0.8)
        if train_size == 0:
            train_size = 1  
            
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print("Optimizing ...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, time_step, X.shape[2]), n_trials=20)

        best_params = study.best_params
        print("Best hyperparameters:", best_params)

        final_model = create_model(best_params, time_step, X.shape[2])
        
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        final_model.fit(X_train, y_train, epochs=50, batch_size=best_params['batch_size'], 
                       verbose=1, callbacks=[early_stopping])

        last_12_months = scaled_data[-time_step:].reshape(1, time_step, scaled_data.shape[1])
        predicted_scaled = final_model.predict(last_12_months)
        
        predicted_sales_scaled = predicted_scaled[0][0]
        
        if differenced:
            last_actual_value = merged_data['NetAmount'].iloc[-1]
            
            predicted_sales = last_actual_value + scaler.inverse_transform([[predicted_sales_scaled] + [0] * (scaled_data.shape[1] - 1)])[0][0]
        else:
            predicted_sales = scaler.inverse_transform([[predicted_sales_scaled] + [0] * (scaled_data.shape[1] - 1)])[0][0]

        next_month = merged_data['Year-Month'].iloc[-1] + pd.DateOffset(months=1)
        print(f"Predicted sales for {next_month.strftime('%b %Y')}: {predicted_sales:.2f}")

        shap_report = None
        if SHAP_AVAILABLE:
            try:
                print("\nPerforming SHAP analysis...")
                background_samples = X_train[np.random.choice(X_train.shape[0], min(50, X_train.shape[0]), replace=False)]
                explainer = shap.GradientExplainer(final_model, background_samples)
                
                shap_samples = min(3, len(X_test))
                if shap_samples == 0:
                    X_test_shap = X_train[-1:] 
                    shap_samples = 1
                else:
                    X_test_shap = X_test[:shap_samples]
                    
                shap_values = explainer.shap_values(X_test_shap)
                shap_values_reshaped = np.array(shap_values).reshape(shap_samples, time_step, X.shape[2])
                
                influential_reasons = {}
                for i in range(shap_values_reshaped.shape[0]):
                    for j in range(1, X.shape[2]):  
                        month_index = np.argmax(np.abs(shap_values_reshaped[i, :, j]))
                        data_index = train_size + i + month_index if i < len(X_test) else month_index
                        if data_index < len(merged_data):
                            month = merged_data['Year-Month'].iloc[data_index]
                            key = (month, features[j])
                            if key not in influential_reasons or abs(shap_values_reshaped[i, month_index, j]) > abs(influential_reasons[key]):
                                influential_reasons[key] = shap_values_reshaped[i, month_index, j]
                
                top_overall = sorted(influential_reasons.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                print("\nHighest 3 Influential KPI Reasons Overall ")
                
                shap_report = "Top Influential KPI Reasons:\n"
                for rank, ((month, kpi), value) in enumerate(top_overall, 1):
                    direction = "increased" if value > 0 else "decreased"
                    report_line = f"Top {rank}: ({month.strftime('%Y-%m')}) {kpi} {direction} prediction by {abs(value):.4f}"
                    shap_report += report_line + "\n"
                    print(report_line)
            except Exception as e:
                print(f"Error with SHAP analysis: {e}")
                shap_report = f"SHAP analysis error: {str(e)}"
        else:
            print("\nSkipping SHAP analysis as the library is not available.")
            shap_report = "SHAP analysis not available"

        try:
            store_prediction(sales_person_code, 
                             next_month.strftime('%Y-%m'), 
                             predicted_sales, 
                             shap_report)
            print("\nPrediction and SHAP report stored successfully in database.")
        except Exception as e:
            print(f"Error storing prediction in database: {e}")


def create_dataset(data, time_step=12):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step, 0])  
    return np.array(X), np.array(y)


def create_model(params, time_step, features):
    model = Sequential()
    model.add(LSTM(params['n_units'], 
                  return_sequences=(params['n_layers'] == 2), 
                  input_shape=(time_step, features)))
    model.add(Dropout(params['dropout_rate']))
    
    if params['n_layers'] == 2:
        model.add(LSTM(params['n_units']))
        model.add(Dropout(params['dropout_rate']))
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), 
                 loss='mean_squared_error')
    return model


def objective(trial, X_train, y_train, X_test, y_test, time_step, n_features):
    n_layers = trial.suggest_int('n_layers', 1, 2)
    n_units = trial.suggest_int('n_units', 20, 100, step=10)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    model = Sequential()
    model.add(LSTM(n_units, return_sequences=(n_layers == 2), input_shape=(time_step, n_features)))
    model.add(Dropout(dropout_rate))
    
    if n_layers == 2:
        model.add(LSTM(n_units))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=0, callbacks=[early_stopping])
    
    if len(X_test) == 0:
        loss = model.evaluate(X_train, y_train, verbose=0)
    else:
        loss = model.evaluate(X_test, y_test, verbose=0)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict sales for multiple salespersons')
    
    parser.add_argument('--sales_person_codes', nargs='+', type=str, 
                        default=['430'],
                        help='List of salesperson codes to analyze (default: 265 65 88)')
    
    args = parser.parse_args()
    predict_sales(args.sales_person_codes)