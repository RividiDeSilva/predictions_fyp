import pandas as pd
import numpy as np
import sqlite3
import schedule
import time
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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
import schedule
import time
import psycopg2
from dotenv import dotenv_values
import json


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

DB_FILE = "daily_forecast.db"
WEEKLY_DB_FILE = "weekly_forecast.db"
MONTHLY_DB_FILE = "monthly_forecast.db" 

class SalesForecast:
    #initialise the commmon variables/functions
    def __init__(self, db_file):
        self.db_file = db_file
        self.daily_sales = self.load_data()
        self.setup_database()

    def load_data(self):
        data = pd.read_csv('final_modified_sales_data.csv')
        data['DATE'] = pd.to_datetime(data['DATE'])
        data['NetAmount'] = data['NetAmount'].abs()
        return data.groupby(['DATE', 'SalesPersonCode'])['NetAmount'].sum().reset_index()

    def setup_database(self):
        conn = sqlite3.connect(self.db_file)
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

    def save_forecast_to_db(self, sales_code, forecast_df):
        conn = sqlite3.connect(self.db_file)
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

    def predict_sales(self, sales_code):
        self.daily_sales['SalesPersonCode'] = self.daily_sales['SalesPersonCode'].astype(str)

        person_data = self.daily_sales[self.daily_sales['SalesPersonCode'] == sales_code]
        
        if person_data.empty:
            print(f"No data found for SalesPerson {sales_code}.")
            return None

        person_data = person_data.set_index('DATE').asfreq('D').fillna(0)
        
        if len(person_data) < 5:
            print(f"Not enough data for SalesPerson {sales_code}.")
            return None

        try:
            print('>>>>>',person_data['NetAmount'])
            model = ARIMA(person_data['NetAmount'], order=(7, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=7)

            last_date = person_data.index[-1]
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

            results_df = pd.DataFrame({
                'DATE': forecast_dates,
                'Predicted NetAmount': forecast.values
            })

            self.save_forecast_to_db(sales_code, results_df)
            return results_df

        except Exception as e:
            print(f"ARIMA failed for {sales_code} due to {e}. Switching to ETS...")
            return self.predict_ets(sales_code)

    def predict_ets(self, sales_code):
        person_data = self.daily_sales[self.daily_sales['SalesPersonCode'] == sales_code]

        if len(person_data) < 3:
            print(f"Not enough data for {sales_code}, using mean-based prediction.")
            mean_forecast = [person_data['NetAmount'].mean()] * 7
        else:
            model = ExponentialSmoothing(person_data['NetAmount'], trend="add", seasonal=None)
            model_fit = model.fit()
            mean_forecast = model_fit.forecast(steps=7)

        last_date = person_data['DATE'].max()
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

        results_df = pd.DataFrame({
            'DATE': forecast_dates,
            'Predicted NetAmount': mean_forecast
        })

        self.save_forecast_to_db(sales_code, results_df)
        return results_df

    def run_forecast(self):
        sales_codes = ['265', '430', '544', '525', '254']
        all_forecasts = {}
        
        for code in sales_codes:
            print(code)
            forecast_df = self.predict_sales(code)
            print(f"Forecast for {code}:")
            print(forecast_df)
            if forecast_df is not None:
                all_forecasts[code] = forecast_df
        
        self.detect_consecutive_decreases(all_forecasts)

     #detect decrease of sales for 3 consecutive days   

    def detect_consecutive_decreases(self, forecasts=None, min_consecutive_decreases=3):

        decreasing_sales = {}
        
        if forecasts is None:
            forecasts = {}
            sales_codes = ['265', '430', '544', '525', '254']
            for code in sales_codes:
                forecast_df = self.predict_sales(code)
                if forecast_df is not None:
                    forecasts[code] = forecast_df
        
        for code, forecast_df in forecasts.items():
            if forecast_df is None or len(forecast_df) < min_consecutive_decreases:
                continue
                
            decrease_periods = []
            current_decrease = []
            
            for i in range(1, len(forecast_df)):
                current_amount = forecast_df.iloc[i]['Predicted NetAmount']
                previous_amount = forecast_df.iloc[i-1]['Predicted NetAmount']
                
                if current_amount < previous_amount:
                    if not current_decrease:
                        current_decrease.append({
                            'date': forecast_df.iloc[i-1]['DATE'].strftime('%Y-%m-%d') if isinstance(forecast_df.iloc[i-1]['DATE'], pd.Timestamp) else str(forecast_df.iloc[i-1]['DATE']),
                            'amount': float(previous_amount)
                        })
                    
                    current_decrease.append({
                        'date': forecast_df.iloc[i]['DATE'].strftime('%Y-%m-%d') if isinstance(forecast_df.iloc[i]['DATE'], pd.Timestamp) else str(forecast_df.iloc[i]['DATE']),
                        'amount': float(current_amount)
                    })
                else:
                    if len(current_decrease) >= min_consecutive_decreases:
                        decrease_periods.append(current_decrease)
                    current_decrease = []
            
            if len(current_decrease) >= min_consecutive_decreases:
                decrease_periods.append(current_decrease)
                
            if decrease_periods:
                decreasing_sales[code] = decrease_periods
        
        if decreasing_sales:
            with open('decreasing_sales_alerts.json', 'w') as f:
                json.dump(decreasing_sales, f, indent=4)
            
            print(f"Saved {len(decreasing_sales)} sales people with decreasing trends to decreasing_sales_alerts.json")
        else:
            print("No consecutive decreases detected.")
            
        return decreasing_sales



class WeeklySalesForecast:
    def __init__(self, db_file):
        self.db_file = db_file
        self.data = self.load_data()
        self.setup_database()

    def load_data(self):
        try:
            data = pd.read_csv('final_modified_sales_data.csv')
            
            data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce')
            
            if data['DATE'].isna().any():
                print(f"Warning: {data['DATE'].isna().sum()} date entries could not be parsed")
            
            data['NetAmount'] = data['NetAmount'].abs()
            
            data['SalesPersonCode'] = data['SalesPersonCode'].astype(str)
            
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()  

    def setup_database(self):
        conn = sqlite3.connect(self.db_file)
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

    def save_forecast_to_db(self, sales_code, forecast_df):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        for _, row in forecast_df.iterrows():
            try:
                if isinstance(row['Week'], pd.Timestamp):
                    week_str = row['Week'].strftime('%Y-%m-%d')
                else:
                    week_str = str(row['Week'])
                
                cursor.execute('''INSERT OR IGNORE INTO weekly_forecast (SalesPersonCode, Week, Predicted_NetAmount) 
                                  VALUES (?, ?, ?)''', (sales_code, week_str, row['Predicted NetAmount']))
            
            except Exception as e:
                print(f"Failed to insert row for {sales_code}: {e}")

        conn.commit()
        conn.close()

    def predict_weekly_sales(self, sales_code, last_date=None):
        sales_code = str(sales_code)
        person_data = self.data[self.data['SalesPersonCode'] == sales_code]

        print(f"Records found for {sales_code}: {len(person_data)}")
        
        if person_data.empty:
            print(f"No data found for SalesPerson {sales_code}.")
            return None

        if last_date is None:
            last_date = person_data['DATE'].max()
        
        print(f"DEBUG - last_date for {sales_code}: {last_date}")
        
        if pd.isna(last_date):
            print(f"Invalid last date for SalesPerson {sales_code}. Using current date as fallback.")
            last_date = pd.Timestamp.now().normalize()
            print(f"Using fallback date: {last_date}")
            
        forecast_week_start = last_date + timedelta(days=1)
        forecast_week_end = forecast_week_start + timedelta(days=6)


        try:
            #sum of weekly netamount
            weekly_sales = person_data.groupby(pd.Grouper(key='DATE', freq='W'))['NetAmount'].sum().reset_index()
            print(f"Weekly data points available: {len(weekly_sales)}")
            
            if len(weekly_sales) < 5:
                print(f"Not enough weekly data for SalesPerson {sales_code}. Need at least 5 weeks, found {len(weekly_sales)}.")
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

                print(f"Forecast results for {sales_code}: {results_df}")
                self.save_forecast_to_db(sales_code, results_df)
                return results_df

            except Exception as e:
                print(f"ARIMA failed for {sales_code} due to {e}. Switching to mean-based prediction...")
                
                mean_forecast = [weekly_sales['NetAmount'].mean()]
                print(f"Using mean-based forecast: {mean_forecast[0]}")

                results_df = pd.DataFrame({
                    'Week': [forecast_week_start],
                    'Predicted NetAmount': mean_forecast
                })

                self.save_forecast_to_db(sales_code, results_df)
                return results_df
        
        except Exception as e:
            print(f"Error creating weekly sales aggregation for {sales_code}: {e}")
            return None

    def generate_weekly_forecasts(self, sales_codes=['265', '430', '544', '525', '254']):
        forecasts = {}
        
        for sales_code in sales_codes:
            print(f"\n--- Processing Sales Person {sales_code} ---")
            filtered_data = self.data[self.data['SalesPersonCode'] == str(sales_code)]
            
            if filtered_data.empty:
                print(f"No data available for Sales Person {sales_code}")
                continue
                
            last_date = filtered_data['DATE'].max()
            
            forecast = self.predict_weekly_sales(sales_code, last_date)
            
            if forecast is not None:
                forecasts[sales_code] = forecast
                
                try:
                    start_date = forecast['Week'].dt.strftime('%Y-%m-%d').values[0]
                    end_date = (forecast['Week'] + timedelta(days=6)).dt.strftime('%Y-%m-%d').values[0]
                    amount = forecast['Predicted NetAmount'].values[0]
                    
                    print(f"\nWeekly Forecast for Sales Person {sales_code}:")
                    print(f"Prediction Week: {start_date} to {end_date}")
                    print(f"Predicted Net Amount: {amount:,.2f}")
                except Exception as e:
                    print(f"Error formatting forecast output for {sales_code}: {e}")
                    print(f"Raw forecast data: {forecast}")
            else:
                print(f"No forecast could be generated for Sales Person {sales_code}")
        
        return forecasts
    

class MonthlySalesForecast:
    def __init__(self, db_path='monthly_forecast.db'):
        self.db_path = db_path
        self.create_forecast_table()
        self.shap_available = False
        try:
            import shap
            self.shap_available = True
        except ImportError:
            print("SHAP library not available. SHAP analysis will be skipped.")

    def create_forecast_table(self):
        conn = sqlite3.connect(self.db_path)
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

    def store_prediction(self, sales_person_code, month, predicted_sales, shap_report):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        predicted_sales_float = float(predicted_sales)
        
        cursor.execute("""
            INSERT INTO monthly_forecast (SalesPersonCode, Month, Predicted_NetAmount, SHAP_Report)
            VALUES (?, ?, ?, ?)
        """, (sales_person_code, month, predicted_sales_float, shap_report))
        
        conn.commit()
        conn.close()
        
    def add_lagged_features(self, df, lag_periods=[1, 3, 6, 12]):
        """Add lagged features to capture different time horizons of influence"""
        for lag in lag_periods:
            if len(df) > lag:
                df[f'NetAmount_lag_{lag}'] = df['NetAmount'].shift(lag)
        
        if len(df) > 3:
            df['NetAmount_rolling_mean_3'] = df['NetAmount'].rolling(window=3).mean()
            df['NetAmount_rolling_std_3'] = df['NetAmount'].rolling(window=3).std()
        
        if len(df) > 6:
            df['NetAmount_rolling_mean_6'] = df['NetAmount'].rolling(window=6).mean()
        
        #checking of common patterns
        df['month_sin'] = np.sin(2 * np.pi * df['DATE'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['DATE'].dt.month / 12)
        
        df['quarter'] = df['DATE'].dt.quarter
        
        df['is_december'] = (df['DATE'].dt.month == 12).astype(int)

        #addition of null values
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        if lag_cols:
            for col in lag_cols:
                if df[col].isna().any():
                    global_mean = df[col].mean()
                    df[col].fillna(global_mean, inplace=True)
        
        return df

    def objective(self, trial, X_train, y_train, X_test, y_test, time_step, n_features):
        """Optuna objective function for hyperparameter optimization"""
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

        try:
            predictions = model.predict(X_test, batch_size=batch_size)
            loss = np.mean(np.square(y_test - predictions))
        except ValueError:
            loss = np.mean(np.square(y_train - model.predict(X_train, batch_size=batch_size)))
        
        return loss

    def predict_sales(self, sales_person_codes=['265', '430', '544', '525', '254']):
        """Predict monthly sales for specified salesperson codes with top 3 KPI impacts"""
        if isinstance(sales_person_codes, str):
            sales_person_codes = [sales_person_codes]

        for sales_person_code in sales_person_codes:
            print("\n" + "="*50)
            print(f"Analyzing sales data for SalesPersonCode: {sales_person_code}")
            print("="*50)
            
            try:
                data = pd.read_csv('final_modified_sales_data.csv')
                data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce')
                
                invalid_dates = data['DATE'].isna().sum()
                if invalid_dates > 0:
                    print(f"Warning: Removed {invalid_dates} rows with invalid dates")
                    data = data.dropna(subset=['DATE'])
                
                data['NetAmount'] = data['NetAmount'].abs()
                data['SalesPersonCode'] = data['SalesPersonCode'].astype(str)
                
                print(f"Total data rows: {len(data)}")
            except FileNotFoundError:
                print("ERROR: Could not find sales data file 'final_modified_sales_data.csv'")
                continue
            except Exception as e:
                print(f"ERROR loading sales data: {str(e)}")
                continue

            try:
                kpi_data = pd.read_csv('KPI.csv')
                kpi_data['Year-Month'] = pd.to_datetime(kpi_data['Year-Month'], format='%Y-%m', errors='coerce')
                
                invalid_dates = kpi_data['Year-Month'].isna().sum()
                if invalid_dates > 0:
                    print(f"Warning: Removed {invalid_dates} rows with invalid Year-Month in KPI data")
                    kpi_data = kpi_data.dropna(subset=['Year-Month'])
                    
                kpi_data['Person Code'] = kpi_data['Person Code'].astype(str)
                
                print(f"Total KPI data rows: {len(kpi_data)}")
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
            
            print(f"Found {len(sales_filtered)} records for SalesPersonCode {sales_person_code}")
            
            sales_monthly = sales_filtered.resample('ME', on='DATE')['NetAmount'].sum().reset_index()
            print(f"Monthly data points: {len(sales_monthly)}")
            
            plt.figure(figsize=(12, 6))
            plt.plot(sales_monthly['DATE'], sales_monthly['NetAmount'], marker='o')
            plt.title(f'Historical Sales for Person Code {sales_person_code}')
            plt.xlabel('Date')
            plt.ylabel('Net Amount')
            plt.grid(True)
            plt.tight_layout()
            # try:
            #     plt.savefig(f'sales_history_{sales_person_code}.png')
            # except Exception as e:
            #     print(f"Warning: Could not save plot: {e}")
            # plt.close()
            
            sales_monthly['month'] = sales_monthly['DATE'].dt.month
            monthly_avg = sales_monthly.groupby('month')['NetAmount'].mean()
            plt.figure(figsize=(10, 5))
            monthly_avg.plot(kind='bar')
            plt.title(f'Average Sales by Month - Person {sales_person_code}')
            plt.xlabel('Month')
            plt.ylabel('Average Net Amount')
            plt.grid(True, axis='y')
            plt.tight_layout()
            # try:
            #     plt.savefig(f'monthly_avg_{sales_person_code}.png')
            # except Exception as e:
            #     print(f"Warning: Could not save plot: {e}")
            # plt.close()
            
            sales_monthly['Year-Month'] = sales_monthly['DATE'].dt.to_period('M').dt.to_timestamp()

            #merging the monthly KPIs and monthly NetAmount
            merged_data = sales_monthly.merge(kpi_data[kpi_data['Person Code'] == sales_person_code], 
                                        on=['Year-Month'], how='left')
            
            print(f"Merged data points: {len(merged_data)}")
            
            if len(merged_data) < 5:  
                print(f"Not enough data points for SalesPersonCode {sales_person_code} after merging with KPI data")
                print(f"Found only {len(merged_data)} months of data. Need at least 5 months.")
                continue

            merged_data = self.add_lagged_features(merged_data)

            #defining the KPIs
            numeric_cols = ['NetAmount', 'Punctuality', 'Absenteeism', 'Maintaining Env', 'Sales Perf', 'Cust. Sat']
            numeric_cols.extend([col for col in merged_data.columns if 'lag' in col or 'rolling' in col or 'month_' in col])
            
            corr = merged_data[numeric_cols].corr()['NetAmount'].sort_values(ascending=False)
            print("\nFeature correlation with NetAmount:")
            print(corr)
            
            print("\nTop 5 correlated features with NetAmount:")
            print(corr.iloc[1:6])  #
            
            if merged_data[numeric_cols].isna().any().any():
                print("Warning: Missing values found in the data. Filling with forward fill then backward fill.")
                merged_data[numeric_cols] = merged_data[numeric_cols].fillna(method='ffill').fillna(method='bfill')
            
            for col in numeric_cols:
                merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')
                
            if merged_data[numeric_cols].isna().any().any():
                merged_data = merged_data.dropna(subset=numeric_cols)
                print(f"Dropped rows with NaN values after numeric conversion. Remaining rows: {len(merged_data)}")
            
            if len(merged_data) < 5:
                print(f"Not enough data points after handling missing values. Only {len(merged_data)} months remaining.")
                continue

            X_features = merged_data.drop(['DATE', 'Year-Month', 'month', 'SalesPersonCode', 'Person Code', 'Total (out of 50)'], axis=1, errors='ignore')
            feature_names = X_features.columns.tolist()

            #Preprocessing techinques check this
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            
            y_values = merged_data['NetAmount'].values.reshape(-1, 1)
            scaled_y = scaler_y.fit_transform(y_values)
            
            scaled_X = scaler_X.fit_transform(X_features)
            
            X_df = pd.DataFrame(scaled_X, columns=X_features.columns)
            
            time_step = min(4, len(X_df) - 2)  
            print(f"Using time_step of {time_step} months for predictions")
            
            X, y = self.create_dataset(X_df.values, scaled_y, time_step)
            
            if len(X) == 0:
                print(f"Not enough data points to create sequences with time_step={time_step}")
                continue
                
            train_size = int(len(X) * 0.8)
            if train_size == 0:
                train_size = 1  
                
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]


            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_test, y_test, time_step, X.shape[2]), n_trials=20)

            best_params = study.best_params
            print("Best hyperparameters:", best_params)
            models = []
            model_names = []
            
            print("\nTraining basic LSTM model...")
            
            params = {
                'n_units': 64,
                'n_layers': 2,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 16
            }
            
            #Model that used the hard coded parameters
            basic_model = self.create_model(params, time_step, X.shape[2])
            
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            try:
                history = basic_model.fit(
                    X_train, y_train, 
                    epochs=100, 
                    batch_size=best_params['batch_size'],
                    verbose=1, 
                    callbacks=[early_stopping], 
                    validation_split=0.2
                )
                
                models.append(basic_model)
                model_names.append("Basic LSTM")
                
                plt.figure(figsize=(10, 5))
                plt.plot(history.history['loss'], label='Training Loss')
                if 'val_loss' in history.history:
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title(f'Model Training History - Person {sales_person_code}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                # try:
                #     plt.savefig(f'training_history_{sales_person_code}.png')
                # except Exception as e:
                #     print(f"Warning: Could not save plot: {e}")
                # plt.close()
                
            except Exception as e:
                print(f"Error training basic model: {e}")
            
            if len(X_train) >= 8:
                print("\nTraining simpler LSTM model with fewer units...")
                simple_params = {
                    'n_units': 32,
                    'n_layers': 1,
                    'dropout_rate': 0.1,
                    'learning_rate': 0.01,
                    'batch_size': 8
                }
                
                simple_model = self.create_model(simple_params, time_step, X.shape[2])
                
                try:
                    simple_history = simple_model.fit(
                        X_train, y_train, 
                        epochs=50, 
                        batch_size=simple_params['batch_size'],
                        verbose=1, 
                        callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
                    )
                    
                    
                    models.append(simple_model)
                    model_names.append("Simple LSTM")
                except Exception as e:
                    print(f"Error training simple model: {e}")
            
            best_model = None
            best_loss = float('inf')
            best_name = None
            
            for i, model in enumerate(models):
                if len(X_test) > 0:  
                    try:
                        loss = model.evaluate(X_test, y_test, verbose=0)
                        print(f"{model_names[i]} Test Loss: {loss:.4f}")
                        
                        if loss < best_loss:
                            best_loss = loss
                            best_model = model
                            best_name = model_names[i]
                    except Exception as e:
                        print(f"Error evaluating {model_names[i]}: {e}")
                    try:
                        loss = model.evaluate(X_train, y_train, verbose=0)
                        print(f"{model_names[i]} Training Loss: {loss:.4f}")
                        
                        if loss < best_loss:
                            best_loss = loss
                            best_model = model
                            best_name = model_names[i]
                    except Exception as e:
                        print(f"Error evaluating {model_names[i]} on training data: {e}")
            
            if best_model is None and len(models) > 0:
                best_model = models[0]
                best_name = model_names[0]
                print(f"Using {best_name} as fallback since evaluation failed")
            
            if best_model is None:
                print("ERROR: No model trained successfully. Skipping prediction.")
                continue
                
            print(f"\nUsing {best_name} for predictions")
            
            if len(X_test) > 0:
                try:
                    test_pred = best_model.predict(X_test)
                    test_pred_inv = scaler_y.inverse_transform(test_pred)
                    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
                    
                    test_rmse = np.sqrt(np.mean(np.square(test_pred_inv - y_test_inv)))
                    test_mape = np.mean(np.abs((y_test_inv - test_pred_inv) / y_test_inv)) * 100
                    
                    print(f"Test RMSE: {test_rmse:.2f}")
                    print(f"Test MAPE: {test_mape:.2f}%")
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(y_test_inv, label='Actual')
                    plt.plot(test_pred_inv, label='Predicted')
                    plt.title(f'Test Predictions vs Actuals - Person {sales_person_code}')
                    plt.xlabel('Test Sample Index')
                    plt.ylabel('Net Amount')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    # try:
                    #     plt.savefig(f'test_predictions_{sales_person_code}.png')
                    # except Exception as e:
                    #     print(f"Warning: Could not save plot: {e}")
                    # plt.close()
                except Exception as e:
                    print(f"Error evaluating on test data: {e}")
            
            try:
                input_sequence = X_df.values[-time_step:].reshape(1, time_step, X_df.shape[1])
                predicted_scaled = best_model.predict(input_sequence)
                
                predicted_sales = scaler_y.inverse_transform(predicted_scaled)[0][0]
                
                avg_3m = merged_data['NetAmount'].iloc[-3:].mean()
                avg_all = merged_data['NetAmount'].mean()
                std_all = merged_data['NetAmount'].std()
                
                min_allowed = max(avg_3m * 0.5, avg_all - 2 * std_all)
                max_allowed = min(avg_3m * 1.5, avg_all + 2 * std_all)
                
                original_prediction = predicted_sales
                
                if predicted_sales < min_allowed:
                    print(f"Warning: Prediction ({predicted_sales:.2f}) below reasonable minimum. Adjusting to {min_allowed:.2f}")
                    predicted_sales = min_allowed
                elif predicted_sales > max_allowed:
                    print(f"Warning: Prediction ({predicted_sales:.2f}) above reasonable maximum. Adjusting to {max_allowed:.2f}")
                    predicted_sales = max_allowed
                
                predicted_sales = abs(predicted_sales)
                
                next_month = merged_data['Year-Month'].iloc[-1] + pd.DateOffset(months=1)
                print(f"\nPredicted sales for {next_month.strftime('%b %Y')}: {predicted_sales:.2f}")
                if original_prediction != predicted_sales:
                    print(f"Original prediction was: {original_prediction:.2f}")
                
                last_1m = merged_data['NetAmount'].iloc[-1]
                last_3m_avg = merged_data['NetAmount'].iloc[-3:].mean()
                last_year_same_month = None
                
                for i in range(len(merged_data)):
                    if merged_data['DATE'].iloc[i].month == next_month.month and \
                    merged_data['DATE'].iloc[i].year == next_month.year - 1:
                        last_year_same_month = merged_data['NetAmount'].iloc[i]
                        break
                
                print(f"Last month sales: {last_1m:.2f}")
                print(f"Last 3 months average: {last_3m_avg:.2f}")
                if last_year_same_month is not None:
                    print(f"Same month last year: {last_year_same_month:.2f}")
                
                pct_change_1m = (predicted_sales - last_1m) / last_1m * 100
                pct_change_3m = (predicted_sales - last_3m_avg) / last_3m_avg * 100
                
                print(f"Percent change from last month: {pct_change_1m:.2f}%")
                print(f"Percent change from 3-month average: {pct_change_3m:.2f}%")
            except Exception as e:
                print(f"Error in prediction: {e}")
                continue
                    
            try:
                kpi_columns = ['Punctuality', 'Absenteeism', 'Maintaining Env', 'Sales Perf', 'Cust. Sat']
                
                kpi_importance = {}
                for feature in kpi_columns:
                    if feature in merged_data.columns:
                        corr_val = merged_data[['NetAmount', feature]].corr().iloc[0, 1]
                        if not np.isnan(corr_val):
                            # Store absolute impact as percentage
                            impact_percentage = abs(corr_val) * 100
                            kpi_importance[feature] = impact_percentage
                
                sorted_kpi_importance = sorted(kpi_importance.items(), key=lambda x: x[1], reverse=True)
                
                top_3_kpi = sorted_kpi_importance[:3]
                
                shap_report = "Top 3 KPI factors impacting sales:\n"
                for rank, (feature, impact) in enumerate(top_3_kpi, 1):
                    direction = "positive" if merged_data[['NetAmount', feature]].corr().iloc[0, 1] > 0 else "negative"
                    report_line = f"{rank}. {feature}: {direction} impact of {impact:.2f}%"
                    shap_report += report_line + "\n"
                    print(report_line)
                    
                if self.shap_available:
                    print("\nAttempting SHAP analysis for KPI metrics...")
                    try:
                        import shap
                        
                        last_sample = input_sequence.reshape(1, time_step * X_df.shape[1])
                        
                        background = X_train[np.random.choice(X_train.shape[0], min(10, X_train.shape[0]), replace=False)]
                        explainer = shap.KernelExplainer(
                            lambda x: best_model.predict(x.reshape(x.shape[0], time_step, X_df.shape[1])).flatten(), 
                            background.reshape(background.shape[0], time_step * X_df.shape[1])
                        )
                        
                        shap_values = explainer.shap_values(last_sample)
                        
                        feature_names_expanded = []
                        for t in range(time_step):
                            for f in feature_names:
                                feature_names_expanded.append(f"{f}_t-{time_step-t}")
                        
                        kpi_shap_importance = {}
                        total_shap_sum = sum(abs(shap_values[0]))
                        
                        for i, full_name in enumerate(feature_names_expanded):
                            base_feature = full_name.split('_t-')[0]
                            if base_feature in kpi_columns:
                                if base_feature not in kpi_shap_importance:
                                    kpi_shap_importance[base_feature] = 0
                                impact = abs(shap_values[0][i])
                                kpi_shap_importance[base_feature] += impact
                        
                        for feature in kpi_shap_importance:
                            kpi_shap_importance[feature] = (kpi_shap_importance[feature] / total_shap_sum) * 100
                            
                        sorted_kpi_shap = sorted(kpi_shap_importance.items(), key=lambda x: x[1], reverse=True)
                        top_3_kpi_shap = sorted_kpi_shap[:3]
                        
                        if top_3_kpi_shap:  
                            shap_report = "Top 3 KPI factors impacting sales (SHAP analysis):\n"
                            for rank, (feature, impact) in enumerate(top_3_kpi_shap, 1):
                                report_line = f"{rank}. {feature}: {impact:.2f}% impact on prediction"
                                shap_report += report_line + "\n"
                                print(report_line)
                    
                    except Exception as e:
                        print(f"Error in SHAP KPI analysis: {e}")
                    
            except Exception as e:
                print(f"Error in KPI feature importance analysis: {e}")
                shap_report = "KPI impact analysis failed: " + str(e)

            try:
                predicted_sales_float = float(predicted_sales)
                
                self.store_prediction(
                    sales_person_code, 
                    next_month.strftime('%Y-%m'), 
                    predicted_sales_float, 
                    shap_report 
                )
                print("\nPrediction and analysis report stored successfully in database.")
            except Exception as e:
                print(f"Error storing prediction in database: {e}")

    def create_dataset(self, data, target, time_step=3):
        """Create time series dataset with specified time step"""
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:i + time_step])
            y.append(target[i + time_step])
        return np.array(X), np.array(y)

    def create_model(self, params, time_step, n_features):
        """Create LSTM model with given parameters"""
        model = Sequential()
        
        model.add(LSTM(params['n_units'], 
                      return_sequences=(params['n_layers'] > 1), 
                      input_shape=(time_step, n_features)))
        model.add(Dropout(params['dropout_rate']))
        
        for i in range(1, params['n_layers']):
            return_sequences = i < params['n_layers'] - 1
            model.add(LSTM(params['n_units'], return_sequences=return_sequences))
            model.add(Dropout(params['dropout_rate']))
        
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']), 
                     loss='mean_squared_error')
        return model

    def run_if_first_of_month(self):
        if datetime.now().day == 1:
            self.predict_sales()

class SalesDataFetcher:
    def __init__(self, env_file=".env", csv_file="final_modified_sales_data.csv"):
        self.config = dotenv_values(env_file)
        self.csv_file = csv_file
        self.conn = None
        self.cursor = None

    def connect_db(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                host=self.config['DB_HOST'],
                database=self.config['DB_NAME'],
                user=self.config['DB_USER'],
                password=self.config['DB_PASSWORD']
            )
            self.cursor = self.conn.cursor()
            print("Database connection successful!")
        except psycopg2.Error as e:
            print(f"Error connecting to the database: {e}")
            exit()

    def get_last_even_date(self):
        today = datetime.today().date()
        return today - timedelta(days=1) if today.day % 2 == 0 else today - timedelta(days=2)

    def fetch_sales_data(self):
        last_even_date = self.get_last_even_date()
        query = f"""
            SELECT sale_date, salesperson_code, sale_time, net_amount
            FROM sales_data_schema.sale_data
            WHERE sale_date = '{last_even_date.strftime('%Y-%m-%d')}'
            ORDER BY sale_time;
        """
        return pd.read_sql_query(query, self.conn)

    def process_and_save_data(self, df):
        df.columns = ["DATE", "SalesPersonCode", "Time", "NetAmount"]
        df["Time"] = pd.to_datetime(df["Time"], format='%H:%M:%S').dt.strftime('%I:%M %p')

        if os.path.exists(self.csv_file):
            existing_df = pd.read_csv(self.csv_file)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(self.csv_file, index=False)
        print(f"CSV file updated: {self.csv_file}")

    def close_connection(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def run(self):
        """Executes the full pipeline: connect, fetch, process, and save."""
        self.connect_db()
        df = self.fetch_sales_data()
        self.process_and_save_data(df)
        self.close_connection()               

    


fetcher = SalesDataFetcher()
schedule.every().day.at("10:00").do(fetcher.run)

forecast = SalesForecast(DB_FILE)
schedule.every().monday.at("21:01").do(forecast.run_forecast)

weeklyforecast = WeeklySalesForecast(WEEKLY_DB_FILE)
schedule.every().monday.at("15:42").do(weeklyforecast.generate_weekly_forecasts) 

monthlyforecast = MonthlySalesForecast()
schedule.every().day.at("06:00").do(monthlyforecast.run_if_first_of_month)

while True:
    schedule.run_pending()
    time.sleep(60)  

