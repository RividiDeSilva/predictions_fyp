import pandas as pd
import numpy as np
import sqlite3
import schedule
import time
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import argparse
import matplotlib.pyplot as plt
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys
from statsmodels.tsa.stattools import adfuller
import psycopg2
from dotenv import dotenv_values
import json
import math

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping
        TF_AVAILABLE = True
    except ImportError:
        print("ERROR: Could not import TensorFlow/Keras")
        TF_AVAILABLE = False
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

class ModelComparison:
    """Class to compare ARIMA and LSTM models for time series forecasting"""
    
    def __init__(self, data_path='final_sales_data.csv'):
        self.data = pd.read_csv(data_path)
        self.data['DATE'] = pd.to_datetime(self.data['DATE'])
        self.data['NetAmount'] = self.data['NetAmount'].abs()
        self.results = {}
        
    def prepare_data_for_lstm(self, series, n_steps=7):
        """Convert time series to supervised learning format for LSTM"""
        X, y = [], []
        for i in range(len(series) - n_steps):
            X.append(series[i:i + n_steps])
            y.append(series[i + n_steps])
        return np.array(X), np.array(y)
    
    def create_lstm_model(self, input_shape):
        """Create and compile LSTM model"""
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=Adam(0.001), loss='mse')
        return model
    
    def compare_models_daily(self, sales_code, test_size=30, forecast_days=7):
        """Compare ARIMA and LSTM models on daily sales data"""
        print(f"\n---- Comparing models for daily sales - SalesPerson {sales_code} ----")
        
        person_data = self.data[self.data['SalesPersonCode'] == sales_code]
        
        if person_data.empty or len(person_data) < (test_size + 14):  # Need enough data
            print(f"Not enough data for SalesPerson {sales_code}.")
            return None
        
        daily_sales = person_data.groupby(['DATE'])['NetAmount'].sum().reset_index()
        daily_sales = daily_sales.set_index('DATE').asfreq('D').fillna(0)
        
        train_data = daily_sales[:-test_size]
        test_data = daily_sales[-test_size:]
        
        results = {
            'sales_code': sales_code,
            'data_points': len(daily_sales),
            'forecast_type': 'daily',
            'models': {}
        }
        
        print("Training ARIMA model...")
        try:
            best_aic = float('inf')
            best_order = None
            
            for p in range(0, 8, 2):  #
                for d in range(0, 3):  
                    for q in range(0, 3):  
                        try:
                            model = ARIMA(train_data, order=(p, d, q))
                            model_fit = model.fit()
                            aic = model_fit.aic
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            print(f"Best ARIMA order: {best_order}")
            best_order = (7, 1, 1)  
            model = ARIMA(train_data, order=best_order)
            model_fit = model.fit()
            
            arima_predictions = []
            history = train_data.copy()
            
            for t in range(len(test_data)):
                model = ARIMA(history, order=best_order)
                model_fit = model.fit()
                yhat = model_fit.forecast(steps=1)
                arima_predictions.append(yhat[0])
                history = pd.concat([history, test_data.iloc[t:t+1]])
            
            arima_rmse = math.sqrt(mean_squared_error(test_data, arima_predictions))
            arima_mae = mean_absolute_error(test_data, arima_predictions)
            
            final_model = ARIMA(daily_sales, order=best_order)
            final_model_fit = final_model.fit()
            arima_forecast = final_model_fit.forecast(steps=forecast_days)
            
            results['models']['ARIMA'] = {
                'order': best_order,
                'rmse': arima_rmse,
                'mae': arima_mae,
                'forecast': arima_forecast.tolist()
            }
            
            print(f"ARIMA RMSE: {arima_rmse:.2f}")
            print(f"ARIMA MAE: {arima_mae:.2f}")
            
        except Exception as e:
            print(f"ARIMA modeling failed: {e}")
            results['models']['ARIMA'] = {'error': str(e)}
        
        if TF_AVAILABLE:
            print("Training LSTM model...")
            try:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(daily_sales.values.reshape(-1, 1))
                
                n_steps = 7  
                X, y = self.prepare_data_for_lstm(scaled_data, n_steps)
                
                train_end = len(daily_sales) - test_size - n_steps
                X_train, y_train = X[:train_end], y[:train_end]
                X_test, y_test = X[train_end:], y[train_end:]
                
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                
                model = self.create_lstm_model((n_steps, 1))
                early_stop = EarlyStopping(monitor='val_loss', patience=10)
                history = model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                lstm_predictions = model.predict(X_test)
                
                lstm_predictions = scaler.inverse_transform(lstm_predictions)
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                
                lstm_rmse = math.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
                lstm_mae = mean_absolute_error(y_test_actual, lstm_predictions)
                
                lstm_future = []
                last_sequence = scaled_data[-n_steps:].reshape(1, n_steps, 1)
                
                for _ in range(forecast_days):
                    next_pred = model.predict(last_sequence)[0]
                    lstm_future.append(next_pred)
                    last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)
                
                lstm_future = scaler.inverse_transform(np.array(lstm_future).reshape(-1, 1)).flatten().tolist()
                
                results['models']['LSTM'] = {
                    'rmse': float(lstm_rmse),
                    'mae': float(lstm_mae),
                    'forecast': lstm_future
                }
                
                print(f"LSTM RMSE: {lstm_rmse:.2f}")
                print(f"LSTM MAE: {lstm_mae:.2f}")
                
                if arima_rmse > 0:
                    error_diff_pct = ((arima_rmse - lstm_rmse) / arima_rmse) * 100
                    if error_diff_pct > 0:
                        print(f"LSTM improves forecast accuracy by {abs(error_diff_pct):.2f}% compared to ARIMA")
                    else:
                        print(f"ARIMA improves forecast accuracy by {abs(error_diff_pct):.2f}% compared to LSTM")
                
                self.plot_comparison(daily_sales.index[-test_size:], test_data.values, 
                                   arima_predictions, lstm_predictions.flatten(),
                                   sales_code, 'daily')

            except Exception as e:
                print(f"LSTM modeling failed: {e}")
                results['models']['LSTM'] = {'error': str(e)}
        
        self.results[f"{sales_code}_daily"] = results
        return results
    
    def compare_models_weekly(self, sales_code, test_size=8, forecast_weeks=4):
        """Compare ARIMA and LSTM models on weekly sales data"""
        print(f"\n---- Comparing models for weekly sales - SalesPerson {sales_code} ----")
        
        person_data = self.data[self.data['SalesPersonCode'] == sales_code]
        
        if person_data.empty:
            print(f"No data found for SalesPerson {sales_code}.")
            return None
        
        weekly_sales = person_data.groupby(pd.Grouper(key='DATE', freq='W'))['NetAmount'].sum().reset_index()
        weekly_sales = weekly_sales.set_index('DATE')
        
        if len(weekly_sales) < (test_size + 8):  # Need enough data
            print(f"Not enough weekly data for SalesPerson {sales_code}. Only {len(weekly_sales)} weeks available.")
            return None
        
        train_data = weekly_sales[:-test_size]
        test_data = weekly_sales[-test_size:]
        
        results = {
            'sales_code': sales_code,
            'data_points': len(weekly_sales),
            'forecast_type': 'weekly',
            'models': {}
        }
        
        print("Training ARIMA model for weekly data...")
        try:
            best_aic = float('inf')
            best_order = None
            
            for p in range(0, 5):
                for d in range(0, 2):
                    for q in range(0, 2):
                        try:
                            model = ARIMA(train_data, order=(p, d, q))
                            model_fit = model.fit()
                            aic = model_fit.aic
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            print(f"Best ARIMA order for weekly: {best_order}")
            
            model = ARIMA(train_data, order=best_order)
            model_fit = model.fit()
            
            arima_predictions = []
            history = train_data.copy()
            
            for t in range(len(test_data)):
                model = ARIMA(history, order=best_order)
                model_fit = model.fit()
                yhat = model_fit.forecast(steps=1)
                arima_predictions.append(yhat[0])
                history = pd.concat([history, test_data.iloc[t:t+1]])
            
            arima_rmse = math.sqrt(mean_squared_error(test_data, arima_predictions))
            arima_mae = mean_absolute_error(test_data, arima_predictions)
            
            final_model = ARIMA(weekly_sales, order=best_order)
            final_model_fit = final_model.fit()
            arima_forecast = final_model_fit.forecast(steps=forecast_weeks)
            
            results['models']['ARIMA'] = {
                'order': best_order,
                'rmse': arima_rmse,
                'mae': arima_mae,
                'forecast': arima_forecast.tolist()
            }
            
            print(f"ARIMA Weekly RMSE: {arima_rmse:.2f}")
            print(f"ARIMA Weekly MAE: {arima_mae:.2f}")
            
        except Exception as e:
            print(f"ARIMA weekly modeling failed: {e}")
            results['models']['ARIMA'] = {'error': str(e)}
        
        if TF_AVAILABLE:
            print("Training LSTM model for weekly data...")
            try:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(weekly_sales.values.reshape(-1, 1))
                
                n_steps = 4  
                X, y = self.prepare_data_for_lstm(scaled_data, n_steps)
                
                train_end = len(weekly_sales) - test_size - n_steps
                X_train, y_train = X[:train_end], y[:train_end]
                X_test, y_test = X[train_end:], y[train_end:]
                
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                
                model = self.create_lstm_model((n_steps, 1))
                early_stop = EarlyStopping(monitor='val_loss', patience=10)
                history = model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                lstm_predictions = model.predict(X_test)
                
                lstm_predictions = scaler.inverse_transform(lstm_predictions)
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                
                lstm_rmse = math.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
                lstm_mae = mean_absolute_error(y_test_actual, lstm_predictions)
                
                lstm_future = []
                last_sequence = scaled_data[-n_steps:].reshape(1, n_steps, 1)
                
                for _ in range(forecast_weeks):
                    next_pred = model.predict(last_sequence)[0]
                    lstm_future.append(next_pred)
                    last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)
                
                lstm_future = scaler.inverse_transform(np.array(lstm_future).reshape(-1, 1)).flatten().tolist()
                
                results['models']['LSTM'] = {
                    'rmse': float(lstm_rmse),
                    'mae': float(lstm_mae),
                    'forecast': lstm_future
                }
                
                print(f"LSTM Weekly RMSE: {lstm_rmse:.2f}")
                print(f"LSTM Weekly MAE: {lstm_mae:.2f}")
                
                if arima_rmse > 0:
                    error_diff_pct = ((arima_rmse - lstm_rmse) / arima_rmse) * 100
                    if error_diff_pct > 0:
                        print(f"LSTM improves weekly forecast accuracy by {abs(error_diff_pct):.2f}% compared to ARIMA")
                    else:
                        print(f"ARIMA improves weekly forecast accuracy by {abs(error_diff_pct):.2f}% compared to LSTM")
                
                self.plot_comparison(weekly_sales.index[-test_size:], test_data.values, 
                                   arima_predictions, lstm_predictions.flatten(),
                                   sales_code, 'weekly')

            except Exception as e:
                print(f"LSTM weekly modeling failed: {e}")
                results['models']['LSTM'] = {'error': str(e)}
        
        self.results[f"{sales_code}_weekly"] = results
        return results
    
    def plot_comparison(self, dates, actual, arima_pred, lstm_pred, sales_code, time_period):
        """Plot comparison between actual values and predictions from both models"""
        plt.figure(figsize=(12, 6))
        plt.plot(dates, actual, 'b-', label='Actual Sales')
        plt.plot(dates, arima_pred, 'r--', label='ARIMA Predictions')
        plt.plot(dates, lstm_pred, 'g--', label='LSTM Predictions')
        plt.title(f'{time_period.capitalize()} Sales Forecasting Comparison - Sales Person {sales_code}')
        plt.xlabel('Date')
        plt.ylabel('Net Amount')
        plt.legend()
        plt.grid(True)
        
        arima_rmse = math.sqrt(mean_squared_error(actual, arima_pred))
        lstm_rmse = math.sqrt(mean_squared_error(actual, lstm_pred))
        
        plt.figtext(0.15, 0.05, f'ARIMA RMSE: {arima_rmse:.2f}', ha='left')
        plt.figtext(0.85, 0.05, f'LSTM RMSE: {lstm_rmse:.2f}', ha='right')
        
        plt.savefig(f'forecast_comparison_{sales_code}_{time_period}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_comparisons(self, sales_codes=['265', '430', '544', '525', '254']):
        """Run comparisons for all specified sales codes"""
        all_results = {
            'daily': {},
            'weekly': {}
        }
        
        for code in sales_codes:
            daily_results = self.compare_models_daily(code)
            weekly_results = self.compare_models_weekly(code)
            
            if daily_results:
                all_results['daily'][code] = daily_results
            if weekly_results:
                all_results['weekly'][code] = weekly_results
        
        # with open('model_comparison_results.json', 'w') as f:
        #     json.dump(all_results, f, indent=4)
        
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, all_results):
        """Generate a summary report of model comparisons"""
        daily_wins = {'ARIMA': 0, 'LSTM': 0}
        weekly_wins = {'ARIMA': 0, 'LSTM': 0}
        
        summary = "# Model Comparison Summary Report\n\n"
        summary += "## Daily Forecasting Results\n\n"
        summary += "| Sales Code | Data Points | ARIMA RMSE | LSTM RMSE | Better Model | Improvement % |\n"
        summary += "|------------|-------------|------------|-----------|--------------|---------------|\n"
        
        for code, results in all_results['daily'].items():
            if 'models' in results and 'ARIMA' in results['models'] and 'LSTM' in results['models']:
                if 'rmse' in results['models']['ARIMA'] and 'rmse' in results['models']['LSTM']:
                    arima_rmse = results['models']['ARIMA']['rmse']
                    lstm_rmse = results['models']['LSTM']['rmse']
                    
                    if arima_rmse < lstm_rmse:
                        better = "ARIMA"
                        improvement = ((lstm_rmse - arima_rmse) / lstm_rmse) * 100
                        daily_wins['ARIMA'] += 1
                    else:
                        better = "LSTM"
                        improvement = ((arima_rmse - lstm_rmse) / arima_rmse) * 100
                        daily_wins['LSTM'] += 1
                    
                    summary += f"| {code} | {results['data_points']} | {arima_rmse:.2f} | {lstm_rmse:.2f} | {better} | {improvement:.2f}% |\n"
        
        summary += "\n## Weekly Forecasting Results\n\n"
        summary += "| Sales Code | Data Points | ARIMA RMSE | LSTM RMSE | Better Model | Improvement % |\n"
        summary += "|------------|-------------|------------|-----------|--------------|---------------|\n"
        
        for code, results in all_results['weekly'].items():
            if 'models' in results and 'ARIMA' in results['models'] and 'LSTM' in results['models']:
                if 'rmse' in results['models']['ARIMA'] and 'rmse' in results['models']['LSTM']:
                    arima_rmse = results['models']['ARIMA']['rmse']
                    lstm_rmse = results['models']['LSTM']['rmse']
                    
                    if arima_rmse < lstm_rmse:
                        better = "ARIMA"
                        improvement = ((lstm_rmse - arima_rmse) / lstm_rmse) * 100
                        weekly_wins['ARIMA'] += 1
                    else:
                        better = "LSTM"
                        improvement = ((arima_rmse - lstm_rmse) / arima_rmse) * 100
                        weekly_wins['LSTM'] += 1
                    
                    summary += f"| {code} | {results['data_points']} | {arima_rmse:.2f} | {lstm_rmse:.2f} | {better} | {improvement:.2f}% |\n"
        
        
        with open('model_comparison_report.md', 'w') as f:
            f.write(summary)
        
        print("\nSummary report generated: model_comparison_report.md")


class SalesForecast:
    def __init__(self, db_file):
        self.db_file = db_file
        self.daily_sales = self.load_data()

    def load_data(self):
        data = pd.read_csv('final_sales_data.csv')
        data['DATE'] = pd.to_datetime(data['DATE'])
        data['NetAmount'] = data['NetAmount'].abs()
        return data.groupby(['DATE', 'SalesPersonCode'])['NetAmount'].sum().reset_index()


    def predict_sales(self, sales_code):
        person_data = self.daily_sales[self.daily_sales['SalesPersonCode'] == sales_code]
        
        if person_data.empty:
            print(f"No data found for SalesPerson {sales_code}.")
            return None

        person_data = person_data.set_index('DATE').asfreq('D').fillna(0)
        
        if len(person_data) < 5:
            print(f"Not enough data for SalesPerson {sales_code}.")
            return None

        try:
            best_aic = float('inf')
            best_order = (7, 1, 1)  
            
            for p in [1, 3, 5, 7]:
                for d in [0, 1]:
                    for q in [0, 1]:
                        try:
                            temp_model = ARIMA(person_data['NetAmount'], order=(p, d, q))
                            temp_result = temp_model.fit()
                            if temp_result.aic < best_aic:
                                best_aic = temp_result.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            print(f"Using ARIMA{best_order} for {sales_code}")
            model = ARIMA(person_data['NetAmount'], order=best_order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=7)

            last_date = person_data.index[-1]
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

            results_df = pd.DataFrame({
                'DATE': forecast_dates,
                'Predicted NetAmount': forecast.values
            })

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

            return results_df

    def predict_lstm(self, sales_code):
        """Make predictions using LSTM model"""
        if not TF_AVAILABLE:
            print("TensorFlow/Keras not available. Cannot use LSTM model.")
            return None
            
        person_data = self.daily_sales[self.daily_sales['SalesPersonCode'] == sales_code]
        
        if person_data.empty:
            print(f"No data found for SalesPerson {sales_code}.")
            return None
            
        ts_data = person_data.groupby('DATE')['NetAmount'].sum().reset_index()
        ts_data = ts_data.set_index('DATE').asfreq('D').fillna(method='ffill').fillna(0)
        
        if len(ts_data) < 30:  
            print(f"Not enough data for LSTM prediction for {sales_code}.")
            return None
            
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(ts_data.values.reshape(-1, 1))
            
            n_steps = 7  
            X, y = [], []
            for i in range(len(scaled_data) - n_steps):
                X.append(scaled_data[i:i + n_steps, 0])
                y.append(scaled_data[i + n_steps, 0])
                
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10)
            model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2, 
                     callbacks=[early_stop], verbose=0)
            
            last_sequence = scaled_data[-n_steps:].reshape(1, n_steps, 1)
            forecasts = []
            
            for _ in range(7):  
                next_pred = model.predict(last_sequence)[0]
                forecasts.append(float(next_pred))
                last_sequence = np.append(last_sequence[:, 1:, :], 
                                        [[next_pred]], axis=1)
            
            forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
            
            last_date = ts_data.index[-1]
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
            
            results_df = pd.DataFrame({
                'DATE': forecast_dates,
                'Predicted NetAmount': forecasts.flatten()
            })
            
            return results_df
            
        except Exception as e:
            print(f"LSTM prediction failed for {sales_code}: {e}")
            return None

    def run_forecast(self, include_lstm=False):
        sales_codes = ['265', '430', '544', '525', '254']
        all_forecasts = {}
        
        for code in sales_codes:
            arima_forecast_df = self.predict_sales(code)
            print(f"ARIMA Forecast for {code}:")
            print(arima_forecast_df)
            
            if include_lstm and TF_AVAILABLE:
                lstm_forecast_df = self.predict_lstm(code)
                print(f"LSTM Forecast for {code}:")
                print(lstm_forecast_df)
                
                if lstm_forecast_df is not None and arima_forecast_df is not None:
                    # Compare forecasts
                    self.compare_forecasts(code, arima_forecast_df, lstm_forecast_df)
            
            if arima_forecast_df is not None:
                all_forecasts[code] = arima_forecast_df
        
        self.detect_consecutive_decreases(all_forecasts)
        
    def compare_forecasts(self, sales_code, arima_df, lstm_df):
        """Compare ARIMA and LSTM forecasts visually"""
        plt.figure(figsize=(12, 6))
        plt.plot(arima_df['DATE'], arima_df['Predicted NetAmount'], 'r-', label='ARIMA Forecast')
        plt.plot(lstm_df['DATE'], lstm_df['Predicted NetAmount'], 'b-', label='LSTM Forecast')
        plt.title(f'Forecast Comparison for SalesPerson {sales_code}')
        plt.xlabel('Date')
        plt.ylabel('Net Amount')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'forecast_compare_{sales_code}.png')
        plt.close()

    def detect_consecutive_decreases(self, forecasts=None, min_consecutive_decreases=3):

        decreasing_sales = {}
        
        if forecasts is None:
            forecasts = {}
            sales_codes = ['265', '430', '544', '525', '254', 'S238', 'S332']
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
        
        # if decreasing_sales:
        #     with open('decreasing_sales_alerts.json', 'w') as f:
        #         json.dump(decreasing_sales, f, indent=4)
            
        #     print(f"Saved {len(decreasing_sales)} sales people with decreasing trends to decreasing_sales_alerts.json")
        # else:
        #     print("No consecutive decreases detected.")
            
        return decreasing_sales


class WeeklySalesForecast:
    def __init__(self, db_file):
        self.db_file = db_file
        self.data = self.load_data()

    def load_data(self):
        data = pd.read_csv('final_sales_data.csv')
        data['DATE'] = pd.to_datetime(data['DATE'])
        data['NetAmount'] = data['NetAmount'].abs()
        return data


    def predict_weekly_sales(self, sales_code, last_date, model_type='ARIMA'):
        person_data = self.data[self.data['SalesPersonCode'] == sales_code]

        if person_data.empty:
            print(f"No data found for SalesPerson {sales_code}.")
            return None

        forecast_week_start = last_date + timedelta(days=1)
        forecast_week_end = forecast_week_start + timedelta(days=6)

        weekly_sales = person_data.groupby(pd.Grouper(key='DATE', freq='W'))['NetAmount'].sum().reset_index()

        if len(weekly_sales) < 5:
            print(f"Not enough data for SalesPerson {sales_code}.")
            return None

        if model_type == 'ARIMA':
            try:
                weekly_sales_series = weekly_sales.set_index('DATE')['NetAmount']
                
                best_aic = float('inf')
                best_order = (1, 1, 1)  
                
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 2):
                            try:
                                temp_model = ARIMA(weekly_sales_series, order=(p, d, q))
                                temp_result = temp_model.fit()
                                if temp_result.aic < best_aic:
                                    best_aic = temp_result.aic
                                    best_order = (p, d, q)
                            except:
                                continue

                model = ARIMA(weekly_sales_series, order=best_order)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=1)  

                results_df = pd.DataFrame({
                    'Week': [forecast_week_start],
                    'Predicted NetAmount': forecast.values
                })

                return results_df

            except Exception as e:
                print(f"ARIMA failed for {sales_code} due to {e}. Switching to mean-based prediction...")
                return self.predict_weekly_with_ets(sales_code, forecast_week_start)
        elif model_type == 'LSTM' and TF_AVAILABLE:
            return self.predict_weekly_with_lstm(sales_code, weekly_sales, forecast_week_start)
        else:
            return self.predict_weekly_with_ets(sales_code, forecast_week_start)

    def predict_weekly_with_ets(self, sales_code, forecast_week_start):
        """Use Exponential Smoothing for weekly forecasts"""
        person_data = self.data[self.data['SalesPersonCode'] == sales_code]
        weekly_sales = person_data.groupby(pd.Grouper(key='DATE', freq='W'))['NetAmount'].sum().reset_index()
        
        try:
            model = ExponentialSmoothing(weekly_sales['NetAmount'], trend='add')
            model_fit = model.fit()
            forecast = model_fit.forecast(1)
            
            results_df = pd.DataFrame({
                'Week': [forecast_week_start],
                'Predicted NetAmount': [forecast[0]]
            })
            
            return results_df
        except:
            mean_forecast = [weekly_sales['NetAmount'].mean()]
            
            results_df = pd.DataFrame({
                'Week': [forecast_week_start],
                'Predicted NetAmount': mean_forecast
            })

            return results_df

    def predict_weekly_with_lstm(self, sales_code, weekly_sales, forecast_week_start):
        """Use LSTM for weekly forecasting"""
        if len(weekly_sales) < 12:  
            print(f"Not enough weekly data for LSTM model for {sales_code}")
            return self.predict_weekly_with_ets(sales_code, forecast_week_start)
            
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(weekly_sales['NetAmount'].values.reshape(-1, 1))
            
            n_steps = 4  
            X, y = [], []
            for i in range(len(scaled_data) - n_steps):
                X.append(scaled_data[i:i + n_steps, 0])
                y.append(scaled_data[i + n_steps, 0])
                
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10)
            model.fit(X, y, batch_size=16, epochs=50, validation_split=0.2, 
                     callbacks=[early_stop], verbose=0)
            
            last_sequence = scaled_data[-n_steps:].reshape(1, n_steps, 1)
            next_week_pred = model.predict(last_sequence)[0]
            
            next_week_forecast = scaler.inverse_transform(np.array([next_week_pred]))[0][0]
            
            results_df = pd.DataFrame({
                'Week': [forecast_week_start],
                'Predicted NetAmount': [next_week_forecast]
            })
            
            return results_df
            
        except Exception as e:
            print(f"LSTM weekly forecast failed for {sales_code}: {e}")
            return self.predict_weekly_with_ets(sales_code, forecast_week_start)

    def generate_weekly_forecasts(self, sales_codes=['265', '430', '544', '525', '254'], include_lstm=False):
        arima_forecasts = {}
        lstm_forecasts = {}
        
        for sales_code in sales_codes:
            last_date = self.data[self.data['SalesPersonCode'] == sales_code]['DATE'].max()
            
            # Get ARIMA forecast
            arima_forecast = self.predict_weekly_sales(sales_code, last_date, 'ARIMA')
            if arima_forecast is not None:
                arima_forecasts[sales_code] = arima_forecast
                print(f"\nARIMA Weekly Forecast for Sales Person {sales_code}:")
                print(f"Prediction Week: {arima_forecast['Week'].dt.strftime('%Y-%m-%d').values[0]} to {(arima_forecast['Week'] + timedelta(days=6)).dt.strftime('%Y-%m-%d').values[0]}")
                print(f"Predicted Net Amount: {arima_forecast['Predicted NetAmount'].values[0]:,.2f}")
            
            if include_lstm and TF_AVAILABLE:
                lstm_forecast = self.predict_weekly_sales(sales_code, last_date, 'LSTM')
                if lstm_forecast is not None:
                    lstm_forecasts[sales_code] = lstm_forecast
                    print(f"\nLSTM Weekly Forecast for Sales Person {sales_code}:")
                    print(f"Prediction Week: {lstm_forecast['Week'].dt.strftime('%Y-%m-%d').values[0]} to {(lstm_forecast['Week'] + timedelta(days=6)).dt.strftime('%Y-%m-%d').values[0]}")
                    print(f"Predicted Net Amount: {lstm_forecast['Predicted NetAmount'].values[0]:,.2f}")
                    
                    if sales_code in arima_forecasts:
                        self.compare_weekly_forecasts(sales_code, 
                                                     arima_forecasts[sales_code]['Predicted NetAmount'].values[0],
                                                     lstm_forecasts[sales_code]['Predicted NetAmount'].values[0])
                        
        return arima_forecasts, lstm_forecasts
    
    def compare_weekly_forecasts(self, sales_code, arima_value, lstm_value):
        """Compare weekly forecasts from ARIMA and LSTM"""
        diff = abs(arima_value - lstm_value)
        diff_pct = (diff / ((arima_value + lstm_value) / 2)) * 100
        
        print(f"\nComparison for {sales_code}:")
        print(f"ARIMA forecast: {arima_value:,.2f}")
        print(f"LSTM forecast:  {lstm_value:,.2f}")
        print(f"Difference:     {diff:,.2f} ({diff_pct:.2f}%)")
        
        if diff_pct > 20:
            print("Warning: Large difference between forecasts. Consider model evaluation.")


def main():
    parser = argparse.ArgumentParser(description='Run sales forecasting')
    
    parser.add_argument('--compare', action='store_true', help='Run model comparison')
    parser.add_argument('--daily', action='store_true', help='Run daily forecasts')
    parser.add_argument('--weekly', action='store_true', help='Run weekly forecasts')
    parser.add_argument('--include-lstm', action='store_true', help='Include LSTM in forecasts')
    parser.add_argument('--sales-codes', nargs='+', default=['265', '430', '544', '525', '254'], 
                       help='Sales person codes to process')
    
    args = parser.parse_args()
    
    if args.compare:
        print("Running model comparison...")
        comparison = ModelComparison()
        comparison.run_all_comparisons(args.sales_codes)
        
    if args.daily:
        print("Running daily forecasts...")
        daily_forecast = SalesForecast(DB_FILE)
        daily_forecast.run_forecast(args.include_lstm)
        
    if args.weekly:
        print("Running weekly forecasts...")
        weekly_forecast = WeeklySalesForecast(WEEKLY_DB_FILE)
        weekly_forecast.generate_weekly_forecasts(args.sales_codes, args.include_lstm)
        
    if not (args.compare or args.daily or args.weekly):
        print("Running all forecasting components...")
        
        comparison = ModelComparison()
        results = comparison.run_all_comparisons(args.sales_codes)
        
        daily_forecast = SalesForecast(DB_FILE)
        daily_forecast.run_forecast(False) 
        
        weekly_forecast = WeeklySalesForecast(WEEKLY_DB_FILE)
        weekly_forecast.generate_weekly_forecasts(args.sales_codes, False)


if __name__ == "__main__":
    main()