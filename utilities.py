import sqlite3
import os
import json
import re

DB_FILE = "daily_forecast.db"
WEEKLY_DB_FILE = "weekly_forecast.db"
MONTHLY_DB_FILE = "monthly_forecast.db"

def get_daily_forecast(sales_code, date):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''SELECT Predicted_NetAmount FROM daily_forecast 
                      WHERE SalesPersonCode = ? AND DATE = ?''', (sales_code, date))
    result = cursor.fetchone()
    
    conn.close()

    if result:
        return {'SalesPersonCode': sales_code, 'DATE': date, 'Predicted_NetAmount': round(result[0],2)}
    else:
        return {'error': 'No forecast found for the given SalesPersonCode and DATE'}


def get_weekly_forecast(sales_code):
    conn = sqlite3.connect(WEEKLY_DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT predicted_netamount, Week  
        FROM weekly_forecast 
        WHERE SalesPersonCode = ?
        ORDER BY id DESC 
        LIMIT 1
    ''', (sales_code,))
    result = cursor.fetchone()
    
    conn.close()

    if result:
        return {'SalesPersonCode': sales_code, 'Week': result[1], 'Predicted_NetAmount': round(result[0],2)}
    else:
        return {'error': 'No forecast found for the given SalesPersonCode and DATE'}

def get_monthly_forecast(sales_code):
    conn = sqlite3.connect(MONTHLY_DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT predicted_netamount, Month, SHAP_Report  
        FROM monthly_forecast 
        WHERE SalesPersonCode = ?
        ORDER BY id DESC 
        LIMIT 1
    ''', (sales_code,))
    result = cursor.fetchone()
    
    conn.close()

    if result:
        shap_report_text = result[2]
        shap_factors = []

        if shap_report_text:  # only if not None or empty
            for line in shap_report_text.strip().split('\n'):
                line = line.strip()

                match_original = re.match(r'Top (\d+) \((\d{4}-\d{2})\) (.*?) (increased|decreased) prediction by ([\d.]+)', line)
                if match_original:
                    rank, date, metric, effect, value = match_original.groups()
                    shap_factors.append({
                        "rank": int(rank),
                        # "date": date,
                        "metric": metric,
                        # "effect": effect,
                        "value": float(value)
                    })
                    continue

                match_simple = re.match(r'(\d+)\.\s(.*?):\s([\d.]+)% impact on prediction', line)
                if match_simple:
                    rank, metric, value = match_simple.groups()
                    shap_factors.append({
                        "rank": int(rank),
                        # "date": result[1],  
                        "metric": metric.strip(),
                        # "effect": "impact",  
                        "value": float(value)
                    })
        
        return {
            'SalesPersonCode': sales_code,
            'Month': result[1],
            'Predicted_NetAmount': round(result[0], 2),
            'SHAP_Report': {
                'factors': shap_factors
            }
        }
    else:
        return {'error': 'No forecast found for the given SalesPersonCode'}
    
def load_decreasing_sales():
    file_path = 'decreasing_sales_alerts.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None    