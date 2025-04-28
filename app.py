from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from utilities import get_daily_forecast, get_weekly_forecast, get_monthly_forecast, load_decreasing_sales

app = Flask(__name__)

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin', '')
    response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.route('/get_forecast', methods=['GET'])
def get_forecast():
    """
    Endpoint to get a daily sales forecast for a specific salesperson.

    Query Parameters:
    - SalesPersonCode (required): The unique code of the salesperson.
    - DATE (required): The specific date for which the forecast is requested.

    """
    sales_code = request.args.get('SalesPersonCode')
    date = request.args.get('DATE')

    if not sales_code or not date:
        return jsonify({'error': 'Missing SalesPersonCode or DATE'}), 400
    
    forecast_data = get_daily_forecast(sales_code, date)
    return jsonify(forecast_data)

@app.route('/weekly_forecast', methods=['GET'])
def weekly_forecast():
    """
    Endpoint to get a weekly sales forecast for a specific salesperson.

    Query Parameters:
    - SalesPersonCode (required): The unique code of the salesperson.

    """
    sales_code = request.args.get('SalesPersonCode')

    if not sales_code:
        return jsonify({'error': 'Missing SalesPersonCode'}), 400
    
    forecast_data = get_weekly_forecast(sales_code)
    return jsonify(forecast_data)

@app.route('/monthly_forecast', methods=['GET'])
def monthly_forecast():
    """
    Endpoint to get a monthly sales forecast for a specific salesperson.

    Query Parameters:
    - SalesPersonCode (required): The unique code of the salesperson.

    """
    sales_code = request.args.get('SalesPersonCode')

    if not sales_code:
        return jsonify({'error': 'Missing SalesPersonCode'}), 400
    
    forecast_data = get_monthly_forecast(sales_code)
    return jsonify(forecast_data)

@app.route('/get_decreasing_sales', methods=['GET'])
def get_decreasing_sales():
    """
    Endpoint to detect consecutive sales decreases for all salespeople.

    """
    data = load_decreasing_sales()
    if data:
        return jsonify(data)
    return jsonify({"message": "no consecutive decrease detected."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
