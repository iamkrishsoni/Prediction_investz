import pandas as pd
import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Alpha Vantage API setup
api_key = "YHSR8AOE104C80OV"
all_stocks = ['AAPL', 'MSFT']  # Only two stocks

app = Flask(__name__)
CORS(app)

# Fetch Stock Data
def fetch_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    if "Time Series (Daily)" not in data:
        print(f"Failed to fetch data for {symbol}: {data}")
        return None  # Invalid response
    daily_data = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(daily_data, orient='index')
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    }).astype(float)
    df['Date'] = df.index
    df.index = pd.to_datetime(df['Date'])
    df.sort_index(inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Fetch Company Details
def fetch_company_details(symbol):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    if "Name" not in data:
        print(f"Failed to fetch company details for {symbol}")
        return None
    return {
        "Name": data.get("Name", "N/A"),
        "Symbol": symbol,
        "Sector": data.get("Sector", "N/A"),
        "Industry": data.get("Industry", "N/A"),
        "MarketCap": data.get("MarketCapitalization", "N/A"),
        "Description": data.get("Description", "N/A")
    }

# Prepare Data
def prepare_data(df):
    df['Return'] = df['Close'].pct_change()  # Daily returns
    df['Target'] = (df['Return'] > 0).astype(int)  # Positive or negative
    df['MA10'] = df['Close'].rolling(window=10).mean()  # Moving Average
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df = df.dropna()
    return df

# Train the Model
def train_model():
    combined_data = []
    for stock in all_stocks:  # Use both stocks for training
        print(f"Fetching data for stock: {stock}")
        df = fetch_stock_data(stock)
        if df is not None and not df.empty:
            print(f"Preparing data for stock: {stock}")
            df = prepare_data(df)
            df['Stock'] = stock
            combined_data.append(df)
        else:
            print(f"No data available for stock: {stock}")

    if not combined_data:
        raise ValueError("No valid stock data available for training.")

    combined_data = pd.concat(combined_data)

    # Train a model
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'RSI']
    X = combined_data[features]
    y = combined_data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, features, accuracy

# Initialize model, features, and accuracy
try:
    model, features, accuracy = train_model()
    print(f"Model trained successfully with accuracy: {accuracy}")
except ValueError as e:
    print(f"Error during model training: {e}")
    model, features, accuracy = None, None, None

# API to Fetch Random Predictions
@app.route('/predict', methods=['GET'])
def predict_random_stocks():
    if model is None:
        return jsonify({"error": "Model not trained. Cannot make predictions."}), 500

    predictions = []

    for stock in all_stocks:  # Predict for both stocks
        stock_df = fetch_stock_data(stock)
        if stock_df is None:
            continue
        prepared_data = prepare_data(stock_df)
        latest_data = prepared_data.iloc[-1:][features]
        prediction = model.predict(latest_data)[0]
        company_details = fetch_company_details(stock)
        if company_details:
            company_details['Prediction'] = "Positive" if prediction == 1 else "Negative"
            company_details.update({
                "LatestOpen": prepared_data.iloc[-1]['Open'],
                "LatestHigh": prepared_data.iloc[-1]['High'],
                "LatestLow": prepared_data.iloc[-1]['Low'],
                "LatestClose": prepared_data.iloc[-1]['Close'],
                "LatestVolume": prepared_data.iloc[-1]['Volume']
            })
            predictions.append(company_details)

    return jsonify({
        "accuracy": accuracy,
        "predictions": predictions
    })

# API to Search Specific Stock
@app.route('/search', methods=['GET'])
def search_stock_with_prediction():
    if model is None:
        return jsonify({"error": "Model not trained. Cannot make predictions."}), 500

    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    # Fetch stock data
    stock_df = fetch_stock_data(symbol)
    if stock_df is None:
        return jsonify({"error": "Invalid stock symbol or data not available"}), 404

    # Prepare the data for prediction
    try:
        prepared_data = prepare_data(stock_df)
        latest_data = prepared_data.iloc[-1:][features]
    except Exception as e:
        return jsonify({"error": "Unable to prepare data for prediction", "details": str(e)}), 500

    # Fetch company details
    company_details = fetch_company_details(symbol)
    if not company_details:
        return jsonify({"error": "Company details not available"}), 404

    # Predict using the model
    prediction = model.predict(latest_data)[0]
    company_details['Prediction'] = "Positive" if prediction == 1 else "Negative"

    # Add the latest stock price details
    company_details.update({
        "LatestOpen": prepared_data.iloc[-1]['Open'],
        "LatestHigh": prepared_data.iloc[-1]['High'],
        "LatestLow": prepared_data.iloc[-1]['Low'],
        "LatestClose": prepared_data.iloc[-1]['Close'],
        "LatestVolume": prepared_data.iloc[-1]['Volume']
    })

    return jsonify(company_details)

# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)
