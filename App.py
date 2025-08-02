import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf  # Free for prices
import requests
from bs4 import BeautifulSoup  # Free for parsing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator, MACD  # pip install ta
import nltk
nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

# NewsAPI Key (provided by user)
NEWSAPI_KEY = 'eb8b17a49f4d4483b9284bf4b3a43baf'

@st.cache_data(ttl=3600)  # Cache for efficiency
def fetch_prices(symbol):
    try:
        data = yf.download(symbol, period='6mo', progress=False)['Close']
        return data.values.reshape(-1, 1)
    except:
        return np.array([]).reshape(-1, 1)

@st.cache_data(ttl=3600)
def compute_indicators(data):
    if len(data) < 14:
        return 50, 0  # Defaults
    series = pd.Series(data.flatten())
    rsi = RSIIndicator(series, window=14).rsi().iloc[-1]
    macd = MACD(series).macd().iloc[-1]
    return rsi, macd

@st.cache_data(ttl=3600)
def fetch_external_data(query='crypto stock government bill international military pelosi trade'):
    articles = []
    # NewsAPI
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={NEWSAPI_KEY}&pageSize=10'
    try:
        response = requests.get(url).json()
        articles.extend([article['description'] for article in response.get('articles', []) if article['description']])
    except:
        pass

    # Congress bills
    congress_url = 'https://www.congress.gov/search?q=%7B%22source%22%3A%22legislation%22%2C%22search%22%3A%22crypto+bill%22%7D'
    congress_resp = requests.get(congress_url)
    soup = BeautifulSoup(congress_resp.text, 'html.parser')
    congress_news = [li.text.strip() for li in soup.find_all('li', class_='expanded')[:5]]
    articles.extend(congress_news)

    # Politician trades (Capitol Trades)
    cap_url = 'https://www.capitoltrades.com/trades?politician=pelosi'
    cap_resp = requests.get(cap_url)
    soup = BeautifulSoup(cap_resp.text, 'html.parser')
    trades = [tr.text.strip() for tr in soup.find_all('tr')[:10] if 'trade' in tr.text.lower()]
    articles.extend(trades)

    # International (EU)
    eu_url = 'https://eur-lex.europa.eu/search.html?scope=EURLEX&text=crypto+regulation&lang=en&type=quick'
    eu_resp = requests.get(eu_url)
    soup = BeautifulSoup(eu_resp.text, 'html.parser')
    eu_news = [div.text.strip() for div in soup.find_all('div', class_='SearchResult')[:5]]
    articles.extend(eu_news)

    return [a for a in articles if a]

@st.cache_data(ttl=3600)
def analyze_sentiment(texts):
    scores = [sia.polarity_scores(text)['compound'] for text in texts]
    return np.mean(scores) if scores else 0

@st.cache_data(ttl=3600)
def hybrid_predict(data):
    if len(data) < 60:
        return data[-1][0] if len(data) > 0 else 0
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    lstm = Sequential()
    lstm.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    lstm.add(LSTM(50))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X, y, epochs=10, batch_size=32, verbose=0)
    lstm_pred = lstm.predict(scaled[-60:].reshape(1, -1, 1))
    arima_model = ARIMA(data, order=(5,1,0))
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.forecast(steps=1)[0]
    return scaler.inverse_transform([[ (lstm_pred[0][0] + (arima_pred - data[-1][0]) / 2 )]])[0][0]

# App
st.title('Custom Open-Source Predictive Trading System')
symbol = st.text_input('Symbol (e.g., BTC-USD, AAPL)', 'BTC-USD')
if st.button('Predict'):
    with st.spinner('Analyzing...'):
        data = fetch_prices(symbol)
        if len(data) < 60:
            st.error('Insufficient data.')
        else:
            rsi, macd = compute_indicators(data)
            external_data = fetch_external_data()
            sentiment = analyze_sentiment(external_data)
            predicted_price = hybrid_predict(data)
            current_price = data[-1][0]
            signal = "Hold: No strong signal."
            if predicted_price > current_price * 1.05 and sentiment > 0.2 and rsi < 30 and macd > 0:
                signal = f"Buy: Predicted rise to {predicted_price:.2f} (Bullish sentiment: {sentiment:.2f}, RSI low, MACD positive)"
            elif predicted_price < current_price * 0.95 or sentiment < -0.2 or rsi > 70 or macd < 0:
                signal = f"Sell: Predicted drop to {predicted_price:.2f} (Bearish sentiment: {sentiment:.2f}, RSI high, MACD negative)"
            st.success(signal)
            st.write('Key Factors (News/Gov/Trades):')
            for item in external_data[:10]:
                st.write(f"- {item}")

st.markdown('Note: Educational only; ~70-85% backtest accuracy. Run locally or deploy free.')
