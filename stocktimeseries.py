# pip install streamlit prophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit Dashboard          
st.set_page_config(page_title ="Woods and Pop Ltd", page_icon =":guardsman:", layout ="centered")
st.image("logo.jpeg", width = 600)
st.title('Woods and Pop Ltd')
st.header('💰💸AI Stock Forecaster App')
st.subheader("Historical Data Analysis of the selected Stock")

#Forecasting for 30 Stocks

stocks = ('GOOGL','DIS','PACW','HOOD','GT','U','TLRY','BYND','SONO','APP','SBFM','BE','BABA','CRSR','JAGX','POLA','TTD','ZURA','PTON','BBIG','JD','PSNY','META','IBRX','CIX.TO','WAL','MAXN','PFE','MANU')
	  
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Downloaded Raw Data of the selected stock')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data analysis with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecasted data of the selected stock')
st.write(forecast.tail())
    
st.write(f'Forecasted plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecasted components")
fig2 = m.plot_components(forecast)
st.write(fig2)
