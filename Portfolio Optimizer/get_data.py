import pandas as pd
import pandas_datareader as pdr
import numpy as np
import datetime as dt

start_date = dt.datetime(2016,3,1)
end_date = dt.datetime(2021,12,8)
days_between = end_date - start_date
rf = 0.01


# Import CRIX Historical Data
df = pd.read_excel('Data\CRIX.xlsx')
crix_df = df.set_index('Date')

# Get Data for each Asset Class
tickers = ['SPY', 'IYR', 'GLD', 'BND' ]
ticker_prices = pdr.DataReader(tickers, 'yahoo', start= start_date, end= end_date)['Adj Close']

# Combine each dataset and remove NaN Rows
data_frames = [crix_df, ticker_prices]
price_data = pd.concat(data_frames, axis=1,).dropna()

price_data.to_excel('Data\price_data.xlsx')