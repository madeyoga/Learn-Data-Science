import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')
### 1
##start = dt.datetime(2000, 1, 1)
##end   = dt.datetime(2018, 12, 14)
##
##df = web.DataReader('TSLA', 'yahoo', start, end)
##print(df.tail())
##
##df.to_csv('Datasets/tsla.csv')

df = pd.read_csv('Datasets/tsla.csv', parse_dates=True, index_col=0)
### 2 
##df['100MovingAvg'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
##df.dropna(inplace=True)
print(df.head())

### 3
##df.plot()
##df['Adj Close'].plot()
##df[['Open', 'High']].plot()
##plt.show()

### 5 open high low close data
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)

df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

print(df_ohlc.head())

##df_ohlc = df['Adj Close'].resample('10D').mean()

ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

plt.show()

### 4
##ax1.plot(df.index, df['Adj Close'])
##ax1.plot(df.index, df['100MovingAvg'])
##ax1.legend(['Adj Close', '100 Moving Average'])
##ax2.bar(df.index, df['Volume'])

##plt.show()
