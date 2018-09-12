import json
import requests
from urllib.request import urlopen
from collections import OrderedDict
from bs4 import BeautifulSoup
import dateparser
import datetime
from datetime import timedelta
from datetime import datetime
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas as pd
from pandas.io.json import json_normalize
import pandas_datareader.data as web
import numpy as np
import plotly as plt
import plotly.graph_objs as go

class MarketsListings:
	def __init__(self):
		self._cryptocurrency_base_url = 'https://api.coinmarketcap.com/v2/'
		self._cryptoccurency_listings = self.get_coin_translator()
		self._stock_listings = self.get_nasdaq_listings()

	def get_coin_translator(self):
		'''
    	coinmarketcap contains listings of cryptocurrencies
    	each one contaning the id,name,symbol and website_slug.
    	This function gets every coin listed and stores it in memory.
    	Due to coinmarketcap api, this information is essential in order
    	to request the correct data.
    	'''
		raw_data = requests.get(self._cryptocurrency_base_url+'listings/')
		raw_json = raw_data.json()

		data = {}
		for listing in raw_json['data']:
			data[listing['name'].upper()] = listing
		return data

	def get_nasdaq_listings(self):
		'''
		returns the stock market listings of their ticker and security name
		'''
		listings = get_nasdaq_symbols()
		
		data = OrderedDict()
		for i in range (len(listings)):
			data[listings.iloc[i]['NASDAQ Symbol']] = listings.iloc[i]['Security Name']	
		return data

	def get_stock_listings(self):
		'''
		returns the stock listings variable
		'''
		return self._stock_listings

	def get_cryptocurrency_listings(self):
		'''
		returns the cryptocurrency listings from coinmarketcap
		'''
		return self._cryptoccurency_listings

	def get_top_crypto(self,num = 25):
		'''
		returns the top cryptocurrencies on coinmarketcap
		@param num (int): number of top coins one desires to request. Accepted Values: Positive integers
		'''
		return json.loads(requests.get(self._cryptocurrency_base_url+'ticker/?limit='+str(num)).text,object_pairs_hook = OrderedDict)
	
	def get_crypto_id(self,name):
		'''
		returns the cryptocurrency id on coinmarketcap.com
		@param name (string): name of the coin to pull.
		'''
		return self._cryptoccurency_listings[name.upper()]["id"]

class Asset:
	'''
	Retrieves stock information using pandas data reader. This can also retrieve cryptocurrency data from coinmarketcap.com

	'''
	#used as defualt starting dates when pulling data
	default_start = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
	default_end = datetime.now().strftime('%Y%m%d')
	
	def __init__(self,asset,start=default_start,end=default_end):
		'''
		Initializes the Asset class. 
		@param asset (string): The asset name. For stock, provide a ticker ex)AAPL. For cryptocurrency provide the full
		name of the coin ex)bitcoin.
		@param start_date (string): The begining date of the quotes. format: "YYYYmmdd"
		@param end_date (string): The end date of when to stop pulling quotes. format: "YYYYmmdd" 
		'''
		self.asset = asset.upper()
		self.candles = self.historical_data(start,end)
		if isinstance(self.candles,str):
			raise ValueError(self.candles)
		self.ta = Technical_Analysis(self.candles)

	def historical_data(self,start_date = default_start, end_date = default_end):
		'''
			Pulls historical stock data using pandas webdata reader and historical cryptocurrency data between two dates
			@param start_date (string): The begining date of the quotes. format: "YYYYmmdd"
			@param end_date (string): The end date of when to stop pulling quotes. format: "YYYYmmdd" 
		'''
		try:
			return web.DataReader(self.asset,'iex',start_date,end_date)
		except:
			pass

		try:
			url = 'https://coinmarketcap.com/currencies/'+self.asset+'/historical-data/?start='+start_date+'&end='+end_date
			raw_html = urlopen(url)
			soup = BeautifulSoup(raw_html.read(),"html.parser")
			history = soup.find("table",attrs = {"table"})
			headings = [th.get_text().replace("*","").replace(" ","").lower() for th in history.find("tr").find_all("th")]
			candle_data = OrderedDict()
			data = {}
			count = 0 #this is used to determine which column is being used
			date = ""
			#this gets all the rows from the table. It is one constant stream of data
			for td in history.find_all("td"):
				value = td.text.strip()
				if count == 0:
					#Creates the date as the key
					date = str(dateparser.parse(value)).split(" ")[0]#need to remove dateparser 
					candle_data[date] = []
					count+=1
				elif count == len(headings)-1:
					#the final column gets assigned and gets put into the json dictionary
					data[headings[count]] = int(value.replace(",","").replace("-","0"))
					candle_data[date] = data
					data = {}
					count = 0
				else:
					data[headings[count]] = float(value.replace(",",""))
					count+=1
			
			candle_data = OrderedDict(reversed(list(candle_data.items()))) #reversed it in order to have the most recent items on the bottom
			
			frames = []
			for day in candle_data.keys(): #builds the panda dataframe
				df = pd.DataFrame(candle_data[day],index = [day])
				frames.append(df)

			return pd.concat(frames)
		except:
			return "No Such Asset Exists"
		
	def change_asset(self,asset,start = default_start,end = default_end):
		'''
		Changes the asset. If it fails to find the asset, the original asset remains
		@param asset (string): The asset name. For stock, provide a ticker ex)AAPL. For cryptocurrency provide the full
		name of the coin ex)bitcoin.
		@param start_date (string): The begining date of the quotes. format: "YYYYmmdd"
		@param end_date (string): The end date of when to stop pulling quotes. format: "YYYYmmdd" 
		'''
		old_asset = self.asset
		old_candles = self.candles
		self.asset = asset
		self.candles = self.historical_data(start,end)

		if isinstance(self.candles,str):
			self.asset = old_asset
			self.candles = old_candles
			return print("No Such Asset Exists")

	def quote(self):
		'''
		returns the latest day quote
		'''
		return self.candles.tail(1)

	def sma(self,time_period = 10, series_type = "close", comb = False):
		'''
		Queries Simple Moving Average
		'''
		return self.data_compliance(self.ta.sma(time_period,series_type),comb)

	def ema(self,time_period = 10, series_type = "close",comb = False):
		'''
		Queries Exponential Moving Average
		'''
		return self.data_compliance(self.ta.ema(time_period,series_type),comb)

	def macd(self,fast_period = 12,slow_period = 26,signal_period = 9, series_type = "close",comb = False):
		'''
		Queries Moving Average Convergence/Divergence
		'''
		return self.data_compliance(self.ta.macd(fast_period,slow_period,signal_period,series_type),comb)

	def rsi(self,time_period = 14, series_type = "close",comb = False):
		'''
		Queries the Relative Strength Index
		'''
		return self.data_compliance(self.ta.rsi(time_period,series_type),comb)

	def cci(self,time_period = 20,comb = False):
		'''
		Queries the Commodity Channel Index 
		'''
		return self.data_compliance(self.ta.cci(time_period),comb)

	def bbands(self,time_period = 20, series_type = "close", nbdevup = 2, nbdevdn = 2, comb = False):
		'''
		Queries the Bollinger Bands
		'''
		return self.data_compliance(self.ta.bbands(time_period,series_type,nbdevup,nbdevdn),comb)

	def obv(self,comb=False):
		'''
		Queries the On-Balance Volume
		'''
		return self.data_compliance(self.ta.obv(),comb)

	def data_compliance(self,data,combine):
		'''
		Combines data frames together with the candle dataframes
		@param data: A dataFrame.
		'''
		if combine:
			self.candles = pd.concat([self.candles,data],axis=1,sort=True)
			return self.candles
		else:
			return data

	def graph(self,indicators=[]):
		graph = Graph(self.candles,self.asset)
		graph.plot()

class Technical_Analysis:
	'''
	This is holds code for Technical Analasis on prices.
	The class needs to first take in a pandas dataframe consisting of closing,open,high,low prices.
	'''
	def __init__(self,df):
		self.df = df

	def check_intervals(self,time_period):
		'''
		Checks to make sure there are enough candle stick data for the time period
		@param time_period (int): amount of data points one desires to check
		'''
		if time_period <= len(self.df.index):
			return True
		else:
			return False

	def sma(self,time_period, series_type):
		'''
		Queries the Simple Moving Average
		@param time_period (int): amount of data points one desires to use. 
		@param series_type (string): The price type in the time series. 
		'''

		if self.check_intervals(time_period):
			value = self.df[series_type].rolling(window=time_period).mean()
			return pd.DataFrame(value.values,index=value.index,columns=['sma_'+str(time_period)]).dropna()
		else:
			return "Not enough data points, try to get more historical data"

	def ema(self,time_period,series_type):
		'''
		Queries the Exponential Moving Average
		@param time_period (int): number of data points used to calculate each moving average value.
		@param series_type (string): The price type in the time series.
		'''
		if self.check_intervals(time_period):
			sma = self.df[series_type].rolling(window=time_period,min_periods=time_period).mean()[:time_period]
			rest = self.df[series_type][time_period:]
			ema = pd.concat([sma,rest]).ewm(span=time_period,adjust=False).mean()
			return pd.DataFrame(ema.values,index=ema.index,columns=['ema_'+str(time_period)]).dropna()
		else:
			return "Not enough data points, try to get more historical data"

	def macd(self,fast_period,slow_period,signal_period,series_type):
		'''
		Queries the Moving Average Convergence/Divergence
		@param fast_period (int). Default = 12
		@param slow_period (int). Default = 26
		@param signal_period (int). Default = 9 
		'''
		if self.check_intervals(slow_period+signal_period):
			fast_period = self.ema(fast_period,series_type)['ema_'+str(fast_period)]
			slow_period = self.ema(slow_period,series_type)['ema_'+str(slow_period)]
			macd_line = (fast_period-slow_period).dropna()
		
			init_sum = macd_line.rolling(window=signal_period,min_periods=signal_period).mean()[:signal_period].dropna()
			rest = macd_line[signal_period:]
			signal_line = pd.concat([init_sum,rest]).ewm(span=signal_period,adjust=False).mean()
			
			macd_histogram = macd_line-signal_line
			join = pd.concat([macd_line,signal_line,macd_histogram],axis=1,sort=True)

			return pd.DataFrame(join.values,index=join.index,columns=['macd_line',"signal_line","macd_histogram"])
		else:
			return "Not enough data points, try to get more historical data"
	
	def rsi(self,time_period, series_type):
		'''
		Queries the Relative Strength Index
		Used some of the code from http://www.andrewshamlet.net/2017/06/10/python-tutorial-rsi/
		@param time_period (int): Number of data points used to calculate the rsi. 
		@param series_type (string): Price type in the time series. 
		'''
		if self.check_intervals(time_period):
			value = self.df[series_type].diff().dropna()
			gain = value * 0
			loss = value * 0

			gain[value>0] = value[value>0]
			loss[value<0] = -value[value<0]
			
			gain[gain.index[time_period-1]] = np.mean(gain[:time_period])
			loss[loss.index[time_period-1]] = np.mean(loss[:time_period])

			gain = gain.drop(gain.index[:(time_period-1)])
			loss = loss.drop(loss.index[:(time_period-1)])

			rs = gain[0]/loss[0]
			rsi_values = [100-(100/(1+rs))]
			
			for i in range(1,len(gain.index)):
				gain[i] = ((gain[i-1]*(time_period-1)) + gain[i])/time_period
				loss[i] = ((loss[i-1]*(time_period-1)) + loss[i])/time_period
				rsi_values.append(100-(100/(1+(gain[i]/loss[i]))))

			return pd.DataFrame(rsi_values,index=gain.index,columns=['rsi_'+str(time_period)])
		else:
			return "Not enough data points, try to get more historical data"

	def cci(self,time_period = 20):
		'''
		Queries the Commodity Channel Index 
		@param time_period (int): Number of data points to calculate CCI.
		'''
		if self.check_intervals(time_period):
			tp = (self.df['high'] + self.df['low']+self.df['close'])/3
			cci = (tp - tp.rolling(window=time_period).mean()) / (.015 * tp.rolling(window=time_period).std())
			return pd.DataFrame(cci.values,index=cci.index,columns=['cci_'+str(time_period)]).dropna()
		else:
			return "Not enough data points, try to get more historical data"

	def bbands(self,time_period, series_type, nbdevup, nbdevdn):
		'''
		Queries the Bollinger Bands
		@param time_period (int): Number of data points used to calculate BBands. 
		@param series_type (string): Desired price type in the time series.
		@param nbdevup (int): Standard deviation multiplier of the upper band. 
		@param nbdevdn (int): Standard deviation multiplier of the lower band.
		'''
		if self.check_intervals(time_period):
			stdev = self.df[series_type].rolling(window=time_period).std().dropna()
			middle_band = self.df[series_type].rolling(window=time_period).mean().dropna()
			upper_band = middle_band+(stdev*nbdevup)
			lower_band = middle_band-(stdev*nbdevup)
			join = pd.concat([middle_band,upper_band,lower_band],axis=1,sort=True)
			return pd.DataFrame(join.values,index=join.index,columns=['middle_band_'+str(time_period),'upper_band_'+str(time_period),'lower_band_'+str(time_period)])
		else:
			return "Not enough data points, try to get more historical data"

	def obv(self):
		'''
		Queries the On-Balance Volume
		'''
		dif_prices = self.df['close'].diff()#checks if previous price was > or < to 0
		volume = self.df['volume'].copy() 
		volume[dif_prices < 0] = -volume #negates the volume where previous price was less
		return pd.DataFrame(volume.cumsum().values,index=volume.index,columns=['obv'])

class Graph:
	def __init__(self,df,asset):
		self.df = df
		self.asset = asset
		self.name = self.asset+'_'+self.df.index[0]+'_'+self.df.index[-1]

	def plot(self):
		trace = go.Candlestick(x=self.df.index,
                       open=self.df.open,
                       high=self.df.high,
                       low=self.df.low,
                       close=self.df.close)
		data = [trace]
		plt.offline.plot(data, filename=self.name+'.html')

	def volume_bars(self):
		colors = []

		for i in range(len(self.df.close)):
		    if i != 0:
		        if self.df.close[i] > self.df.close[i-1]:
		            colors.append('#17BECF')
		        else:
		            colors.append('#7F7F7F')
		    else:
		        colors.append('#7F7F7F')
		
		return go.Bar(x=self.df.index,y=self.df.volume,marker=dict(color=colors),name='Volume')