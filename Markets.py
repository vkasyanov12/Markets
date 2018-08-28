import json
import requests
from urllib.request import urlopen
from collections import OrderedDict
from bs4 import BeautifulSoup
import collections
import dateparser
import datetime
from datetime import timedelta
from datetime import datetime
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas as pd
from pandas.io.json import json_normalize
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
#from matplotlib.finance import candlestick2_ohlc
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

style.use('ggplot')

'''
works only with python 2.7
'''

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

class Stock:
	'''
	Retrieves stock information using Alpha Vantage API. 
	'''
	def __init__(self,ticker,start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),end_date = datetime.now().strftime('%Y%m%d')):
		'''
		Initializes the Stock Object
		@param ticker (string): Ticker of the stock. 
		@param start_date (string): The begining date of the quotes. format: "YYYYmmdd"
		@param end_date (string): The end date of when to stop pulling quotes. format: "YYYYmmdd"
		'''
		self.stock_ticker = ticker.upper()
		self.candles = self.historical_data(start_date,end_date)
		self.ta = Technical_Analysis(self.candles)

	def historical_data(self,start_date= (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),end_date = datetime.now().strftime('%Y%m%d')):
		'''
		Pulls historical data on the stock
		@param start_date (string): The begining date of the quotes. format: "YYYYmmdd"
		@param end_date (string): The end date of when to stop pulling quotes. format: "YYYYmmdd"
		'''
		self.candles = web.DataReader(self.stock_ticker,'iex',start_date,end_date)
		return self.candles

	def quote(self):
		'''
		returns the latest stock information
		'''
		return self.candles.tail(1)

	def sma(self,time_period = 10, series_type = "close"):
		'''
		Queries Simple Moving Average
		'''
		return self.ta.sma(time_period,series_type)

	def ema(self,time_period = 10, series_type = "close"):
		'''
		Queries Exponential Moving Average
		'''
		return self.ta.ema(time_period,series_type)

	def macd(self,fast_period = 12,slow_period = 26,signal_period = 9, series_type = "close"):
		'''
		Queries Moving Average Convergence/Divergence
		'''
		return self.ta.macd(fast_period,slow_period,signal_period,series_type)

	def rsi(self,time_period = 14, series_type = "close"):
		'''
		Queries the Relative Strength Index
		'''
		return self.ta.rsi(time_period,series_type)

	def cci(self,time_period = 20):
		'''
		Queries the Commodity Channel Index 
		'''
		return self.ta.cci(time_period)

	def bbands(self,time_period = 20, series_type = "close", nbdevup = 2, nbdevdn = 2):
		'''
		Queries the Bollinger Bands
		'''
		return self.ta.bbands(time_period,series_type,nbdevup,nbdevdn)

class Coin:
	'''
	This is an api for coinmarketcap.com. It pulls current and historical data from the website
	and returns values in json format Currently, historical data only supports daily information.
	'''
	def __init__(self,coin_id,start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),end_date = datetime.now().strftime('%Y%m%d')):
		'''
		inititializes the object
		@param coin_id (int): cryptocurrency id from coinmarketcap.com. This reliant on Market class. Accepted Values: Positive integers
		@param start_date (string): The begining date of the quotes. format: "YYYYmmdd", Accepted values: Dates>20130428
		@param end_date (string): The end date of when to stop pulling quotes. format: "YYYYmmdd"
		'''
		self._base_url = 'https://api.coinmarketcap.com/v2/'
		self._id = coin_id
		self._symbol = None
		self._website_slug = None 
		self._name = None
		self.candles = None
		self.get_info(coin_id)
		self.historical_data(start_date,end_date)
		self.ta = Technical_Analysis(self.candles)

	def get_info(self,coin_id):
		'''
		sets the init objects
		@param coin_id (int): Assigned in constructor
		'''
		data = requests.get(self._base_url+'ticker/'+str(coin_id)).json()
		self._symbol = data["data"]['symbol']
		self._website_slug = data['data']['website_slug']
		self._name = data['data']['name']

	def quote(self):
		'''
		returns the current quote
		'''
		return requests.get(self._base_url+'ticker/'+str(self._id)+'/').json()["data"]["quotes"]['USD']['price']

	def historical_data(self,start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'), end_date = datetime.now().strftime('%Y%m%d'),data_type = 2):
		'''
			Pulls historical data of a cryptocurrency between two dates provided. Coinmarketcap
			does not have historical data api, so an html parser was used in order
			to pull the data and return it as a ordered_dict.
			@param start_date (string): The begining date of the quotes. format: "YYYYmmdd", Accepted values: Dates>20130428
			@param end_date (string): The end date of when to stop pulling quotes. format: "YYYYmmdd" 
		'''

		url = 'https://coinmarketcap.com/currencies/'+self._website_slug+'/historical-data/?start='+start_date+'&end='+end_date
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
				date = str(dateparser.parse(value)).split(" ")[0]
				candle_data[date] = []
				count+=1
			elif count == len(headings)-1:
				#the final column gets assigned and gets put into the json dictionary
				data[headings[count]] = int(value.replace(",",""))
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

		self.candles =  pd.concat(frames)
		return self.candles

	def sma(self,time_period = 10, series_type = "close"):
		'''
		Queries the Simple Moving Average
		'''
		return self.ta.sma(time_period,series_type)

	def ema(self,time_period = 10, series_type = "close"):
		'''
		Queries the Exponential Moving Average
		'''
		return self.ta.ema(time_period,series_type)

	def macd(self,fast_period = 12,slow_period = 26,signal_period = 9, series_type = "close"):
		'''
		Queries the Moving Average Convergence/Divergence
		'''
		return self.ta.macd(fast_period,slow_period,signal_period,series_type)

	def rsi(self,time_period = 14, series_type = "close"):
		'''
		Queries the Relative Strength Index
		'''
		return self.ta.rsi(time_period,series_type)

	def cci(self,time_period = 20):
		'''
		Queries the Commodity Channel Index 
		'''
		return self.ta.cci(time_period)

	def bbands(self,time_period = 20, series_type = "close", nbdevup = 2, nbdevdn = 2):
		'''
		Queries the Bollinger Bands
		'''
		return self.ta.bbands(time_period,series_type,nbdevup,nbdevdn)

	def graph(self,indicators=[]):
		#CURRENTLY WORKING ON GRAPHING THE DATA
		graph = Graph(self.graph_builder())
		graph.test()

class Technical_Analysis:
	'''
	This is holds code for Technical Analasis on prices.
	The class needs to first take in a pandas dataframe consisting of closing,open,high,low prices.
	'''
	def __init__(self,df):
		self.candles = df

	def check_intervals(self,time_period):
		'''
		Checks to make sure there are enough candle stick data for the time period
		@param time_period (int): amount of data points one desires to check
		'''
		if time_period <= len(self.candles.index):
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
			value = self.candles[series_type].rolling(window=time_period).mean()
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
			sma = self.candles[series_type].rolling(window=time_period,min_periods=time_period).mean()[:time_period]
			rest = self.candles[series_type][time_period:]
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
		#not finished, still trying to figure out how to calculate rsi properly with pandas
		if self.check_intervals(time_period):
			value = self.candles[series_type].diff().dropna()
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
			tp = (self.candles['high'] + self.candles['low']+self.candles['close'])/3
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
			stdev = self.candles[series_type].rolling(window=time_period).std().dropna()
			middle_band = self.candles[series_type].rolling(window=time_period).mean().dropna()
			upper_band = middle_band+(stdev*nbdevup)
			lower_band = middle_band-(stdev*nbdevup)
			join = pd.concat([middle_band,upper_band,lower_band],axis=1,sort=True)
			return pd.DataFrame(join.values,index=join.index,columns=['middle_band',"upper_band","lower_band"])
		else:
			return "Not enough data points, try to get more historical data" 

class Graph:
	def __init__(self,df):
		self.df = df

	def test(self):
		ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
		ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1)

		ax1.plot(self.df.index,self.df['close'])
		#ax2.bar(self.df.index,self.df['volume'])
		plt.show()
