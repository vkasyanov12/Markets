import json
import requests
from urllib.request import urlopen
from collections import OrderedDict
from bs4 import BeautifulSoup
import collections
import dateparser
import datetime
from datetime import timedelta
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas as pd
from pandas.io.json import json_normalize
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

	def get_DJI(self):
		'''
		returns the Stock object of the Dow Jones Industrial Average
		'''
		return Stock('.DJI')

	def get_nasdaq_composite(self):
		'''
		returns the Stock object of the Nasdaq Composite
		'''
		return Stock('.IXIC')

	def get_SP500(self):
		'''
		returns the Stock object of the S&P500
		'''
		return Stock('.INX')

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
	https://www.alphavantage.co/documentation/ 
	'''
	def __init__(self,ticker,candle_time = "Daily",interval = 5):
		'''
		Initializes the Stock Object
		@param ticker (string): Ticker of the stock. 
		@param candle_time (string): Canlestick length. Default "Daily", Accepted Values: "Intra","Daily"
		@param interval (int): When candle_time = "Intra", a time needs to be entered in minutes for each candle stick. Default value = 5, Accepted Values: 1,5,15,30,60
		'''
		self.__apikey = "&apikey=50EK1RZLJ0GSEQ4K"
		self._base_url = "https://www.alphavantage.co/query?function="
		self._ticker = ticker.upper()
		self._candle_time = candle_time.lower()
		self._interval = str(interval)
		self._time_series = { 
			"intra" : "TIME_SERIES_INTRADAY&symbol=",
			"daily" : "TIME_SERIES_DAILY&symbol="}
			#"weekly" : "TIME_SERIES_WEEKLY&symbol=",
			#"monthly" : "TIME_SERIES_MONTHLY&symbol="}
		self._indicators = {
			"SMA" : "SMA&symbol=",
			"EMA" : "EMA&symbol=",
			"MACD" : "MACD&symbol=",
			"STOCH" : "STOCH&symbol=",
			"RSI" : "RSI&symbol=",
			"CCI" : "CCI&symbol=",
			"BBANDS" : "BBANDS&symbol="}

	def data_compliance(self,data,data_type):
		'''
		Returns the proper data type for the user based upon the type preferred.
		@param data (ordered_dict or dict): Data in form of a dictionary
		'''
		data = OrderedDict(reversed(list(data.items()))) #must first reverse the data where oldest comes first then recent

		if data_type == 0: #returns json
			return json.dumps(data)
		else:
			frames = []
			for day in data.keys(): #builds the panda dataframe
				df = pd.DataFrame(data[day],index = [day])
				frames.append(df)
			return pd.concat(frames)

	def quote(self):
		'''
		returns the current price,volume etc... of the stock
		'''
		url = self._base_url+"BATCH_STOCK_QUOTES&symbols="+self._ticker+self.__apikey
		return requests.get(url).json()["Stock Quotes"][0]['2. price']

	def historical_data(self,data_type = 0):
		'''
		Queries the stock price based upon the candlestick time
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe
		'''
		if self._candle_time == "intra":#The url needs to change slightly if it's intra day.
			url = self._base_url+self._time_series[self._candle_time]+self._ticker+"&interval="+self._interval+"min"+self.__apikey
			data = json.loads(requests.get(url).text,object_pairs_hook = OrderedDict)['Time Series ('+self._interval+'min)'] 
		else:
			url = self._base_url+self._time_series[self._candle_time]+self._ticker+self.__apikey 
			data = json.loads(requests.get(url).text,object_pairs_hook = OrderedDict)['Time Series ('+self._candle_time.capitalize()+')']

		return self.data_compliance(data,data_type)

	def indicator_builder(self,indicator):
		'''
		Builds the begining of the URL for specific indicators. 
		@param indicator String: Specific indicator to query.
		'''
		if self._candle_time == "intra":#The url needs to change slightly if it's intra day.
			return self._base_url+self._indicators[indicator]+self._ticker+"&interval="+self._interval+"min"
		else:
			return self._base_url+self._indicators[indicator]+self._ticker+"&interval="+self._candle_time

	def sma(self,time_period = 10,series_type = "close",data_type = 0):
		'''
		Queries the Simple Moving Average on the stock.
		@param time_period (int): number of data points used to calculate each moving average value. Default = 10, Accepted Values: Positive values
		@param series_type (string): The price type in the time series. Default = "close". Accepted Values: "close","open","high","low"
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe
		'''
		data = json.loads(requests.get(self.indicator_builder("SMA")+
							"&time_period="+str(time_period)+"&series_type="+
							series_type.lower()+self.__apikey).text,object_pairs_hook = OrderedDict)['Technical Analysis: SMA']
		
		return self.data_compliance(data,data_type)

	def ema(self,time_period = 200,series_type = "close",data_type = 0):
		'''
		Queries the Exponential Moving Average on the stock.
		@param time_period (int): number of data points used to calculate each moving average value. Default = 200, Accepted Values: Positive values
		@param series_type (string): The price type in the time series. Default = "close". Accepted Values: "close","open","high","low"
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe
		'''
		data = json.loads(requests.get(self.indicator_builder("EMA")+"&time_period="+
									   str(time_period)+"&series_type="+series_type.lower()+
									   self.__apikey).text,object_pairs_hook = OrderedDict)['Technical Analysis: EMA']
		
		return self.data_compliance(data,data_type)

	def macd(self,fast_period = 12,slow_period = 26,signal_period = 9, series_type = "close",data_type = 0):
		'''
		Queries the Moving Average Convergence/Divergence on the stock
		@param fast_period (int). Default = 12, Accepted Values: Positive integers
		@param slow_period (int). Default = 26, Accepted Values: Positive integers
		@param signal_period (int). Default = 9, Accepted Values: Positive integers
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe
		'''
		data = json.loads(requests.get(self.indicator_builder("MACD")+"&series_type="+series_type+"&fastperiod="+
			   						str(fast_period)+"&slowperiod="+str(slow_period)+"&signalperiod="+
			   						str(signal_period)+self.__apikey).text,object_pairs_hook = OrderedDict)['Technical Analysis: MACD']

		return self.data_compliance(data,data_type)

	def stoch(self,fastk_period = 5,slowk_period = 3,slowd_period = 3,slowk_matype = 0,slowd_matype = 0,data_type = 0):
		'''
		Quries the Stochastic Oscillator
		@param fastk_period (int): Time period of the fastk moving average. Default = 5, Accepted Values: Positive integers	
		@param slowk_period (int): Time period of the slowk moving average. Default = 3, Accepted Values: Positive integers
		@param slowd_period (int): Time period of the slowd moving average. Default = 3, Accepted Values: Positive integers
		@param slowk_matype (int): Moving average type for the slowk moving average. Default = 0, 
		Accepted Values: 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 
		3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 
		6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA).
		@param slowd_matype (int): Moving average type for the slowd moving average. Default = 0. 
		Accepted Values: 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 
		3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 
		5 = Triangular Moving Average (TRIMA), 6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA)
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe
		'''
		data = json.loads(requests.get(self.indicator_builder("STOCH")+"&fastkperiod="+str(fastk_period)+"&slowkperiod="+
										str(slowk_period)+"&slowdperiod="+str(slowd_period)+"&slowkmatype="+str(slowk_matype)+
										"&slowdmatype="+str(slowd_matype)+self.__apikey).text,object_pairs_hook = OrderedDict)['Technical Analysis: STOCH']
		
		return self.data_compliance(data,data_type)

	def rsi(self,time_period = 14,series_type = "close",data_type = 0):
		'''
		Queries the Relative Strength Index
		@param time_period (int): Number of data points used to calculate the rsi. Default = 14, Accepted Values: Positive integers
		@param series_type (string): Price type in the time series. Default = "close", Accepted Values: "close","open","high","low"
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe
		'''
		data = json.loads(requests.get(self.indicator_builder("RSI")+"&time_period="+str(time_period)+"&series_type="+
										series_type+self.__apikey).text,object_pairs_hook = OrderedDict)['Technical Analysis: RSI']

		return self.data_compliance(data,data_type)

	def cci(self,time_period = 60,data_type = 0):
		'''
		Queries the Commodity Channel Index.
		@param time_period (int): Number of data points to calculate CCI. Default = 60, Accepted Values: Positive Integers
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe
		'''
		data = json.loads(requests.get(self.indicator_builder("CCI")+"&time_period="+str(time_period)+
									   self.__apikey).text,object_pairs_hook = OrderedDict)['Technical Analysis: CCI']
		
		return self.data_compliance(data,data_type)

	def bbands(self,time_period = 60,series_type = "close",nbdevup = 2,nbdevdn = 2,matype = 0,data_type = 0):
		'''
		Queries the Bollinger Bands.
		@param time_period (int): Number of data points used to calculate BBands. Default = 60. Accepted Values: Positive integers
		@param series_type (string): Desired price type in the time series. Default = "close". Accepted Values: "close","open","high","low"
		@param nbdevup (int): Standard deviation multiplier of the upper band. Default = 2. Accepted Values: Positive integers
		@param nbdevdn (int): Standard deviation multiplier of the lower band. Default = 2. Accepted Values: Positive integers
		@param matype (int): Moving average type of the time series. Default = 0. 
		Accepted Values: 0 = Simple Moving Average (SMA), 1 = Exponential Moving Average (EMA), 2 = Weighted Moving Average (WMA), 
		3 = Double Exponential Moving Average (DEMA), 4 = Triple Exponential Moving Average (TEMA), 5 = Triangular Moving Average (TRIMA), 
		6 = T3 Moving Average, 7 = Kaufman Adaptive Moving Average (KAMA), 8 = MESA Adaptive Moving Average (MAMA)
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe
		'''
		data = json.loads(requests.get(self.indicator_builder("BBANDS")+"&time_period="+str(time_period)+"&series_type="+
			   							series_type+"&nbdevup="+str(nbdevup)+"&nbdevdn="+str(nbdevdn)+"&matype="+
			   							str(matype)+self.__apikey).text,object_pairs_hook = OrderedDict)['Technical Analysis: BBANDS']

		return self.data_compliance(data,data_type)

class Coin:
	'''
	This is an api for coinmarketcap.com. It pulls current and historical data from the website
	and returns values in json format Currently, historical data only supports daily information.
	'''
	def __init__(self,coin_id):
		'''
		inititializes the object
		@param coin_id (int): cryptocurrency id from coinmarketcap.com. This reliant on Market class. Accepted Values: Positive integers
		@@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe
		'''
		self._base_url = 'https://api.coinmarketcap.com/v2/'
		self._id = coin_id
		self._symbol = None
		self._website_slug = None 
		self._name = None
		self.candles = None
		self.get_info(coin_id)
		self.historical_data()
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

	def historical_data(self,start_date = (datetime.datetime.now() - timedelta(days=30)).strftime('%Y%m%d'), end_date = datetime.datetime.now().strftime('%Y%m%d'),data_type = 2):
		'''
			Pulls historical data of a cryptocurrency between two dates provided. Coinmarketcap
			does not have historical data api, so an html parser was used in order
			to pull the data and return it as a ordered_dict.
			@param start_date (string): The begining date of the quotes. format: "YYYYmmdd", Accepted values: Dates greater than 20130428
			@param end_date (string): The end date of when to stop pulling quotes. format: "YYYYmmdd" Accepted values: Dates before current date
			@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe
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
		return self.ta.sma(time_period,series_type)

	def ema(self,time_period = 10, series_type = "close"):
		return self.ta.ema(time_period,series_type)

	def macd(self,fast_period = 12,slow_period = 26,signal_period = 9, series_type = "close"):
		return self.ta.macd(fast_period,slow_period,signal_period,series_type)

	def rsi(self,time_period = 14, series_type = "close"):
		return self.ta.rsi(time_period,series_type)

	def cci(self,time_period = 20):
		return self.ta.cci(time_period)

	def bbands(self,time_period = 20, series_type = "close", nbdevup = 2, nbdevdn = 2):
		return self.ta.bbands(time_period,series_type,nbdevup,nbdevdn)

	def graph(self,indicators=[]):
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
		Queries the Simple Moving Average on the Coin
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
		@param time_period (int): Number of data points used to calculate the rsi. 
		@param series_type (string): Price type in the time series. 
		'''
		#not finished, still trying to figure out how to calculate rsi properly with pandas
		value = self.candles[series_type].diff().dropna()
		gain = value * 0
		loss = value * 0

		gain[value>0] = value[value>0]
		loss[value<0] = -value[value<0]
		
		gain[gain.index[time_period-1]] = np.mean(gain[:time_period])
		loss[loss.index[time_period-1]] = np.mean(loss[:time_period])

		gain = gain.drop(gain.index[:(time_period-1)])
		loss = loss.drop(loss.index[:(time_period-1)])
	
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
