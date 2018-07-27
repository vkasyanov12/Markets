import json
import requests
from collections import OrderedDict
from bs4 import BeautifulSoup
import urllib2
import collections
import dateparser
import datetime
from datetime import timedelta
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas as pd
from pandas.io.json import json_normalize
import numpy
import matplotlib.pyplot as plt

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
		self.daily_candles = None
		self.get_info(coin_id)
		self.get_historical_data()

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
		return requests.get(self._base_url+'ticker/'+str(self._id)+'/').json()["data"]["quotes"]

	def get_historical_data(self,start_date = (datetime.datetime.now() - timedelta(days=30)).strftime('%Y%m%d'), end_date = datetime.datetime.now().strftime('%Y%m%d'),data_type = 2):
		'''
			Pulls historical data of a cryptocurrency between two dates provided. Coinmarketcap
			does not have historical data api, so an html parser was used in order
			to pull the data and return it as a ordered_dict.
			@param start_date (string): The begining date of the quotes. format: "YYYYmmdd", Accepted values: Dates greater than 20130428
			@param end_date (string): The end date of when to stop pulling quotes. format: "YYYYmmdd" Accepted values: Dates before current date
			@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe
		'''

		url = 'https://coinmarketcap.com/currencies/'+self._website_slug+'/historical-data/?start='+start_date+'&end='+end_date
		raw_html = urllib2.urlopen(url)
		soup = BeautifulSoup(raw_html.read(),"html.parser")

		history = soup.find("table",attrs = {"table"})
		headings = [th.get_text().replace("*","").replace(" ","").lower() for th in history.find("tr").find_all("th")]
		candle_data = OrderedDict()
		data = {}
		count = 0 #this is used to determine which column is being used
		date = ""
		#this gets all the rows from the table. It is one constant stream of data
		for td in history.find_all("td"):
			if count == 0:
				#Creates the date as the key
				date = str(dateparser.parse(td.text.strip())).split(" ")[0]
				candle_data[date] = []
				count+=1
			elif count == len(headings)-1:
				#the final column gets assigned and gets put into the json dictionary
				data[headings[count]] = td.text.strip()
				candle_data[date] = data
				data = {}
				count = 0
			else:
				data[headings[count]] = td.text.strip()
				count+=1
		
		candle_data = OrderedDict(reversed(list(candle_data.items()))) #reversed it in order to have the most recent items on the bottom
		self.daily_candles = candle_data

		return self.data_compliance(candle_data,data_type)

	def data_compliance(self,data,data_type):
		'''
		Returns the proper data type for the user based upon the type preferred.
		@param data (ordered_dict or dict): Data in form of a dictionary
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe, 2 = Ordered Python Dictionary
		'''
		if data_type == 0: #returns json
			return json.dumps(data)
		elif data_type == 1:
			frames = []
			for day in data.keys(): #builds the panda dataframe
				df = pd.DataFrame(data[day],index = [day])
				frames.append(df)
			return pd.concat(frames)
		else:
			return data
		
	def check_intervals(self,time_period):
		'''
		Checks to make sure there are enough candle stick data for the time period
		@param time_period (int): amount of data points one desires to check
		'''
		if time_period <= len(self.daily_candles.keys()):
			return True
		else:
			return False

	def sma(self,time_period = 10, series_type = "close",data_type = 2):
		'''
		Queries the Simple Moving Average on the Coin
		@param time_period (int): amount of data points one desires to use. Default = 10, Accepted Values: Positive Integers
		@param series_type (string): The price type in the time series. Default = "close". Accepted Values: "close","open","high","low"
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe, 2 = Ordered Python Dictionary
		'''
		sma_dict = OrderedDict()
		sma_sum = [] 

		if self.check_intervals(time_period):
			for day in range(len(self.daily_candles.keys())):
				if len(sma_sum) < time_period:
					sma_sum.append(float(self.daily_candles.values()[day][series_type]))

				if len(sma_sum) == time_period:
					sma_dict[self.daily_candles.keys()[day]] = {"sma":numpy.average(sma_sum)}
					sma_sum.pop(0)

			return self.data_compliance(sma_dict,data_type)
		else:
			return "Not enough data points"

	def ema(self,time_period = 10, series_type = "close",data_type = 2):
		'''
		Queries the Exponential Moving Average on the Coin.
		@param time_period (int): number of data points used to calculate each moving average value. Default = 10, Accepted Values: Positive values
		@param series_type (string): The price type in the time series. Default = "close". Accepted Values: "close","open","high","low"
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe, 2 = Ordered Python Dictionary
		'''
		ema_dict = OrderedDict()
		init_sum = [] #used to calculate the initial begining sum for ema, first time_period sma
		multiplier = (2.0 / (time_period + 1)) 

		if self.check_intervals(time_period):#checks if there is enough intervals for calculation		
			for day in range(time_period): #initializes the begining value for the ema
				init_sum.append(float(self.daily_candles.values()[day][series_type]))

			ema_dict[self.daily_candles.keys()[time_period-1]] = {"ema":numpy.average(init_sum)}
			for day in range(time_period,len(self.daily_candles.keys())):
				current_value = float(self.daily_candles.values()[day][series_type])
				previous_day = ema_dict[self.daily_candles.keys()[day-1]]["ema"]
				ema_dict[self.daily_candles.keys()[day]] = {"ema":((current_value - previous_day) * multiplier) + previous_day}

			return self.data_compliance(ema_dict,data_type)
		else:
			return "Not enough data points"

	def macd(self,fast_period = 12,slow_period = 26,signal_period = 9, series_type = "close",data_type=2):
		'''
		Queries the Moving Average Convergence/Divergence on the Coin
		@param fast_period (int). Default = 12, Accepted Values: Positive integers
		@param slow_period (int). Default = 26, Accepted Values: Positive integers
		@param signal_period (int). Default = 9, Accepted Values: Positive integers
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe, 2 = Ordered Python Dictionary
		'''
		if self.check_intervals(slow_period+signal_period):#checks to make sure there is enough data points
	
			fast_period_data = self.ema(fast_period, series_type)
			slow_period_data = self.ema(slow_period, series_type)
			macd = OrderedDict() #holds macd_line,signal_line,macd_histogram
			macd_line = OrderedDict()
			signal_line = OrderedDict()
			macd_histogram = OrderedDict()

			for key in slow_period_data.keys(): #calculates macd_line
				macd_line[key] = fast_period_data[key]["ema"] - slow_period_data[key]["ema"]

			init_sum = []#used for signal_line sum		
			for i in range(signal_period):#initializes the signal_line data
				init_sum.append(macd_line.values()[i])
			
			signal_line[macd_line.keys()[signal_period-1]] = numpy.average(init_sum)
			multiplier = multiplier = (2.0 / (signal_period + 1)) 

			for day in range(signal_period,len(macd_line.keys())):#calculates signal_line
				current_value = macd_line.values()[day]
				previous_day = signal_line[macd_line.keys()[day-1]]
				signal_line[macd_line.keys()[day]] = ((current_value - previous_day) * multiplier) + previous_day
				macd_histogram[macd_line.keys()[day]] = macd_line[macd_line.keys()[day]] - signal_line[macd_line.keys()[day]]	
			
			for key in signal_line.keys(): #calculates macd_histogram and assigns all values to macd
				macd_histogram[key] = macd_line[key] - signal_line[key]
				macd[key] = {'macd_line':macd_line[key], 'signal_line':signal_line[key],'macd_histogram':macd_histogram[key]} 
			
			return self.data_compliance(macd,data_type)
		
		else:
			return "Not enough data points"

	def rsi(self,time_period = 14, series_type = "close",data_type = 2):
		'''
		Queries the Relative Strength Index
		@param time_period (int): Number of data points used to calculate the rsi. Default = 14, Accepted Values: Positive integers
		@param series_type (string): Price type in the time series. Default = "close", Accepted Values: "close","open","high","low"
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe, 2 = Ordered Python Dictionary
		'''
		gain = 0 #used to measure average gain
		loss = 0 #used to measure average loss
		rsi = OrderedDict() #holds the rsi by date

		if self.check_intervals(time_period): #checks to make sure there is enough data points
			for day in range(0,time_period): #need to calculate the initial rsi
				value = float(self.daily_candles.values()[day][series_type]) - float(self.daily_candles.values()[day-1][series_type])
				
				if value > 0:
					gain += value
				elif value < 0:
					loss += abs(value)
				
			gain = gain/time_period
			loss = loss/time_period

			rsi[self.daily_candles.keys()[time_period-1]] = {"rsi":100 - (100 / (1 +(gain/loss)))}#sets the first day rsi
			
			for day in range(time_period,len(self.daily_candles.keys())):
				value = float(self.daily_candles.values()[day][series_type]) - float(self.daily_candles.values()[day-1][series_type])

				if value > 0:
					gain = (gain * (time_period - 1) + value) / time_period
					loss = (loss * (time_period - 1)) / time_period
				elif value < 0:
					loss = (loss * (time_period - 1) + abs(value)) / time_period
					gain = (gain * (time_period - 1)) / time_period

				rsi[self.daily_candles.keys()[day]] = {"rsi":100 - (100 / (1 +(gain/loss)))}
			
			return self.data_compliance(rsi,data_type)

		else:
			return "Not enough data points"


	def stoch(self,slowk_period = 3, slowd_period = 3,time_period = 14, series_type = "close",data_type = 2):
		'''
		Quries the Stochastic Oscillator on the Coin
		@param slowk_period (int): Time period of the slowk moving average. Default = 3, Accepted Values: Positive integers
		@param slowd_period (int): Time period of the slowd moving average. Default = 3, Accepted Values: Positive integers
		@param time_period (int): Number of data points used to calculate the rsi. Default = 14, Accepted Values: Positive integers
		@param series_type (string): Price type in the time series for the rsi. Default = "close", Accepted Values: "close","open","high","low"
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe, 2 = Ordered Python Dictionary

		NOTE:THE CALCULATIONS SEEM TO BE OFF, CHECK LATER WHEN GRAPH IS IMPLEMENTED
		'''
		stoch = OrderedDict() #holds the stoch data
		initial_data = self.rsi(time_period,series_type) #initial rsi data required in order to calculate stoch
		
		rsi_list = [] #holds the time_period average
		fastk = [] #holds fastk calculations
		slowk = [] #holds the slowk calculations
		
		if self.check_intervals(time_period+slowk_period+slowd_period): #checks to see if there is enough data points
			#fastk requires the rsi, slowk requires fastk and slowd requires slowk.
			for day in initial_data.keys():
				if len(rsi_list) < time_period: #first create the list of rsi values to average depending on the time period
					rsi_list.append(initial_data[day]["rsi"]) 

				if len(rsi_list) == time_period:
					fastk_calc = ((initial_data[day]["rsi"] - min(rsi_list))/(max(rsi_list) - min(rsi_list))) * 100
					rsi_list.pop(0) #removes the first value for the next value to be appended
					
					if len(fastk) < slowk_period: #creates the fastk list to average later for the slowk
						fastk.append(fastk_calc)

					if len(fastk) == slowk_period: #calculates the slowk
						slowk_calc = sum(fastk) / slowk_period
						slowk.append(slowk_calc) #adds the slowk list in order to average the slowd later
						fastk.pop(0)

						if len(slowk) == slowd_period: #calculates the slowd
							slowd_calc = sum(slowk) / slowd_period
							stoch[day] = {"slowk_k" : slowk_calc, "slow_d" : slowd_calc}
							slowk.pop(0)
			
			return self.data_compliance(stoch,data_type)
		else:
			return "Not enough data points"

	def cci(self,time_period = 20,data_type = 2):
		'''
		Queries the Commodity Channel Index on the Coin.
		@param time_period (int): Number of data points to calculate CCI. Default = 20, Accepted Values: Positive Integers
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe, 2 = Ordered Python Dictionary
		'''
		constant = .015
		tp_sma = [] 
		cci = OrderedDict()
		mean_dev = 0

		if self.check_intervals(time_period):
			for day in self.daily_candles.keys():
				#tp = typical price
				tp = (float(self.daily_candles[day]["high"]) + float(self.daily_candles[day]["low"]) + float(self.daily_candles[day]["close"])) / 3

				if len(tp_sma) < time_period: #builds the simple moving average
					tp_sma.append(tp)

				if len(tp_sma) == time_period:
					tp_avg = numpy.average(tp_sma)
					
					for value in tp_sma: #calculates the mean deviation
						mean_dev += abs(tp_avg - value)
					
					mean_dev = mean_dev / time_period
					
					cci[day] = {"cci":(tp - tp_avg)/(constant * mean_dev)}
					tp_sma.pop(0) #need to remove the first value due to moving average
					mean_dev = 0

			return self.data_compliance(cci,data_type)
		else:
			return "Not enough data points"

	def bbands(self,time_period = 20, series_type = "close", nbdevup = 2, nbdevdn = 2,data_type = 2):
		'''
		Queries the Bollinger Bands on the coin.
		@param time_period (int): Number of data points used to calculate BBands. Default = 60. Accepted Values: Positive integers
		@param series_type (string): Desired price type in the time series. Default = "close". Accepted Values: "close","open","high","low"
		@param nbdevup (int): Standard deviation multiplier of the upper band. Default = 2. Accepted Values: Positive integers
		@param nbdevdn (int): Standard deviation multiplier of the lower band. Default = 2. Accepted Values: Positive integers
		@param data_type (int): Indicates data type to return as for all functions. Accepted Values: 0 = Json, 1 = Pandas Dataframe, 2 = Ordered Python Dictionary
		'''
		bbands = OrderedDict()
		sma_band = [] 
		bband_avg = 0 #used to calculate the middle_band,upper_band and lowe_band

		if self.check_intervals(time_period):
			for day in self.daily_candles.keys():
				if len(sma_band) < time_period:
					sma_band.append(float(self.daily_candles[day][series_type])) #adds to the list in order to sum up later

				if len(sma_band) == time_period:
					bband_avg = numpy.average(sma_band)  #calculates middle band
					bbands[day] = {"middle_band": bband_avg,
								   "upper_band":bband_avg+(numpy.std(sma_band)*nbdevup),  
								   "lower_band":bband_avg-(numpy.std(sma_band)*nbdevup)}

					sma_band.pop(0)

			return self.data_compliance(bbands,data_type)
		else:
			return "Not enough data points"

class Graph:
	def __init__(self):
		dunno = None

	def test(self):
		ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
		ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1)

		ax1.plot()
		plt.show()
