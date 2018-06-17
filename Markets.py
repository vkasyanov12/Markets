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
		@param candle_time (string): Canlestick length. Default "Daily", Accepted Values: "Intra","Daily","Weekly","Monthly"
		@param interval (int): When candle_time = "Intra", a time needs to be entered in minutes for each candle stick. Default value = 5, Accepted Values: 1,5,15,30,60
		'''
		self.__apikey = "&apikey=50EK1RZLJ0GSEQ4K"
		self._base_url = "https://www.alphavantage.co/query?function="
		self._ticker = ticker.upper()
		self._candle_time = candle_time.lower()
		self._interval = str(interval)
		self._time_series = { 
			"intra" : "TIME_SERIES_INTRADAY&symbol=",
			"daily" : "TIME_SERIES_DAILY&symbol=",
			"weekly" : "TIME_SERIES_WEEKLY&symbol=",
			"monthly" : "TIME_SERIES_MONTHLY&symbol="}
		self._indicators = {
			"SMA" : "SMA&symbol=",
			"EMA" : "EMA&symbol=",
			"MACD" : "MACD&symbol=",
			"STOCH" : "STOCH&symbol=",
			"RSI" : "RSI&symbol=",
			"CCI" : "CCI&symbol=",
			"BBANDS" : "BBANDS&symbol="}

	def quote(self):
		'''
		returns the current price,volume etc... of the stock
		'''
		url = self._base_url+"BATCH_STOCK_QUOTES&symbols="+self._ticker+self.__apikey
		return json.loads(requests.get(url).text,object_pairs_hook = OrderedDict)

	def historical_data(self):
		'''
		Queries the stock price based upon the candlestick time
		'''
		if self._candle_time == "intra":#The url needs to change slightly if it's intra day.
			url = self._base_url+self._time_series[self._candle_time]+self._ticker+"&interval="+self._interval+"min"+self.__apikey
		else:
			url = self._base_url+self._time_series[self._candle_time]+self._ticker+self.__apikey 
		return json.loads(requests.get(url).text,object_pairs_hook = OrderedDict)

	def indicator_builder(self,indicator):
		'''
		Builds the begining of the URL for specific indicators. 
		@param indicator String: Specific indicator to query. 
		'''
		if self._candle_time == "intra":#The url needs to change slightly if it's intra day.
			return self._base_url+self._indicators[indicator]+self._ticker+"&interval="+self._interval+"min"
		else:
			return self._base_url+self._indicators[indicator]+self._ticker+"&interval="+self._candle_time

	def sma(self,time_period = 10,series_type = "close"):
		'''
		Queries the Simple Moving Average on the stock.
		@param time_period (int): number of data points used to calculate each moving average value. Default = 10, Accepted Values: Positive values
		@param series_type (string): The price type in the time series. Default = "close". Accepted Values: "close","open","high","low"
		'''
		return json.loads(requests.get(self.indicator_builder("SMA")+"&time_period="+str(time_period)+"&series_type="+series_type.lower()+self.__apikey).text,object_pairs_hook = OrderedDict)

	def ema(self,time_period = 200,series_type = "close"):
		'''
		Queries the Exponential Moving Average on the stock.
		@param time_period (int): number of data points used to calculate each moving average value. Default = 200, Accepted Values: Positive values
		@param series_type (string): The price type in the time series. Default = "close". Accepted Values: "close","open","high","low"
		'''
		return json.loads(requests.get(self.indicator_builder("EMA")+"&time_period="+str(time_period)+"&series_type="+series_type.lower()+self.__apikey).text,object_pairs_hook = OrderedDict)

	def macd(self,fast_period = 12,slow_period = 26,signal_period = 9, series_type = "close"):
		'''
		Queries the Moving Average Convergence/Divergence on the stock
		@param fast_period (int). Default = 12, Accepted Values: Positive integers
		@param slow_period (int). Default = 26, Accepted Values: Positive integers
		@param signal_period (int). Default = 9, Accepted Values: Positive integers
		'''
		return json.loads(requests.get(self.indicator_builder("MACD")+"&series_type="+series_type+"&fastperiod="+
			   str(fast_period)+"&slowperiod="+str(slow_period)+"&signalperiod="+str(signal_period)+self.__apikey).text,object_pairs_hook = OrderedDict)

	def stoch(self,fastk_period = 5,slowk_period = 3,slowd_period = 3,slowk_matype = 0,slowd_matype = 0):
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
		'''
		return json.loads(requests.get(self.indicator_builder("STOCH")+"&fastkperiod="+str(fastk_period)+"&slowkperiod="+str(slowk_period)+"&slowdperiod="+
			   str(slowd_period)+"&slowkmatype="+str(slowk_matype)+"&slowdmatype="+str(slowd_matype)+self.__apikey).text,object_pairs_hook = OrderedDict)

	def rsi(self,time_period = 60,series_type = "close"):
		'''
		Queries the Relative Strength Index
		@param time_period (int): Number of data points used to calculate the rsi. Default = 60, Accepted Values: Positive integers
		@param series_type (string): Price type in the time series. Default = "close", Accepted Values: "close","open","high","low"
		'''
		return json.loads(requests.get(self.indicator_builder("RSI")+"&time_period="+str(time_period)+"&series_type="+series_type+self.__apikey).text,object_pairs_hook = OrderedDict)

	def cci(self,time_period = 60):
		'''
		Queries the Commodity Channel Index.
		@param time_period (int): Number of data points to calculate CCI. Default = 60, Accepted Values: Positive Integers
		'''
		return json.loads(requests.get(self.indicator_builder("CCI")+"&time_period="+str(time_period)+self.__apikey).text,object_pairs_hook = OrderedDict)

	def bbands(self,time_period = 60,series_type = "close",nbdevup = 2,nbdevdn = 2,matype = 0):
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
		'''
		return json.loads(requests.get(self.indicator_builder("BBANDS")+"&time_period="+str(time_period)+"&series_type="+
			   series_type+"&nbdevup="+str(nbdevup)+"&nbdevdn="+str(nbdevdn)+"&matype="+str(matype)+self.__apikey).text,object_pairs_hook = OrderedDict)
class Coin:
	'''
	This is an api for coinmarketcap.com. It pulls current and historical data from the website
	and returns values in json format Currently, historical data only supports daily information.
	'''
	def __init__(self,coin_id):
		'''
		inititializes the object
		@param coin_id (int): cryptocurrency id from coinmarketcap.com. This reliant on Market class. Accepted Values: Positive integers
		'''
		self._base_url = 'https://api.coinmarketcap.com/v2/'
		self._id = coin_id
		self._symbol = None
		self._website_slug = None 
		self._name = None
		self.get_info(coin_id)
		self.daily_candles = self.get_historical_data()

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

	def get_historical_data(self,start_date = (datetime.datetime.now() - timedelta(days=30)).strftime('%Y%m%d'), end_date = datetime.datetime.now().strftime('%Y%m%d')):
		'''
			Pulls historical data of a cryptocurrency between two dates provided. Coinmarketcap
			does not have historical data api, so an html parser was used in order
			to pull the data and return it as a json value.
			@param start_date (string): The begining date of the quotes. format: "YYYYmmdd", Accepted values: Dates greater than 20130428
			@param end_date (string): The end date of when to stop pulling quotes. format: "YYYYmmdd" Accepted values: Dates before current date
		'''

		url = 'https://coinmarketcap.com/currencies/'+self._website_slug+'/historical-data/?start='+start_date+'&end='+end_date
		raw_html = urllib2.urlopen(url)
		soup = BeautifulSoup(raw_html.read(),"html.parser")

		history = soup.find("table",attrs = {"table"})
		headings = [th.get_text().replace("*","").lower() for th in history.find("tr").find_all("th")]

		json_data = collections.OrderedDict()
		data = {}
		count = 0 #this is used to determine which column is being used
		date = ""
		#this gets all the rows from the table. It is one constant stream of data
		for td in history.find_all("td"):
			if count == 0:
				#Creates the date as the key
				date = str(dateparser.parse(td.text.strip())).split(" ")[0]
				json_data[date] = ""
				count+=1
			elif count == len(headings)-1:
				#the final column gets assigned and gets put into the json dictionary
				data[headings[count]] = td.text.strip()
				json_data[date] = data
				data = {}
				count = 0
			else:
				data[headings[count]] = td.text.strip()
				count+=1

		self.daily_candles = json_data     
		return json_data

	def check_intervals(self,time_period):
		'''
		Checks to make sure there are enough candle stick data for the time period
		@param time_period (int): amount of data points one desires to check
		'''
		if time_period <= len(self.daily_candles.keys()):
			return True
		else:
			return False

	def sma(self,time_period = 10, series_type = "close"):
		'''
		queris the Simple Moving Average on the coin
		@param time_period (int): amount of data points one desires to use. Default = 10, Accepted Values: Positive Integers
		@param series_type (string): The price type in the time series. Default = "close". Accepted Values: "close","open","high","low"
		'''
		sma_dict = collections.OrderedDict()
		candle_list = self.daily_candles.items() #creates ordered dict as a tuple [tuple index][0 = date, 1 = dictionary][dictionary key]
		sma_sum = [] 

		if self.check_intervals(time_period):
			for i in range(len(candle_list)-1,-1,-1): #need to start from the end of the list because the oldest values are on the end of the list 
				if len(sma_sum) < time_period: 
					sma_sum.append(float(candle_list[i][1][series_type.lower()])) 
				
				if len(sma_sum) == time_period:
					sma_dict[candle_list[i][0]] = sum(sma_sum) / time_period
					sma_sum.pop(0)
	
		return sma_dict

	def ema(self,time_period = 10, series_type = "close"):
		'''
		Queries the Exponential Moving Average on the Coin.
		@param time_period (int): number of data points used to calculate each moving average value. Default = 200, Accepted Values: Positive values
		@param series_type (string): The price type in the time series. Default = "close". Accepted Values: "close","open","high","low"
		'''
		ema_dict = collections.OrderedDict()
		candle_list = self.daily_candles.items() #creates ordered dict as a tuple [tuple index][0 = date, 1 = dictionary][dictionary key]
		init_sum = [] #used to calculate the initial begining sum for ema, first time_period sma
		multiplier = (2.0 / (time_period + 1)) 
		ema_switch = False #after calculating init_sum, turns switches into ema

		if self.check_intervals(time_period):
			for i in range(len(candle_list)-1,-1,-1): #need to start from the end of the list because the oldest values are on the end of the list 
				if len(init_sum) < time_period: 
					init_sum.append(float(candle_list[i][1][series_type.lower()]))

				elif (len(init_sum) == time_period and ema_switch == False): #once the intital value is finished calculating
					ema_dict[candle_list[i][0]] = sum(init_sum) / time_period
					ema_switch = True

				else:
					current_value = float(candle_list[i][1][series_type.lower()])
					previous_day = ema_dict[candle_list[i+1][0]] #because the list is backwards, the previous value is after current value
					ema_dict[candle_list[i][0]] = ((current_value - previous_day) * multiplier) + previous_day
					
		return ema_dict
