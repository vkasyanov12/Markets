import json
import requests
from urllib.request import urlopen
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
import randomcolor

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
			candle_data = {}
	
			for heading in headings:#initial assignment of the candle_data
				candle_data[heading] = []

			count = 0 #this is used to determine which column is being used
			date = "" #holds the current date

			#this gets all the rows from the table. It is one constant stream of data
			for td in history.find_all("td"):
				value = td.text.strip()
				if count == 0:
					date = str(dateparser.parse(value)).split(" ")[0] #gets the date
					candle_data[headings[count]].append(date) 
					count+=1
				elif count == len(headings)-1:
					#last column before count reset
					candle_data[headings[count]].append(int(value.replace(",","").replace("-","0")))
					count = 0 #this will reset to 0 when we have reached the final column
				else:
					candle_data[headings[count]].append(float(value.replace(",","")))
					count+=1
			
			df = pd.DataFrame(data=candle_data)
			df.set_index(headings[0],inplace=True)
			
			return df.reindex(index=df.index[::-1])
		except:
			return "No Such Asset Exists"
	
	def get_daily_change(self,time_periods=1,comb=True):
		df = self.candles.close.pct_change(periods=time_periods)
		return self.data_compliance(pd.DataFrame(df.values,index=df.index,columns=['% Change']),comb)

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

	def sma(self,time_period = 7, series_type = "close", comb = True):
		'''
		Queries Simple Moving Average
		'''
		return self.data_compliance(self.ta.sma(time_period,series_type),comb)

	def ema(self,time_period = 7, series_type = "close",comb = True):
		'''
		Queries Exponential Moving Average
		'''
		return self.data_compliance(self.ta.ema(time_period,series_type),comb)

	def macd(self,fast_period = 12,slow_period = 26,signal_period = 9, series_type = "close",comb = True):
		'''
		Queries Moving Average Convergence/Divergence
		'''
		return self.data_compliance(self.ta.macd(fast_period,slow_period,signal_period,series_type),comb)

	def rsi(self,time_period = 14, series_type = "close",comb = True):
		'''
		Queries the Relative Strength Index
		'''
		return self.data_compliance(self.ta.rsi(time_period,series_type),comb)

	def cci(self,time_period = 20,comb = True):
		'''
		Queries the Commodity Channel Index 
		'''
		return self.data_compliance(self.ta.cci(time_period),comb)

	def bbands(self,time_period = 20, series_type = "close", nbdevup = 2, nbdevdn = 2, comb = True):
		'''
		Queries the Bollinger Bands
		'''
		return self.data_compliance(self.ta.bbands(time_period,series_type,nbdevup,nbdevdn),comb)

	def obv(self,comb=True):
		'''
		Queries the On-Balance Volume
		'''
		return self.data_compliance(self.ta.obv(),comb)

	def data_compliance(self,data,combine):
		'''
		Combines data frames together with the candle dataframes
		@param data: A dataFrame.
		@param combine: Boolean value to indicate whether or not to combine the data
		'''
		if combine:
			self.candles = pd.concat([self.candles,data],axis=1,sort=True)
			return self.candles
		else:
			return data

	def graph(self,indicators=[]):
		graph = Graph(self.candles,self.asset)
		graph.plot()

	def export_to_csv(self,path=None,name=None):
		if name==None:
			name=self.name = self.asset+'_'+self.candles.index[0]+'_'+self.candles.index[-1]
		
		if path==None:
			self.candles.to_csv(name+".csv")
		else:
			location = path+"\\"+name+".csv"
			self.candles.to_csv(location)

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

			return pd.DataFrame(join.values,index=join.index,columns=['macd_line',"macd_signal_line","macd_histogram"])
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
			return pd.DataFrame(join.values,index=join.index,columns=['bband_middle_band_'+str(time_period),'bband_upper_band_'+str(time_period),'bband_lower_band_'+str(time_period)])
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
	'''
	Graph object that is directly associated with the Asset object
	Ideas used were pulled from
	https://plot.ly/~jackp/17421/plotly-candlestick-chart-in-python/#/
	'''
	def __init__(self,df,asset):
		self.df = df
		self.asset = asset
		self.name = self.asset+'_'+self.df.index[0]+'_'+self.df.index[-1]
		self.fig = None
		self.layout_count = 0 #Some graphs need a seperate yaxis, this keeps track of it
		self.colors_used = []#this keeps track of the colors used for moving averages 
		self.build()

	def build(self):	
		self.build_candles()
		
		indicator_dict = {
			"volume":self.build_volume_bars,
			"sma":self.build_sma,
			"ema":self.build_ema,
			"bband":self.build_bband,
			"cci":self.build_cci,
			"rsi":self.build_rsi,
			"macd":self.build_macd,
			"obv":self.build_obv
		}

		for indicator in self.df.columns.values:
			split_indic = indicator.split("_")
			if split_indic[0] in indicator_dict.keys():
				indicator_dict[split_indic[0]](self.df[indicator])
		
		self.build_layout()

	def build_candles(self):
		data = [dict(
				type = 'candlestick',
			    open = self.df.open,
			    high = self.df.high,
			    low = self.df.low,
			    close = self.df.close,
			    x = self.df.index,
			    yaxis = 'y',
			    name = self.asset,
			    increasing = dict( line = dict(color = '#2f5933')),
			    decreasing = dict( line = dict( color = '#8c0f0f')))]

		layout = dict()
		self.fig = dict(data=data,layout=layout)
		self.layout_count+=1


	def build_layout(self):
		self.fig['layout'] = dict()
		self.fig['layout']['plot_bgcolor'] = 'rgb(191, 191, 191)'
		self.fig['layout']['xaxis'] = dict(rangeslider=dict(visible=False))
		self.fig['layout']['legend'] = dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' )
		self.fig['layout']['margin'] = dict( t=0, b=0, r=35, l=35 )
		
		if self.layout_count==1: #this checks if there is only candle data
			self.fig['layout']['yaxis'] = dict( domain = [0, 1])
		else:
			candle_start = self.layout_count*.10 #y-cordinate of starting candle chart
			current_start = candle_start - .12 #for the y-axis begining spot 
			current_end = candle_start - .02 #for the next y-axis end point

			self.fig['layout']['yaxis'] = dict( domain = [candle_start, 1]) #creates candle y-axis

			for axis in range(1,self.layout_count): 
				self.fig['layout']['yaxis'+str(axis+1)] = dict( domain = [current_start,current_end])
				current_start = current_start - .12 #adjusts the position for the next y-axis
				current_end = current_start + .1

	def build_volume_bars(self,volume):
		colors = []
		self.layout_count+=1
		for i in range(len(self.df.close)):
		    if i != 0:
		        if self.df.close[i] > self.df.close[i-1]:
		            colors.append('#2f5933')
		        else:
		            colors.append('#8c0f0f')
		    else:
		        colors.append('#8c0f0f')
		
		self.fig['data'].append(dict( x=volume.index, y=volume.values,                         
                         marker=dict( color=colors ),
                         type='bar', yaxis='y'+str(self.layout_count), name='volume'))

	def build_sma(self,sma):
		self.fig['data'].append( dict( x=sma.index, y=sma.values, type='scatter', mode='lines', 
                         line = dict( width = 1 ),
                         marker = dict( color = self.random_color() ),
                         yaxis = 'y', name=sma.name ) )

	def build_ema(self,ema):
		self.fig['data'].append( dict( x=ema.index, y=ema.values, type='scatter', mode='lines', 
                         line = dict( width = 1 ),
                         marker = dict( color = self.random_color() ),
                         yaxis = 'y', name=ema.name ) )

	def build_bband(self,bband):
		self.fig['data'].append( dict( x=bband.index, y=bband.values, type='scatter', yaxis='y', 
                         line = dict( width = 1 ),
                         marker=dict(color='#595959'), hoverinfo='none', 
                         legendgroup='Bollinger Bands', name=bband.name) )

	def build_cci(self,cci):
		self.layout_count+=1
		self.fig['data'].append( dict( x=cci.index, y=cci.values, type='scatter', mode='lines', 
                         line = dict( width = 1 ),
                         marker = dict( color = '#595959' ),
                         yaxis = 'y'+str(self.layout_count), name=cci.name ) )

	def build_rsi(self,rsi):
		self.layout_count+=1
		self.fig['data'].append( dict( x=rsi.index, y=rsi.values, type='scatter', mode='lines', 
                         line = dict( width = 1 ),
                         marker = dict( color = '#252770' ),
                         yaxis = 'y'+str(self.layout_count), name=rsi.name ) )

	def build_macd(self,macd):

		if macd.name=='macd_line':
			self.layout_count+=1
			self.fig['data'].append( dict( x=macd.index, y=macd.values, type='scatter', mode='lines', 
                         line = dict( width = 1 ),
                         marker = dict( color = '#000000' ),
                         yaxis = 'y'+str(self.layout_count), name=macd.name ))

		elif macd.name=='macd_signal_line':
			self.fig['data'].append( dict( x=macd.index, y=macd.values, type='scatter', mode='lines', 
                         line = dict( width = 1 ),
                         marker = dict( color = '#990000' ),
                         yaxis = 'y'+str(self.layout_count), name=macd.name ))
	
		if macd.name=='macd_histogram':

			positive = macd * 0
			negative = macd * 0
			final = macd * 0

			positive[macd>0] = '#2f5933'
			negative[macd<0] = '#8c0f0f'
			
			final[positive=='#2f5933'] = positive[positive=='#2f5933']
			final[negative=='#8c0f0f'] = negative[negative=='#8c0f0f']
	
			self.fig['data'].append(dict( x=macd.index, y=macd.values,                         
                         marker=dict( color=final.values ),
                         type='bar', yaxis='y'+str(self.layout_count), name=macd.name))

	def build_obv(self,obv):
		self.layout_count+=1
		self.fig['data'].append( dict( x=obv.index, y=obv.values, type='scatter', mode='lines', 
                         line = dict( width = 1 ),
                         marker = dict( color = '#252770' ),
                         yaxis = 'y'+str(self.layout_count), name=obv.name ) )
	
	def random_color(self):
		'''
		Generates random colors for moving averages
		'''
		random_color = randomcolor.RandomColor().generate()

		while random_color in self.colors_used:
			random_color = randomcolor.RandomColor().generate()
		self.colors_used.append(random_color)
		
		return random_color
	
	def plot(self):
		plt.offline.plot(self.fig, filename=self.name+'.html')