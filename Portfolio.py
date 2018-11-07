import Markets as mk
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import *
import pandas as pd
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import math

class Portfolio:

    def __init__(self,strategy):
        self.strategies = ["momentum win/lose"]
        self.markets = get_nasdaq_symbols().index
        self.portfolio_strategy = self.check_strategy(strategy)
        if self.portfolio_strategy == False:
            raise ValueError("No such strategy exists")
        self.portfolio = None

    def check_strategy(self,strategy):
        if strategy in self.strategies:
            return strategy
        else:
            return False

    def momen_win_loss(self,time_period=6):
        start = (datetime.now() - relativedelta(years=5)) + relativedelta(days=+1) #date starts 4 years and 364 days 
        end = start + relativedelta(months=+time_period)
        portfolio = pd.DataFrame(columns = ["stock","change","position"])
        stocks = ['aapl',"nvda","amzn","AMD","INTC","FB","TWTR","IBM","GOOGL","JPM"]
        
        start_time = datetime.now()
        for stock in stocks:
            try:
                current_asset = mk.Asset(stock,start=start.strftime('%Y%m%d'),end=end.strftime('%Y%m%d'))
                start_value = current_asset.candles.close.loc[start.strftime('%Y-%m-%d')] #not all stocks go back up to 5 years
                end_value = current_asset.candles.close.loc[end.strftime('%Y-%m-%d')]
                portfolio.loc[len(portfolio)+1] = [stock,((end_value-start_value)/start_value) * 100,None]
            except:
                pass
        end_time = datetime.now()
        print(end_time-start_time)
        portfolio = portfolio.sort_values(by=['change'],ascending=False)
        cut_off = math.floor(len(portfolio)*.1)
        
        portfolio[:cut_off].position = "Long" #assigns the long position in the top 10%
        portfolio[-cut_off:].position = "Short" #assigns short position in the bottom 10%
        self.portfolio = portfolio.copy()
        return self.portfolio
        
def main():
    a = Portfolio("momentum win/lose")
    print("done")
    print(a.momen_win_loss())

main()