import Markets as mk
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import *
import pandas as pd
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import math
from multiprocessing import Pool, freeze_support
import os
import numpy as np
import calendar

class Portfolio:
    def __init__(self,strategy="momentum win/lose",start=(datetime.now() - relativedelta(years=1)) - relativedelta(days=1)):
        self.strategies = ["momentum win/lose"]
        self.markets = get_nasdaq_symbols().index
        self.portfolio_strategy = self.check_strategy(strategy)
        if self.portfolio_strategy == False:
            raise ValueError("No such strategy exists")
        self.portfolio = None
        self.start = self.date_adjuster(start)
        self.end = None

    def check_strategy(self,strategy):
        if strategy in self.strategies:
            return strategy
        else:
            return False
    
    def date_adjuster(self,date):
        date_dict = {5:2,6:1} #this is saturday and sunday and how many days we need to move forward for monday

        if date.weekday() in date_dict.keys():
            return date + relativedelta(days=date_dict[date.weekday()])
        else:
            return date
    
    def date_finder(self,date,delta,day_type):
        if delta > 0 :
            date_dict = {
                        "days":date + relativedelta(days=delta),
                        "months":date + relativedelta(months=delta),
                        "years":date + relativedelta(years=delta)
                    }
        else:
            delta = delta * -1
            date_dict = {
                        "days":date - relativedelta(days=delta),
                        "months":date - relativedelta(months=delta),
                        "years":date - relativedelta(years=delta)
                    }
        return date_dict[day_type]
    
    def momen_calc(self,stock):
        try:
            start_date = ((datetime.now() - relativedelta(years=5)) + relativedelta(days=+1)).strftime('%Y%m%d') #pulls all 5 years of data
            j_dates = {
                    "j3":self.date_adjuster(self.date_finder(self.start,3,"months")).strftime('%Y-%m-%d'),
                    "j6":self.date_adjuster(self.date_finder(self.start,6,"months")).strftime('%Y-%m-%d'),
                    "j9":self.date_adjuster(self.date_finder(self.start,9,"months")).strftime('%Y-%m-%d'),
                    "j12":self.date_adjuster(self.date_finder(self.start,12,"months")).strftime('%Y-%m-%d'),
                    }
            
            current_asset = mk.Asset(stock,start=start_date)
            n1 = current_asset.candles.close.loc[self.end.strftime('%Y-%m-%d')]
            n2 = current_asset.candles.close.loc[self.start.strftime('%Y-%m-%d')]
            change = ((n2-n1)/n1) * 100
            j3 = ((current_asset.candles.close.loc[j_dates["j3"]] - n2)/n2) * 100
            j6 = ((current_asset.candles.close.loc[j_dates["j6"]] - n2)/n2) * 100
            j9 = ((current_asset.candles.close.loc[j_dates["j9"]] - n2)/n2) * 100
            j12 = ((current_asset.candles.close.loc[j_dates["j12"]] - n2)/n2) * 100
            
            return [stock,change,None,j3,j6,j9,j12]
        except:
            return [stock,float('nan'),0,0,0,0,None]
    
    def momen_win_loss(self,k=6):
        self.end = self.date_adjuster(self.date_finder(self.start,k*-1,"months"))
        #stocks = ["aapl","NVDA","amzn","amd","goog","F","INTC","IBM","FB","DIS"]
        stocks = self.markets[:200]
        start_time = datetime.now()
        
        pool = Pool(os.cpu_count())
        
        results = pool.map(self.momen_calc,stocks)
        pool.close()
        pool.join()
        
        temp_portfolio = pd.DataFrame(results,columns = ["stock","change","position","j3","j6","j9","j12"]).dropna(subset=["change"])
        temp_portfolio = temp_portfolio.sort_values(by=['change'],ascending=False)
        temp_portfolio = temp_portfolio.set_index("stock")
        
        cut_off = math.floor(len(temp_portfolio)*.1)

        self.portfolio = pd.concat([temp_portfolio[:cut_off],temp_portfolio[-cut_off:]])
   
        end_time = datetime.now()
        print(end_time-start_time)
        return self.portfolio
def main():
    a = Portfolio("momentum win/lose")
    print(a.momen_win_loss())


if __name__ == '__main__':
	freeze_support()
	main()