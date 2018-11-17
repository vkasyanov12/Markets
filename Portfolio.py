import Markets as mk
from datetime import datetime
from dateutil.relativedelta import *
import pandas as pd
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import math
from multiprocessing import Pool, freeze_support
import os

class Portfolio:
    '''
    Portfolio class that creates a portfolio based upon the strategy assigned
    @param strategy (string):Current strategy
    @param start (datetime):begining date of portfolio
    '''
    def __init__(self,strategy="momentum win/lose",start=(datetime.now() - relativedelta(years=1)) - relativedelta(days=1)):
        self.strategies = ["momentum win/lose"]
        self.markets = get_nasdaq_symbols().index #holds all nasdaq traded stocks
        self.portfolio_strategy = self.check_strategy(strategy) 
        if self.portfolio_strategy == False:
            raise ValueError("No such strategy exists")
        self.portfolio = None
        self.start = self.date_adjuster(start) #makes sure the date provided is valid

    def check_strategy(self,strategy):
        '''
        Checks to make sure that this strategy exists
        @param strategy(string): strategy to check for support
        '''
        if strategy in self.strategies:
            return strategy
        else:
            return False
    
    def date_adjuster(self,date):
        '''
        This adjusts the date to make sure it doesn't land on the weekends
        '''
        date_dict = {5:2,6:1} #this is saturday and sunday and how many days we need to move forward for monday

        if date.weekday() in date_dict.keys():
            return date + relativedelta(days=date_dict[date.weekday()])
        else:
            return date
    
    def date_finder(self,date,delta,day_type):
        '''
        Finds the date from the current passed in.
        @param date(datetime): starting date
        @param delta(int): the difference from the current date
        @param day_type(string): different date types to skip over. Accepted Values:days,months,years 
        
        '''
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
    
    def momen_calc(self,parameters):
        '''
        The multiprocessing function that is associated with momentum_win_loss
        @param parameters (list):Holds all the parameters for each thread
        parameter[0] (int): starting index position
        parameter[1] (int): ending index position
        parameter[2] (string): start date where to begin to pull stock data: usually up to 5 years
        parameter[3] (string): Look back date from the self.start date
        parameter[4] (string): j3 is looking 3 months ahead
        parameter[5] (string): j6 is looking 6 months ahead
        parameter[6] (string): j9 is looking 9 months ahead
        parameter[7] (string): j12 is looking 12 months ahead
        '''
        
        results = [] #holds the dataframes

        for index in range(parameters[0],parameters[1]):
            try:
                current_asset = mk.Asset(self.markets[index],start=parameters[2]) #pulls stock data
                n1 = current_asset.candles.close.loc[parameters[3]] #finds the date of k date
                n2 = current_asset.candles.close.loc[self.start.strftime('%Y-%m-%d')]
                t_return = ((n2-n1)/n1)
                j3 = ((current_asset.candles.close.loc[parameters[4]] - n2)/n2) 
                j6 = ((current_asset.candles.close.loc[parameters[5]] - n2)/n2) 
                j9 = ((current_asset.candles.close.loc[parameters[6]] - n2)/n2) 
                j12 = ((current_asset.candles.close.loc[parameters[7]] - n2)/n2)
                results.append([self.markets[index],self.start.strftime('%Y-%m-%d'),t_return,None,j3,j6,j9,j12])
            except:
                results.append([self.markets[index],0,float('nan'),0,0,0,0,None])
        
        return pd.DataFrame(results,columns = ["stock","p_date","return","position","j3","j6","j9","j12"]).dropna(subset=["return"])
    
    def momen_win_loss(self,k=6):
        '''
        Generates a portfolio based on momentum strategy
        @param k (int): look back period.
        '''        
        end_date = self.date_adjuster(self.date_finder(self.start,k*-1,"months")).strftime('%Y-%m-%d') #finds the k date
        start_date = ((datetime.now() - relativedelta(years=5)) + relativedelta(days=+1)).strftime('%Y%m%d') #pulls 5 years of data
        cpu_count = os.cpu_count()
        j_dates = { #gets the j dates
                "j3":self.date_adjuster(self.date_finder(self.start,3,"months")).strftime('%Y-%m-%d'),
                "j6":self.date_adjuster(self.date_finder(self.start,6,"months")).strftime('%Y-%m-%d'),
                "j9":self.date_adjuster(self.date_finder(self.start,9,"months")).strftime('%Y-%m-%d'),
                "j12":self.date_adjuster(self.date_finder(self.start,12,"months")).strftime('%Y-%m-%d'),
                }
           
        work = [] #holds all the params
        split_work = math.floor((len(self.markets)/cpu_count)) #determines how many stocks each core will get 
        start_position = 0#used for distribution for cores
 
        for cpu in range(1,cpu_count+1):
            end_position = (split_work*cpu) - 1 #last index for the core
            
            if cpu == cpu_count: #this makes sure that the last core doesn't miss 1 stock
                end_position = len(self.markets)
            
            parameters = [start_position,end_position,start_date,end_date,j_dates["j3"],j_dates["j6"],j_dates["j9"],j_dates["j12"]]
            start_position = start_position + split_work #updates start position for the next stock
            work.append(parameters)
     
        pool = Pool(cpu_count)
        results = pool.map(self.momen_calc,work)
        pool.close()
        pool.join()
        
        temp_portfolio = pd.concat(results).sort_values(by=['return'],ascending=False).dropna(subset=["return"])
        temp_portfolio = temp_portfolio.set_index("stock")
        
        cut_off = math.floor(len(temp_portfolio)*.1)
        
        self.portfolio = pd.concat([temp_portfolio[:cut_off],temp_portfolio[-cut_off:]])
        self.portfolio.loc[:cut_off,"position"] = "buy"
        self.portfolio.loc[-cut_off:,"position"] = "sell"
        
        return self.portfolio
    
def main():
    a = Portfolio("momentum win/lose")
    print(a.momen_win_loss())


if __name__ == '__main__':
	freeze_support()
	main()