import Markets as mk
from datetime import datetime
from dateutil.relativedelta import *
import pandas as pd
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import math
from multiprocessing import Pool, freeze_support
import os
from scipy.stats import ttest_ind

class Portfolio:
    '''
    Portfolio class that creates a portfolio based upon the strategy assigned
    @param strategy (string):Current strategy
    @param start (datetime):begining date of portfolio
    '''
    def __init__(self,strategy="momentum returns",start=(datetime.now() - relativedelta(years=1)) - relativedelta(days=1)):
        self.strategies = {"momen_returns":self.momen_returns,"momen_low_high":self.momen_low_high}
        self.markets = get_nasdaq_symbols().index[:250] #holds all nasdaq traded stocks
        self.portfolio_strategy = self.check_strategy(strategy) 
        if self.portfolio_strategy == False:
            raise ValueError("No such strategy exists")
        self.portfolio = None
        self.start = start #makes sure the date provided is valid
        self.purchase_dates = [] #used for testing strategies

    def check_strategy(self,strategy):
        '''
        Checks to make sure that this strategy exists
        @param strategy(string): strategy to check for support
        '''
        if strategy in self.strategies.keys():
            return strategy
        else:
            return False
    
    def date_adjuster(self,df,dates):
        '''
        Finds the next nearest index that is within 7 days
        '''
        final_dates = []
        for date in dates:
            if date.strftime('%Y-%m-%d') not in df.index:
                try:
                    value = datetime.strptime(df[df.index >= date.strftime('%Y-%m-%d')].index[0],"%Y-%m-%d")
                    delta = (value - date).days
                except:
                    pass
                try:
                    value2 = datetime.strptime(df[df.index <= date.strftime('%Y-%m-%d')].index[-1],"%Y-%m-%d")
                    delta2 = (date - value2).days
                except:
                    return False

                if delta <=7:
                    final_dates.append(value.strftime('%Y-%m-%d'))
                elif delta2 <=7:
                    final_dates.append(value2.strftime('%Y-%m-%d'))
                else:
                    return False

                delta = 8 #this was done for a quick hack, kind of reset of delta
                delta = 8  
            else:
                final_dates.append(date.strftime('%Y-%m-%d'))
     
        return final_dates

        
    
    def date_finder(self,date,delta,day_type):
        '''
        Finds the date from the current passed in.
        @param date(datetime): starting date
        @param delta(int): the difference from the current date. >0 = look foward, <0 look back
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
    
    def momen_returns(self,parameters):
        '''
        The multiprocessing function that is associated with momentum_win_loss
        This strategy revolves around buying the top % returners and selling bottom % returners losers
        The calculation that is used to determine this: (current_price - k_look_back) / current_price
        @param parameters (list):Holds all the parameters for each thread
        parameter[0] (int): starting index position
        parameter[1] (int): ending index position
        parameter[2] (string): start date where to begin to pull stock data: usually up to 5 years
        parameter[3] purchase date
        parameter[4] (string): Look back date from the self.start date
        parameter[5] (string): j3 is looking 3 months ahead
        parameter[6] (string): j6 is looking 6 months ahead
        parameter[7] (string): j9 is looking 9 months ahead
        parameter[8] (string): j12 is looking 12 months ahead
        '''
        
        results = [] #holds the dataframes

        for index in range(parameters[0],parameters[1]):
            try:
                current_asset = mk.Asset(self.markets[index],start=parameters[2]) #pulls stock data
                proper_dates = self.date_adjuster(current_asset.candles,parameters[3:])
     
                if proper_dates!=False:
                    purchase_date = current_asset.candles.close.loc[proper_dates[0]] #change the start date with date_changer
                    k_price = current_asset.candles.close.loc[proper_dates[1]] #finds the date of k date
                    t_return = ((purchase_date-k_price)/k_price)
                    j3 = ((current_asset.candles.close.loc[proper_dates[2]] - purchase_date)/purchase_date) 
                    j6 = ((current_asset.candles.close.loc[proper_dates[3]] - purchase_date)/purchase_date) 
                    j9 = ((current_asset.candles.close.loc[proper_dates[4]] - purchase_date)/purchase_date) 
                    j12 = ((current_asset.candles.close.loc[proper_dates[5]] - purchase_date)/purchase_date)
                    results.append([self.markets[index],proper_dates[0],t_return,None,j3,j6,j9,j12])
            except:
                pass

        return pd.DataFrame(results,columns = ["stock","p_date","return","position","j3","j6","j9","j12"]).dropna(subset=["return"])
                

    def momen_low_high(self,parameters):
        '''
        The multiprocessing function that is associated with momentum_win_loss
        This strategy revolves around buying the top performers and selling bottom performers.
        The calculation that is used to determine this: (start_price - k_low) / (k_high - k_low)
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
                current_asset = mk.Asset(self.markets[index],start=parameters[2])
                start_price = current_asset.candles.close.loc[self.start.strftime('%Y-%m-%d')]
                subset = current_asset.candles[(current_asset.candles.index >= parameters[3]) & (current_asset.candles.index <= self.start.strftime('%Y-%m-%d'))]
                high_price = subset.close.max()
                low_price = subset.close.min()
                calc = (start_price - low_price) / (high_price - low_price)
                j3 = ((current_asset.candles.close.loc[parameters[4]] - start_price)/start_price) 
                j6 = ((current_asset.candles.close.loc[parameters[5]] - start_price)/start_price) 
                j9 = ((current_asset.candles.close.loc[parameters[6]] - start_price)/start_price) 
                j12 = ((current_asset.candles.close.loc[parameters[7]] - start_price)/start_price)
                results.append([self.markets[index],self.start.strftime('%Y-%m-%d'),calc,None,j3,j6,j9,j12])
            except:
                results.append([self.markets[index],0,float('nan'),0,0,0,0,None])
 
        return pd.DataFrame(results,columns = ["stock","p_date","return","position","j3","j6","j9","j12"]).dropna(subset=["return"])


    def momen_win_loss(self,k=6):
        '''
        Generates a portfolio based on momentum strategy
        @param k (int): look back period.
        '''        
        end_date = self.date_finder(self.start,k*-1,"months") #finds the k date
        start_date = ((datetime.now() - relativedelta(years=5)) + relativedelta(days=+1)).strftime('%Y%m%d') #pulls 5 years of data
        cpu_count = os.cpu_count()
        j_dates = { #gets the j dates
                "j3":self.date_finder(self.start,3,"months"),
                "j6":self.date_finder(self.start,6,"months"),
                "j9":self.date_finder(self.start,9,"months"),
                "j12":self.date_finder(self.start,12,"months"),
                }
           
        work = [] #holds all the params
        split_work = math.floor((len(self.markets)/cpu_count)) #determines how many stocks each core will get 
        start_position = 0#used for distribution for cores
 
        for cpu in range(1,cpu_count+1):
            end_position = (split_work*cpu) - 1 #last index for the core
            
            if cpu == cpu_count: #this makes sure that the last core doesn't miss 1 stock
                end_position = len(self.markets)
            
            parameters = [start_position,end_position,start_date,self.start,end_date,j_dates["j3"],j_dates["j6"],j_dates["j9"],j_dates["j12"]]
            start_position = start_position + split_work #updates start position for the next stock
            work.append(parameters)
                    
        
        pool = Pool(cpu_count)
        results = pool.map(self.strategies[self.portfolio_strategy],work)
        pool.close()
        pool.join()
        
        temp_portfolio = pd.concat(results).sort_values(by=['return'],ascending=False).dropna(subset=["return"])
        temp_portfolio = temp_portfolio.set_index("stock")
 
        cut_off = math.floor(len(temp_portfolio)*.1)

        self.portfolio = pd.concat([temp_portfolio[:cut_off],temp_portfolio[-cut_off:]])
        self.portfolio.loc[:cut_off,"position"] = "buy"
        self.portfolio.loc[-cut_off:,"position"] = "sell"
        
        return self.portfolio
        
    def momen_returns_test(self,parameters):
        portfolios = [[] for i in range(len(self.purchase_dates))] #holds portfolio matrices
        avoid = ["ZJZZT","ZWZZT","ZVV","ZBZX"," TRUE","ZXZZT","ZBZZT","ZAZZT"]
        for stock_index in range(parameters[0],parameters[1]):
            if self.markets[stock_index] not in avoid:
                try:
                    current_asset = mk.Asset(self.markets[stock_index],start=parameters[2])  
                    
                    for index in range(len(self.purchase_dates)):
        
                            k_date = self.date_finder(self.purchase_dates[index],parameters[3],"months")
                            j_dates = { #gets the j dates
                                    "j3":self.date_finder(self.purchase_dates[index],3,"months"),
                                    "j6":self.date_finder(self.purchase_dates[index],6,"months"),
                                    "j9":self.date_finder(self.purchase_dates[index],9,"months"),
                                    "j12":self.date_finder(self.purchase_dates[index],12,"months"),
                                    }
                            date_list = [self.purchase_dates[index],k_date,j_dates["j3"],j_dates["j6"],j_dates["j9"],j_dates["j12"]]
                            proper_dates = self.date_adjuster(current_asset.candles,date_list)
                            
                            if proper_dates!=False:
                                purchase_price = current_asset.candles.close.loc[proper_dates[0]]
                                k_price = current_asset.candles.close.loc[proper_dates[1]]
                                t_return = (purchase_price - k_price) / k_price
                                j3 = ((current_asset.candles.close.loc[proper_dates[2]] - purchase_price)/purchase_price) /3 
                                j6 = ((current_asset.candles.close.loc[proper_dates[3]] - purchase_price)/purchase_price) / 6
                                j9 = ((current_asset.candles.close.loc[proper_dates[4]] - purchase_price)/purchase_price) / 9
                                j12 = ((current_asset.candles.close.loc[proper_dates[5]] - purchase_price)/purchase_price) / 12
                                portfolios[index].append([self.markets[stock_index],proper_dates[0],t_return,None,j3,j6,j9,j12]) 
        
                except:
                    pass
            
        final_portfolios = []  
        for matrix in portfolios:
            final_portfolios.append(pd.DataFrame(matrix,columns = ["stock","p_date","avg_return","position","j3","j6","j9","j12"]).dropna(subset=["avg_return"]))
        
        return final_portfolios

    
    def momen_win_loss_test(self,k=6):
        start_time = datetime.now()
        start_date = ((datetime.now() - relativedelta(years=5)) + relativedelta(days=+1))
        final_purchase_date = (datetime.now() - relativedelta(years=1))

        current_purchase_date = self.date_finder(start_date,k,"months")
        self.purchase_dates.append(current_purchase_date)
        
        while current_purchase_date<=final_purchase_date:
            current_purchase_date = self.date_finder(current_purchase_date,1,"months")
            self.purchase_dates.append(current_purchase_date)
        
        cpu_count = os.cpu_count()
        work = [] #holds all the params
        split_work = math.floor((len(self.markets)/cpu_count)) #determines how many stocks each core will get 
        start_position = 0#used for distribution for cores
        
        for cpu in range(1,cpu_count+1):
            end_position = (split_work*cpu) - 1 #last index for the core
            
            if cpu == cpu_count: #this makes sure that the last core doesn't miss 1 stock
                end_position = len(self.markets)
            
            parameters = [start_position,end_position,start_date.strftime('%Y-%m-%d'),k]
            start_position = start_position + split_work #updates start position for the next stock
            work.append(parameters)
        
        pool = Pool(cpu_count)
        results = pool.map(self.momen_returns_test,work)
        pool.close()
        pool.join()
        
        final_portfolios = []
        current_portfolio = []

        for purchase_date in range(len(results[0])):
            for data in range(len(results)):
                current_portfolio.append(results[data][purchase_date])
            
            temp_portfolio = pd.concat(current_portfolio).sort_values(by=['avg_return'],ascending=False)
            temp_portfolio = temp_portfolio.set_index("stock")
            cut_off = math.floor(len(temp_portfolio)*.1)
            temp_portfolio = pd.concat([temp_portfolio[:cut_off],temp_portfolio[-cut_off:]])
            temp_portfolio.loc[:cut_off,"position"] = "buy"
            temp_portfolio.loc[-cut_off:,"position"] = "sell"
            final_cut_off = math.ceil(len(temp_portfolio)*.02)
            temp_portfolio = pd.concat([temp_portfolio[final_cut_off:],temp_portfolio[:-final_cut_off]])
            temp_portfolio["avg_return"] = temp_portfolio["avg_return"] / k
            final_portfolios.append(temp_portfolio)
            current_portfolio = []
            
        total_portfolio = pd.concat(final_portfolios)
        buy_port = total_portfolio[total_portfolio["position"]=="buy"]
        sell_port = total_portfolio[total_portfolio["position"]=="sell"]
        j_columns = ["j3","j6","j9","j12"]

        results_matrix = []
        for j_column in j_columns:
            t_test_buy = ttest_ind(buy_port[j_column], buy_port['avg_return'])
            j_buy_mean = buy_port[j_column].mean()
            t_test_sell = ttest_ind(sell_port[j_column], sell_port['avg_return'])
            j_sell_mean = (sell_port[j_column].mean())
            sell_row = [j_column+"_sell",j_sell_mean,t_test_sell]
            results_matrix.append(sell_row)
            buy_row = [j_column+"_buy",j_buy_mean,t_test_buy]
            results_matrix.append(buy_row)
            t_buysell = ttest_ind((buy_port[j_column] - sell_port[j_column]).dropna(), (buy_port["avg_return"] - sell_port["avg_return"]).dropna())
            results_matrix.append([j_column+"_buy-sell",j_buy_mean-j_sell_mean,t_buysell])
            
        
        df = pd.DataFrame(results_matrix,columns = ["j_value","k_value_"+str(k),"t/p-values"]).set_index("j_value")
        end_time = datetime.now()
        print(end_time - start_time)
        return df
        
   
def main():
    a = Portfolio("momen_returns")
    print(a.momen_win_loss_test(k=3))
    #test = [1,2,3,4,5,6,7,8,9]


if __name__ == '__main__':
    freeze_support()
    main()