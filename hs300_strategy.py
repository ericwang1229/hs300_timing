from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import math
from datetime import *
import pprint
import functools
from matplotlib.dates import  DateFormatter, WeekdayLocator, HourLocator, \
     DayLocator, MONDAY, MonthLocator,AutoDateLocator

FOLDER_PATH = "C:\\New folder\\hs300_timing\\from2005\\"
FILE_EXTENSION = ".txt"
DATA_FILE_NAMES = ["000908",
                   "000909",
                   "000910",
                   "000911",
                   "000912",
                   "000913",
                   "000914",
                   "000915",
                   "000916",
                   "000917"]
INDEX_HISTORY_FILE_NAME = "000300"
DATE_INDEX = "DateTime"
INIT_DATE = datetime(2006,4,13)
END_DATE = datetime(2014,12,06)
COMMISSION_RATE = 0
ROLLING_WINDOW = 5
COV_ROLLING_WINDOW = 20

class trades:
    def __init__(self):
        self.trades = pd.DataFrame(columns=[DATE_INDEX, "Price", "Lots", "PnL","DailyPnL", "Pos", "Ave_Cost", "DrawDown", "Note"])
    def add(self, datetime, price, lots, note=None):
        self.trades = self.trades.append([{DATE_INDEX:datetime,
                                          "Price":price,
                                          "Lots":lots,
                                          "PnL":np.nan,
                                           "DailyPnL":np.nan,
                                           "Pos":np.nan,
                                           "Ave_Cost":np.nan,
                                           "DrawDown":np.nan,
                                           "Note":note}])
    def position(self):
        return int(sum(self.trades["Lots"]))
    def pnl(self, prices, price_index):
        pnl = 0
        pos = 0
        average_cost = 0
        max_pnl = 0
        max_drawdown = 0
        self.history = pd.merge( prices,self.trades,on=DATE_INDEX,how='outer')
        for rn in range(0,len(self.history)):
            if not math.isnan(self.history.iloc[rn]["Lots"]):
                if pos + self.history.iloc[rn]["Lots"]<>0:
                    average_cost = (pos*average_cost + self.history.iloc[rn]["Lots"]*self.history.iloc[rn]["Price"])/(pos+self.history.iloc[rn]["Lots"])
                    pos += self.history.iloc[rn]["Lots"]
                    pnl += pos*(self.history.iloc[rn][price_index]-average_cost)
                else:
                    pnl += pos *(self.history.iloc[rn]["Price"]-average_cost)
                    average_cost = 0
                    pos = 0
            else:
                pnl += pos*(self.history.iloc[rn][price_index]-average_cost)
                average_cost = self.history.iloc[rn][price_index]
            self.history.ix[rn, "PnL"] = pnl
            self.history.ix[rn, "Pos"] = pos
            self.history.ix[rn, "Ave_Cost"] = average_cost
            if rn>0:
                self.history.ix[rn, "DailyPnL"] = self.history.iloc[rn]["PnL"]-self.history.iloc[rn-1]["PnL"]
            max_pnl = max_pnl if pnl<max_pnl else pnl
            if max_pnl == 0:
                drawdown = 0.0
            else:
##                drawdown = 1.0-pnl/max_pnl
                drawdown = max_pnl - pnl
            max_drawdown = max_drawdown if drawdown<max_drawdown else drawdown
            self.history.ix[rn, "DrawDown"] = drawdown
        self.final_pnl = pnl
        self.max_drawdown = max_drawdown
        self.sharpe_ratio = self.history["DailyPnL"].mean()/self.history["DailyPnL"].std()
    def pprint(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.trades)
    def wining_rate(self):
        wins = 0
        loses = 0
        total_win = 0
        total_lose = 0
        pos = 0
        for rn in range(0, len(self.trades)):
            pos += self.trades.iloc[rn]["Lots"]
            if pos == 0 :
                if self.trades.iloc[rn]["Price"] > self.trades.iloc[rn-1]["Price"]:
                    wins += 1
                    total_win += self.trades.iloc[rn]["Price"] - self.trades.iloc[rn-1]["Price"]
                else:
                    loses += 1
                    total_lose -= self.trades.iloc[rn]["Price"] - self.trades.iloc[rn-1]["Price"]
        self.wins = wins
        self.loses = loses
        self.winning_rate = wins/(wins+loses)
        self.average_win = total_win/wins
        self.average_lose = total_lose/loses
    def plot(self, ax=None):
        date = self.history[DATE_INDEX].tolist()
        prices = self.history[INDEX_HISTORY_FILE_NAME+".Close"].values.tolist()
        buy_dates = [x["DateTime"] for i,x in self.history.iterrows() if x["Lots"]>0]
        buy_prices = [x["Price"] for i,x in self.history.iterrows() if x["Lots"]>0]
        sell_dates = [x["DateTime"] for i,x in self.history.iterrows() if x["Lots"]<0]
        sell_prices = [x["Price"] for i,x in self.history.iterrows() if x["Lots"]<0]
        daily_pnl = self.history["DailyPnL"].tolist()
        pnl = self.history["PnL"].tolist()
        if ax is None:
            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
            dateLocator = AutoDateLocator()
##            dateLocator = DayLocator()
            dateFormatter = DateFormatter('%Y-%m-%d')
            ax1.xaxis.set_major_locator(dateLocator)
            ax1.xaxis.set_major_formatter(dateFormatter)
            ax1.plot(date, prices)
            ax1.plot(buy_dates, buy_prices, "^", markersize = 5, color='m')
            ax1.plot(sell_dates, sell_prices, "v", markersize = 5, color='k')
            ax1.set_title("Price and Trades")
            ax2.plot(date, pnl)
            ax2.set_title("PnL")
            ax3.bar(date, daily_pnl)
            ax1.xaxis_date()
            ax1.autoscale_view()
            plt.setp(plt.gca().get_xticklabels(),rotation=45,horizontalalignment='right')
            plt.show()
        
            
def read_data(name, df=None):
    close_index = name+".Close"
    return_index = name+".Return"
    beta_index = name+".Beta"
    temp_df = pd.read_csv(FOLDER_PATH+name+FILE_EXTENSION,
                          delimiter="\t",
                          parse_dates=True,
                          index_col=False
##                          date_parser=functools.partial(datetime.strptime, format = "%Y/%m/%d")
                          )
    temp_df.columns=[DATE_INDEX,
                     name+".Open",
                     name+".High",
                     name+".Low",
                     name+".Close",
                     name+".Volume",
                     name+".Vol",
                     name+".MA1",
                     name+".MA2",
                     name+".MA3",
                     name+".MA4",
                     name+".MA5",
                     name+".MA6"]
    temp_df = temp_df.drop([
                     name+".Open",
                     name+".High",
                     name+".Low",
                     name+".MA1",
                     name+".Vol",
                  name+".MA2",
                  name+".MA3",
                  name+".MA4",
                  name+".MA5",
                  name+".MA6"],1)
    # Rule out invalid data
    temp_df = temp_df[(temp_df[name+".Volume"]>0)]
    for rn in range(0, len(temp_df)):
        temp_df.ix[rn, DATE_INDEX] = pd.to_datetime(temp_df.iloc[rn][DATE_INDEX])
    # Calculate daily return
    get_return = lambda x: x[1]/x[0]-1
    temp_df[return_index] = pd.rolling_apply(temp_df[close_index], 2, get_return, min_periods=2)    # Calculate beta
    if not df is None:
        temp_df = pd.merge(df, temp_df, on=DATE_INDEX, how='outer')
        temp_df[beta_index] = pd.rolling_cov(temp_df[return_index], temp_df[INDEX_HISTORY_FILE_NAME+".Return"], COV_ROLLING_WINDOW, min_periods=COV_ROLLING_WINDOW)/\
                              pd.rolling_var(temp_df[INDEX_HISTORY_FILE_NAME+".Return"], COV_ROLLING_WINDOW, min_periods=COV_ROLLING_WINDOW)
        # Calculate alpha
        temp_df[name+".Alpha"] = temp_df[return_index] - temp_df[INDEX_HISTORY_FILE_NAME+".Return"]*temp_df[beta_index]
    return temp_df

df = read_data(INDEX_HISTORY_FILE_NAME)
for name in DATA_FILE_NAMES:
    df = read_data(name, df)
    
df["Positive_Alpha_Ratio"] = 0.0
df["Positive_Return_Ratio"] = 0.0
for rn in range(0, len(df)):
##    df.ix[rn, DATE_INDEX] = pd.to_datetime(df.iloc[rn][DATE_INDEX])
    total_number_of_positive_alpha = 0
    total_number_of_positive_return = 0
    for name in DATA_FILE_NAMES:
    # Calculate number(percentage) of equities with positive alpha
        if df.iloc[rn][name+".Alpha"]>0:
            total_number_of_positive_alpha += 1
    # Calculate number(percentage) of equities with positive return
        if df.iloc[rn][name+".Return"]>0:
            total_number_of_positive_return += 1
    df.ix[rn, "Positive_Alpha_Ratio"] = total_number_of_positive_alpha/float(len(DATA_FILE_NAMES))
    df.ix[rn, "Positive_Return_Ratio"] = total_number_of_positive_return/float(len(DATA_FILE_NAMES))

get_average = lambda x: sum(x)/ROLLING_WINDOW
df["Average_Alpha_Ratio"] = pd.rolling_apply(df["Positive_Alpha_Ratio"], ROLLING_WINDOW, get_average, min_periods=ROLLING_WINDOW)
df["Average_Return_Ratio"] = pd.rolling_apply(df["Positive_Return_Ratio"], ROLLING_WINDOW, get_average, min_periods=ROLLING_WINDOW)

##df = df[2375:] #306=2006-04-13

def simu_trade(df, orders, Average_Alpha_Ratio_Thr, Average_Return_Ratio_Thr):
# Trade at close price
    for rn in range(COV_ROLLING_WINDOW, len(df)):
        if df.iloc[rn-1]["Average_Alpha_Ratio"]<Average_Alpha_Ratio_Thr and\
           df.iloc[rn-1]["Average_Return_Ratio"]>Average_Return_Ratio_Thr and\
           orders.position() == 0:
            orders.add(datetime=df.iloc[rn][DATE_INDEX],
                       price=df.iloc[rn][INDEX_HISTORY_FILE_NAME+".Close"],
                       lots=1)
        elif orders.position() > 0 and\
             (df.iloc[rn-1]["Average_Alpha_Ratio"]>Average_Alpha_Ratio_Thr or\
             df.iloc[rn-1]["Average_Return_Ratio"]<Average_Return_Ratio_Thr):
            orders.add(datetime=df.iloc[rn][DATE_INDEX],
                       price=df.iloc[rn][INDEX_HISTORY_FILE_NAME+".Close"],
                       lots=-1)
        #Sell Short
    ##    if df.iloc[rn-1]["Average_Alpha_Ratio"]>Average_Alpha_Ratio_Thr and\
    ##       df.iloc[rn-1]["Average_Return_Ratio"]<Average_Return_Ratio_Thr and\
    ##       orders.position() == 0:
    ##        orders.add(datetime=df.iloc[rn][DATE_INDEX],
    ##                   price=df.iloc[rn][INDEX_HISTORY_FILE_NAME+".Close"],
    ##                   lots=-1)
    ##    elif orders.position() < 0 and\
    ##         (df.iloc[rn-1]["Average_Alpha_Ratio"]<Average_Alpha_Ratio_Thr or\
    ##         df.iloc[rn-1]["Average_Return_Ratio"]>Average_Return_Ratio_Thr):
    ##        orders.add(datetime=df.iloc[rn][DATE_INDEX],
    ##                   price=df.iloc[rn][INDEX_HISTORY_FILE_NAME+".Close"],
    ##                   lots=1)
    return orders
orders = trades()
orders = simu_trade(df, orders, 0.5, 0.5)
price_for_pnl = df[[DATE_INDEX,INDEX_HISTORY_FILE_NAME+".Close"]]
price_for_pnl.columns = [DATE_INDEX, INDEX_HISTORY_FILE_NAME+".Close"]
orders.pnl(price_for_pnl, price_index=INDEX_HISTORY_FILE_NAME+".Close")

excel = pd.merge(df, orders.history, on=DATE_INDEX, how='outer')
writer = pd.ExcelWriter(FOLDER_PATH+"output.xlsx")
excel.to_excel(writer)
writer.save()
orders.plot()        

##pnl = []
##dd = []
##r = []
##for i in range(0,10,1):
##    pnl.append([])
##    dd.append([])
##    r.append([])
##    for j in range(0,10,1):
##        Average_Alpha_Ratio_Thr = i/10.0
##        Average_Return_Ratio_Thr = j/10.0
##        empty_orders = trades()
##        orders = simu_trade(df, empty_orders, Average_Alpha_Ratio_Thr,Average_Return_Ratio_Thr)
##        price_for_pnl = df[[DATE_INDEX,INDEX_HISTORY_FILE_NAME+".Close"]]
##        price_for_pnl.columns = [DATE_INDEX, INDEX_HISTORY_FILE_NAME+".Close"]
##        orders.pnl(price_for_pnl, price_index=INDEX_HISTORY_FILE_NAME+".Close")
##        pnl[i].append((orders.final_pnl))
##        dd[i].append((orders.max_drawdown))
##        r[i].append((orders.final_pnl/orders.max_drawdown))
##        print Average_Alpha_Ratio_Thr.__str__()+", "+\
##              Average_Return_Ratio_Thr.__str__()+", "+\
##              orders.final_pnl.__str__()+", "+\
##              orders.max_drawdown.__str__()
##pnl = pd.DataFrame(pnl)
##writer = pd.ExcelWriter(FOLDER_PATH+"pnl.xlsx")
##pnl.to_excel(writer)
##writer.save()
##dd = pd.DataFrame(dd)
##writer = pd.ExcelWriter(FOLDER_PATH+"dd.xlsx")
##dd.to_excel(writer)
##writer.save()
##r = pd.DataFrame(r)
##writer = pd.ExcelWriter(FOLDER_PATH+"r.xlsx")
##r.to_excel(writer)
##writer.save()

        
