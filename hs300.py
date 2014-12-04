from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import math
from datetime import *
import pprint

FOLDER_PATH = "C:\\New folder\\hs300_timing\\"
FILE_EXTENSION = ".csv"
DATA_FILE_NAMES = ["000928","000929","000930"]
INDEX_HISTORY_FILE_NAME = "000931"
DATE_INDEX = "Date"
INIT_DATE = datetime(2009,1,1)
END_DATE = datetime(2014,12,30)
COMMISSION_RATE = 0
ROLLING_WINDOW = 5
COV_ROLLING_WINDOW = 20

class trades:
    def __init__(self):
        self.trades = pd.DataFrame(columns=["DateTime", "Price", "Lots", "PnL", "Pos", "DrawDown", "Note"])
    def add(self, datetime, price, lots, note=None):
        self.trades = self.trades.append([{"DateTime":datetime,
                                          "Price":price,
                                          "Lots":lots,
                                          "PnL":np.nan,
                                          "Pos":np.nan,
                                          "DrawDown":np.nan,
                                          "Note":note}])
    def position(self):
        return int(sum(self.trades["Lots"]))
    def pnl(self, prices, on_index, price_index):
        pnl = 0
        pos = 0
        max_pnl = 0
        self.history = pd.merge(self.trades, prices, left_on="DateTime", right_on=on_index, how='outer')
        for rn in range(0,len(self.history)):
            if not math.isnan(self.history.iloc[rn]["Lots"]):
                pos += self.history.iloc[rn]["Lots"]
                pnl -= self.history.iloc[rn]["Price"]*self.history.iloc[rn]["Lots"]
            if pos <> 0:
                pnl += pnl*self.history.iloc[rn][price_index]
            self.history.iloc[rn]["PnL"] = pnl
            self.history.iloc[rn]["Pos"] = pos
            max_pnl = max_pnl if pnl<max_pnl else pnl
            drawdown = 1-pnl/max_pnl
            self.history.iloc[rn]["DrawDown"] = drawdown
    def pprint(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.trades)        
        
def read_data(name, df=None):
    close_index = name+".Close"
    return_index = name+".Return"
    beta_index = name+".Beta"
    temp_df = pd.read_csv(FOLDER_PATH+name+FILE_EXTENSION, delimiter="\t")
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
    # Calculate daily return
    get_return = lambda x: x[1]/x[0]-1
    temp_df[return_index] = pd.rolling_apply(temp_df[close_index], 2, get_return, min_periods=2)    # Calculate beta
    if not df is None:
        temp_df = pd.merge(df, temp_df, on=DATE_INDEX)
        temp_df[beta_index] = pd.rolling_cov(temp_df[return_index], temp_df[INDEX_HISTORY_FILE_NAME+".Return"], COV_ROLLING_WINDOW, min_periods=COV_ROLLING_WINDOW)/\
                              pd.rolling_var(temp_df[return_index], COV_ROLLING_WINDOW, min_periods=COV_ROLLING_WINDOW)
        # Calculate alpha
        temp_df[name+".Alpha"] = temp_df[return_index] - temp_df[INDEX_HISTORY_FILE_NAME+".Return"]*temp_df[beta_index]
    return temp_df

df = read_data(INDEX_HISTORY_FILE_NAME)
for name in DATA_FILE_NAMES:
    df = read_data(name, df)

##print(df.head())

orders = trades()
df["Positive_Alpha_Ratio"] = 0
df["Positive_Return_Ratio"] = 0
for rn in range(0, len(df)):
    total_number_of_positive_alpha = 0
    total_number_of_positive_return = 0
    for name in DATA_FILE_NAMES:
    # Calculate number(percentage) of equities with positive alpha
        if df.iloc[rn][name+".Alpha"]>0:
            total_number_of_positive_alpha += 1
    # Calculate number(percentage) of equities with positive return
        if df.iloc[rn][name+".Return"]>0:
            total_number_of_positive_return += 1
    df.iloc[rn]["Positive_Alpha_Ratio"] = total_number_of_positive_alpha/len(DATA_FILE_NAMES)
    df.iloc[rn]["Positive_Return_Ratio"] = total_number_of_positive_return/len(DATA_FILE_NAMES)
get_average = lambda x: sum(x)/ROLLING_WINDOW
df["Average_Alpha_Ratio"] = pd.rolling_apply(df["Positive_Alpha_Ratio"], ROLLING_WINDOW, get_average, min_periods=ROLLING_WINDOW)
df["Average_Return_Ratio"] = pd.rolling_apply(df["Positive_Return_Ratio"], ROLLING_WINDOW, get_average, min_periods=ROLLING_WINDOW)
# Trade at close price
for rn in range(ROLLING_WINDOW, len(df)):
    if df.iloc[rn-1]["Average_Alpha_Ratio"]>0.5 and\
    df.iloc[rn-1]["Average_Return_Ratio"]<0.5 and\
    orders.position() == 0:
        orders.add(datetime=df.iloc[rn][DATE_INDEX],
                   price=df.iloc[rn][INDEX_HISTORY_FILE_NAME+".Close"],
                   lots=1)
    elif orders.position() > 0:
        orders.add(datetime=df.iloc[rn][DATE_INDEX],
                   price=df.iloc[rn][INDEX_HISTORY_FILE_NAME+".Close"],
                   lots=-1)
        

        
        


##def cal_beta(ts1, ts2, date_index=None, price_index=None, min_periods=None):
##    if date_index is None:
##        date_index = "Date"
##    if price_index is None:
##        price_index = "Price"
##    if min_periods is None:
##        min_periods = 1
##    merged_ts = pd.merge(ts1, ts2, on=date_index)
##    cov = merged_ts[price_index+"_x"].cov(merged_ts[price_index+"_y"])
####    cov = pd.expanding_cov(merged_ts, min_periods=min_periods)
##    var = pd.expanding_var(ts1, min_periods)
##    return cov/var
