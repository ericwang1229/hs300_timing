from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import csv
from datetime import *
import pprint

FOLDER_PATH = ""
FILE_EXTENSION = ".csv"
DATA_FILE_NAMES = []
INDEX_HISTORY_FILE_NAME = ""
DATE_INDEX = "Date"
INIT_DATE = datetime(2009,1,1)
END_DATE = datetime(2014,12,30)
COMMISSION_RATE = 0
ROLLING_WINDOW = 5

class trades:
    def __init__(self):
        self.trades = pd.DataFrame(columns=["DateTime", "Price", "Lots", "Note"])
    def add(self, datetime, price, lots, note):
        self.trades = self.trades.append({"DateTime":datetime, "Price":price, "Lots":lots, "Note":note})
    def position(self):
        return int(sum(self.trades["Lots"]))
    def pnl(self):
        assert(self.position == 0)
        return sum([-x["Price"]*x["Lots"] for x in self.trades])
    def print(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.trades)
        
def read_data(name):
    close_index = name+".Close"
    return_index = name+".Return"
    beta_index = name+".Beta"
    temp_df = pd.read_csv(FOLDER_PATH+name+FILE_EXTENSION)
    # Rule out invalid data
    temp_df = df[(df[name+".Volume"]>0)]
    # Calculate daily return
    get_return = lambda x: x[1]/x[0]-1
    temp_df[return_index] = pd.rolling_apply(df[close_index], 2, get_return, min_periods=2)
    # Calculate beta
    temp_df[beta_index] = pd.rolling_cov(df[return_index], temp_df[INDEX_HISTORY_FILE_NAME+".Return"], 200, min_periods=200)/\
                            pd.rolling_var(df[return_index], 200, min_periods=200)
    # Calculate alpha
    temp_df[name+".Alpha"] = temp_df[return_index] - temp_df[INDEX_HISTORY_FILE_NAME+".Return"]*temp_df[beta_index]
    return temp_df

df = read_data(INDEX_HISTORY_FILE_NAME)
for name in DATA_FILE_NAMES:
    df = pd.merge(df, read_data(name), on=DATE_INDEX)

orders = trades()

for row in df:
    total_number_of_positive_alpha = 0
    total_number_of_positive_return = 0
    for name in DATA_FILE_NAME:
    # Calculate number(percentage) of equities with positive alpha
        if row[name+".Alpha"]>0:
            total_number_of_positive_alpha += 1
    # Calculate number(percentage) of equities with positive return
        if row[name+".Return"]>0:
            total_number_of_positive_return += 1
    row["Positive_Alpha_Ratio"] = total_number_of_positive_alpha/len(DATA_FILE_NAMES)
    row["Positive_Return_Ratio"] = total_number_of_positive_return/len(DATA_FILE_NAMES)
    # Trade, entry at open price, exit at


        
        


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
