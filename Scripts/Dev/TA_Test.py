import xlsxwriter
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import sys
from string import digits
import os
import time
from sqlalchemy import create_engine, MetaData
import psycopg2
import io
import ta


USD_JPY_df = pd.read_pickle(
    "C:/Users/pebaqu/OneDrive - SAS/Profiles for l10c581/l10c581/Desktop/Personal/Python/Datasets/2018-07-01_USD_JPY")

global df_usd_jpy
df_usd_jpy = USD_JPY_df.drop(['Complete', 'Time'], axis=1).copy()
df_usd_jpy.reset_index(inplace=True)
df_usd_jpy['TID'] = df_usd_jpy.index
length = max(df_usd_jpy['TID']) + 1

#adds the week number to the dataframe
df_usd_jpy['Week'] = df_usd_jpy['Time'].dt.strftime("%W")
#adds the week number to the dataframe
df_usd_jpy['Weekday'] = df_usd_jpy['Time'].dt.strftime("%A")
#create a counter column for week and day calculations
df_usd_jpy['Counter'] = 1
#how many days are in a week? How many trades in a day # TODO hardcoded week numbers. Need to make dynamic. Maybe set where day count <7?
df_length_base = df_usd_jpy.loc[~df_usd_jpy['Week'].isin(['26','31'])]
df_week = df_length_base[['Week', 'Counter']].groupby(['Week']).sum()
df_weekday_base = df_length_base[['Week','Weekday','Counter']].groupby([df_length_base['Week'], df_length_base['Weekday']]).sum()
#df_weekday = df_weekday_base.groupby(df_weekday_base['Weekday']).mean()
#Set length variables~ # TODO couldn't get weekday to work so just averaging by 6
Len_Week = int(df_week["Counter"].mean())
Len_2_Week = int(Len_Week * 2)
Len_Day = int(Len_Week // 6)
Length = [Len_2_Week, Len_Week, Len_Day]


moving_percents = [.005, .01, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50]
#fast_moving_percents = [.05, .10, .15, .20, .25, .30, .35, .40, .45, .50]

#slow_ma = [int(i * length) for i in slow_moving_percents]
array_len = int(len(moving_percents)) * int(len(Length))
values = np.zeros((array_len,), dtype=[('Len_Name', 'S20'), ('Slow', 'int64'), ('Fast', 'int64')]) #TODO declare as array with column names. calculate length (len(moving_percents) * len(Length)). Need to figure out how to get Length name (e.g. day, week, 2 week)
#create length variables for slow and fast moving averages.

slow_values = []
fast_values = []

for len in Length:
    for per in moving_percents:
        slow_value_calc = int(per*len)
        fast_value_calc = int(per * per * len)
        if slow_value_calc not in slow_values:
            slow_values.append(slow_value_calc)
        if fast_value_calc not in fast_values:
            fast_values.append(fast_value_calc)


'''
for len in Length: #cycle through two week, one week, and one day values
    len_iter = 0
    for slow_per in moving_percents:
        slow_per_iter = 0
        smp_value = int(len * slow_per)
        fast_per_iter = 0
        for fast_per in moving_percents:
            values[fast_per_iter]['Slow'] = smp_value
            values[fast_per_iter]['Fast'] = int(len * slow_per * fast_per)
            fast_per_iter += 1

'''

df_usd_jpy['Bid_AO'] = ta.momentum.ao(df_usd_jpy.Bid_High, df_usd_jpy.Bid_Low)
df_usd_jpy['Ask_AO'] = ta.momentum.ao(df_usd_jpy.Ask_High, df_usd_jpy.Ask_Low)

# Takes a long time to process and is part of rsi
#df_usd_jpy['Bid_MFI'] = ta.momentum.money_flow_index(df_usd_jpy.Bid_High, df_usd_jpy.Bid_Low, df_usd_jpy.Bid_Close, df_usd_jpy.Volume)

#df_usd_jpy['Bid_ATR'] = ta.volatility.average_true_range(df_usd_jpy.Bid_High, df_usd_jpy.Bid_Low, df_usd_jpy.Bid_Close)
#df_usd_jpy['Ask_ATR'] = ta.volatility.average_true_range(df_usd_jpy.Ask_High, df_usd_jpy.Ask_Low, df_usd_jpy.Ask_Close)

df_usd_jpy['Bid_MACD'] = ta.trend.macd(df_usd_jpy.Bid_Close, n_fast=12, n_slow=26)
df_usd_jpy['Ask_MACD'] = ta.trend.macd(df_usd_jpy.Ask_Close, n_fast=12, n_slow=26)


df_usd_jpy['Bid_MACD_Sig'] = ta.trend.macd_signal(df_usd_jpy.Bid_Close, n_fast=12, n_slow=26, n_sign=9)
df_usd_jpy['Ask_MACD_Sig'] = ta.trend.macd_signal(df_usd_jpy.Ask_Close, n_fast=12, n_slow=26, n_sign=9)

df_test = df_usd_jpy[df_usd_jpy['Bid_MACD']<df_usd_jpy['Bid_MACD_Sig']]

df_usd_jpy['Prev_Bid_Close'] = df_usd_jpy['Bid_Close'].shift(-1)
df_usd_jpy['Prev_Ask_Close'] = df_usd_jpy['Ask_Close'].shift(-1)

#df_usd_jpy["Bid_Trend"] = ["Up" if ele[0] < ele[1] else "Unknown" for ele in df_usd_jpy[['Bid_Open', 'Prev_Bid_Close']]]


def ToT_Trend(row):
    if row['Prev_Bid_Close'] < row['Bid_Open']:
        val = 1
    elif row['Prev_Bid_Close'] > row['Bid_Open']:
        val = -1
    else:
        val = 0
    return val

def Trend(row):
    if row['Bid_Close'] < row['Bid_Open']:
        val = 1
    elif row['Bid_Close'] > row['Bid_Open']:
        val = -1
    else:
        val = 0
    return val


df_usd_jpy["Bid_ToT_Trend"] = df_usd_jpy.apply(ToT_Trend, axis = 1)

df_usd_jpy["Bid_Trend"] = df_usd_jpy.apply(Trend, axis = 1)

df_usd_jpy = df_usd_jpy[0:100]

plt.plot(df_usd_jpy.Time, df_usd_jpy.Bid_MACD)
plt.plot(df_usd_jpy.Time, df_usd_jpy.Bid_MACD_Sig)

plt.show()

print("hello")