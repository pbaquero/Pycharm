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