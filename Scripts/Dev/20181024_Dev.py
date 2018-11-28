# TODO split SMA and FMA into new columns
# TODO add time back to output
# TODO create a database that will feed powerbi with tick data and results
# TODO check other datapoints - currently using rolling mean of close rolling(fma_window).mean()
# TODO deal with equity
# TODO pull out arrays into a single location and rationalize definition
# TODO take absolute values for profit and maxdd
# TODO mix and match slow and fast trends across time scales (e.g. 5 second fast with 1 minute slow)



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

dw = 'postgresql://postgres:!QAZxsw2@localhost:5432/PythonForex'
dw = create_engine(dw)
timestamp = time.strftime("%Y%m%d_%H%M%S")

'''
import mpl_finance
from mpl_finance import candlestick_ohlc
from mpl_toolkits import mplot3d
import matplotlib.dates as dates
import datetime
import time
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker

'''


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


def to_pg(df, asset_name):
    table_name = asset_name
    data = io.StringIO()
    df.to_csv(data, header=False, index=False)
    data.seek(0)
    raw = dw.raw_connection()
    curs = raw.cursor()
    curs.execute("select exists(select * from information_schema.tables where table_name=%s)", ('table_name',))
    if curs.fetchone()[0] is True:
        curs.execute("DROP TABLE " + table_name)
    empty_table = pd.io.sql.get_schema(df, table_name, con=dw)
    empty_table = empty_table.replace('"', '')
    curs.execute(empty_table)
    curs.copy_from(data, table_name, sep=',', null = '')
    curs.connection.commit()


def pg_query(query):
    raw = dw.raw_connection()
    curs = raw.cursor()
    curs.execute(query)
    curs.connection.commit()


class Master_Activity:
    def __init__(self):
        self.arr_master = np.zeros((1,), dtype=[('Test_Name', 'S100'), ('Trade', 'int32'), ('Ask', 'float64'),
                                                ('Ask_TID', 'int32'),
                                                ('Units', 'int32'), ('Bid', 'float64'), ('Bid_TID', 'int32'),
                                                ('Profit', 'float64'), ('Stop_Loss_Floor', 'float64'),
                                                ('Stop_Loss_Bid', 'float64'),
                                                ('Stop_Loss_Bid_TID', 'int32'), ('Max_DD', 'float64'),
                                                ('Max_DD_TID', 'int32'),
                                                ('Max_DD_Bid', 'float64'), ('Stop_Loss_Max_DD', 'float64'),
                                                ('Stop_Loss_Max_DD_TID', 'int32'), ('Stop_Loss_Max_DD_Bid', 'float64'),
                                                ('SMA', 'float64'), ('FMA', 'float64'), ('StopLoss', 'float64')])

    def app_activity(self):
        arr_test_name = np.full(trading_test.activity[1:].shape, trading_test.name, dtype='S100')

        new_arr = np.zeros(trading_test.activity[1:].shape,
                           dtype=[('Test_Name', 'S100'), ('Trade', 'int32'), ('Ask', 'float64'), ('Ask_TID', 'int32'),
                                  ('Units', 'int32'), ('Bid', 'float64'), ('Bid_TID', 'int32'),
                                  ('Profit', 'float64'), ('Stop_Loss_Floor', 'float64'), ('Stop_Loss_Bid', 'float64'),
                                  ('Stop_Loss_Bid_TID', 'int32'), ('Max_DD', 'float64'), ('Max_DD_TID', 'int32'),
                                  ('Max_DD_Bid', 'float64'), ('Stop_Loss_Max_DD', 'float64'),
                                  ('Stop_Loss_Max_DD_TID', 'int32'), ('Stop_Loss_Max_DD_Bid', 'float64'),
                                  ('SMA', 'float64'), ('FMA', 'float64'), ('StopLoss', 'float64')])

        new_arr[:]['Test_Name'] = arr_test_name[:]
        new_arr[:]['SMA'] = trading_test.activity[1:]['SMA']
        new_arr[:]['FMA'] = trading_test.activity[1:]['FMA']
        new_arr[:]['StopLoss'] = trading_test.activity[1:]['StopLoss']
        new_arr[:]['Trade'] = trading_test.activity[1:]['Trade']
        new_arr[:]['Ask'] = trading_test.activity[1:]['Ask']
        new_arr[:]['Ask_TID'] = trading_test.activity[1:]['Ask_TID']
        new_arr[:]['Units'] = trading_test.activity[1:]['Units']
        new_arr[:]['Bid'] = trading_test.activity[1:]['Bid']
        new_arr[:]['Bid_TID'] = trading_test.activity[1:]['Bid_TID']
        new_arr[:]['Profit'] = trading_test.activity[1:]['Profit']
        new_arr[:]['Stop_Loss_Floor'] = trading_test.activity[1:]['Stop_Loss_Floor']
        new_arr[:]['Stop_Loss_Bid'] = trading_test.activity[1:]['Stop_Loss_Bid']
        new_arr[:]['Stop_Loss_Bid_TID'] = trading_test.activity[1:]['Stop_Loss_Bid_TID']
        new_arr[:]['Max_DD'] = trading_test.activity[1:]['Max_DD']
        new_arr[:]['Max_DD_TID'] = trading_test.activity[1:]['Max_DD_TID']
        new_arr[:]['Max_DD_Bid'] = trading_test.activity[1:]['Max_DD_Bid']
        new_arr[:]['Stop_Loss_Max_DD'] = trading_test.activity[1:]['Stop_Loss_Max_DD']
        new_arr[:]['Stop_Loss_Max_DD_TID'] = trading_test.activity[1:]['Stop_Loss_Max_DD_TID']
        new_arr[:]['Stop_Loss_Max_DD_Bid'] = trading_test.activity[1:]['Stop_Loss_Max_DD_Bid']

        self.arr_master = np.append(self.arr_master, new_arr)

    def save_activity(self):
        # write array to dataframe
        global df_out
        df_out = pd.DataFrame(data=self.arr_master,  # values
                              columns=self.arr_master.dtype.names)

        df_out['Test_Name'] = df_out['Test_Name'].str.decode('utf-8')

        result_table = 'df_usd_jpy' + timestamp
        base_table = 'df_usd_jpy_base' + timestamp

        to_pg(df_out, result_table)

        query = ('Create table Final_Results' + timestamp + ' as SELECT '
                'a.* '
                ',b.Time as BidTime ,b.Bid_Open ,b.Bid_High ,b.Bid_Low ,b.Bid_Close '
                ',c.Time as AskTime ,c.Ask_Open ,c.Ask_High ,c.Ask_Low ,c.Ask_Close '
                'FROM ' + result_table + ' as a '
                'left join ' + base_table + ' as b '
                'on a.Bid_TID = b.TID '
                'left join ' + base_table + ' as c '
                'on a.Ask_TID = c.TID;')

        pg_query(query)


class Account:
    def __init__(self, name, acct_bal, equity):
        self.name = name
        self.init_bal = acct_bal  # initial balance
        self.acct_bal = acct_bal  # ongoing account balance
        self.equity = equity  # unrealized p/L
        self.buy_status = 1  # these aren't really doing anything
        self.sell_status = 0
        self.trades = 1
        self.profit = 0
        self.units = 0
        self.trade_profit = 0
        # self.activity
        self.activity = np.zeros((1,), dtype=[('Trade', 'int32'), ('Ask', 'float64'), ('Ask_TID', 'int32'),
                                              ('Units', 'int32'), ('Bid', 'float64'), ('Bid_TID', 'int32'),
                                              ('Profit', 'float64'), ('Stop_Loss_Floor', 'float64'),
                                              ('Stop_Loss_Bid', 'float64'),
                                              ('Stop_Loss_Bid_TID', 'int32'), ('Max_DD', 'float64'),
                                              ('Max_DD_TID', 'int32'),
                                              ('Max_DD_Bid', 'float64'), ('Stop_Loss_Max_DD', 'float64'),
                                              ('Stop_Loss_Max_DD_TID', 'int32'), ('Stop_Loss_Max_DD_Bid', 'float64'),
                                              ('SMA', 'float64'), ('FMA', 'float64'), ('StopLoss', 'float64')])
        self.new_activity = np.empty_like(self.activity)
        self.buy_start = 0
        self.buy_end = 0
        self.stop_loss = None  # define stop loss for reporting

    def buy(self, amt, arr_analysis_iter, sma, fma):
        # you buy at the ask and sell at the bid (quoted from perspective of market maker)

        # pull arr_analysis tuple to assign variable values
        ask = arr_analysis[arr_analysis_iter]['Ask']
        TID = arr_analysis[arr_analysis_iter]['TID']
        self.buy_start = arr_analysis_iter
        self.new_activity[0]['SMA'] = sma
        self.new_activity[0]['FMA'] = fma

        # calculate the buy
        self.units = int(amt / ask)  # calcluate units per trade at amount
        self.equity += (self.units * ask)  # set equity to unrealized p/l (this is more accurately "position")
        self.acct_bal -= self.equity  # subtract equity from ongoing account balance

        # update activity
        self.new_activity[0]['Trade'] = self.trades
        self.new_activity[0]['Ask'] = ask
        self.new_activity[0]['Ask_TID'] = TID
        self.new_activity[0]['Units'] = self.units

        # set flags
        self.buy_status = 0
        self.sell_status = 1

    def sell(self, arr_analysis_iter, stop_loss=None):

        self.stop_loss = stop_loss
        self.new_activity[0]['StopLoss'] = stop_loss

        # pull arr_analysis tuple to assign variable values
        bid = arr_analysis[arr_analysis_iter]['Bid']  # pull the current Bid (what you can sell for)
        TID = arr_analysis[arr_analysis_iter]['TID']  # pull the TID for the current Bid
        self.buy_end = arr_analysis_iter + 1  # Define the end of the array as the current row plus 1

        # determine the maximum drawdown
        ask = self.new_activity['Ask']  # the price you bought at
        ask_tid = self.new_activity['Ask_TID']  # the ID of the tick for the price you bought at

        arr_sub_analysis = arr_analysis[self.buy_start:self.buy_end]

        min_bid = np.amin(arr_sub_analysis['Bid'])  # the lowest purchase price in the range of your purchase to sale

        arr_min_bid_tuple = arr_sub_analysis[np.argmin(arr_sub_analysis['Bid'])]  # find the index for the lowest bid
        min_bid_tid = arr_min_bid_tuple['TID']  # get the TID for the lowest bid

        self.new_activity[0]['Max_DD'] = (min_bid * self.units) - self.equity  # calculate the max drawdown amount
        self.new_activity[0]['Max_DD_TID'] = min_bid_tid  # pull the TID for the max dd
        self.new_activity[0]['Max_DD_Bid'] = min_bid  # pull the bid associated with the max dd
        self.new_activity[0]['Stop_Loss_Max_DD'] = (
                                                               min_bid * self.units) - self.equity  # default stop loss max dd amount for graphing
        self.new_activity[0]['Stop_Loss_Max_DD_TID'] = min_bid_tid  # default stop loss max dd tid for graphing
        self.new_activity[0]['Stop_Loss_Max_DD_Bid'] = min_bid  # default stop loss max dd bid for graphing

        if self.stop_loss is not None:  # only calculate stop loss values if stop loss is specified
            stop_loss_floor = (self.equity - self.stop_loss) / self.units  # calculate the Bid value of your loss stop
            self.new_activity[0]['Stop_Loss_Floor'] = stop_loss_floor

            # get array of all ticks where bid is gte to stop loss floor
            arr_gte_stop_loss = np.squeeze(
                np.take(arr_sub_analysis, np.where(arr_sub_analysis['Bid'] >= stop_loss_floor)))

        if stop_loss is None or stop_loss > abs(self.new_activity[0]['Max_DD']):
            self.new_activity[0]['Bid'] = bid  # assign the parameter bid
            self.new_activity[0]['Bid_TID'] = TID  # assign the parameter bid TID
            self.new_activity[0]['Stop_Loss_Floor'] = 0
            self.new_activity[0]['Stop_Loss_Bid'] = 0
            self.new_activity[0]['Stop_Loss_Bid_TID'] = 0

            self.calc_sale()

        else:
            try:
                if arr_gte_stop_loss.size < 2:  # handle data sets where bid is never >= min bid. We actually want to skip stop loss and just use standard logic for realism
                    self.new_activity[0]['Stop_Loss_Bid'] = 0
                    self.new_activity[0]['Stop_Loss_Bid_TID'] = 0
                    self.new_activity[0]['Bid'] = bid  # assign the parameter bid
                    self.new_activity[0]['Bid_TID'] = TID  # assign the parameter bid TID
                    self.calc_sale()

                else:
                    stop_loss_bid_index = np.argmin(arr_gte_stop_loss['TID'])  # get the index of lowest acceptable bid
                    stop_loss_TID = np.amin(arr_gte_stop_loss['TID'])
                    arr_stop_loss_bid = arr_gte_stop_loss[
                        stop_loss_bid_index]  # the row of the lowest purchase price in the range of your purchase to sale
                    stop_loss_bid = arr_stop_loss_bid['Bid']  # the ID of the lowest purchase price

                    self.new_activity[0]['Stop_Loss_Bid'] = stop_loss_bid
                    self.new_activity[0]['Stop_Loss_Bid_TID'] = stop_loss_TID
                    self.new_activity[0]['Bid'] = stop_loss_bid
                    self.new_activity[0]['Bid_TID'] = stop_loss_TID

                    self.new_activity[0]['Stop_Loss_Max_DD'] = (
                                                                           stop_loss_bid * self.units) - self.equity  # calculate the max drawdown amount
                    self.new_activity[0][
                        'Stop_Loss_Max_DD_TID'] = stop_loss_TID  # this is actually the lowest bid and not the dd bid
                    self.new_activity[0][
                        'Stop_Loss_Max_DD_Bid'] = stop_loss_bid  # this is actually the lowest bid and not the dd bid

                    self.calc_sale()

            except:
                print("Unexpected error:", sys.exc_info())
                # print(err)
                print("The current trade is:", self.trades)
                print("The current ask is:", self.new_activity[0]['Ask'])
                print("The current equity is:", self.equity)
                print("The current units are:", self.units)
                print("The current min bid is:", min_bid)
                print("The min bid criteria is:", max_dd_criteria)
                print("Array analysis starts at:", self.buy_start, "and ends at:", self.buy_end)
                sys.exit('Exiting because of error')

        # calc_sale()

    def calc_sale(self):
        # calculate the sale
        self.acct_bal = self.acct_bal + (
                    self.units * self.new_activity[0]['Bid'])  # add realized p/l to account balance
        self.trade_amt = self.units * self.new_activity[0]['Bid']  # calculate trade value
        self.trade_profit = (self.units * self.new_activity[0]['Bid']) - self.equity  # calculate the trade profit
        self.profit += self.trade_profit  # update running profit
        self.equity = 0  # need to change this so it takes bid into accountif stop_loss == None or stop_loss < abs(self.new_activity[0]['Max_DD']):

        # update activity
        # self.new_activity[0]['Bid'] = bid
        # self.new_activity[0]['Bid_TID'] = TID
        self.new_activity[0]['Profit'] = self.trade_profit
        self.activity = np.append(self.activity, self.new_activity)

        # set flags
        self.buy_status = 1
        self.sell_status = 0
        self.trades += 1

    def acct_status(self):
        print("The current account balance is:{0:.3f}".format(self.acct_bal),
              "The current equity is:{0:.3f}".format(self.equity),
              "The profit for this trade was:{0:.3f}".format(self.trade_profit),
              "The current profit is:{0:.3f}".format(self.profit),
              # "The drawdown for this trade is:{0:.3f}".format(self.activity[0]['Max_DD']),
              self.activity,
              sep='\n')

'''
class Metrics:
    def __init__(self, sma, fma):
        self.sma = sma
        self.fma = fma

    def stat_creation(self):
        # Stat column creation
        df_analysis['Bid_Fast_Avg'] = df_analysis.Bid.rolling(self.fma).mean()
        df_analysis['Prev_Bid_Fast_Avg'] = df_analysis['Bid_Fast_Avg'].shift()
        df_analysis['Bid_Slow_Avg'] = df_analysis.Bid.rolling(self.sma).mean()
        df_analysis['Prev_Bid_Slow_Avg'] = df_analysis['Bid_Slow_Avg'].shift()
        df_analysis['Ask_Fast_Avg'] = df_analysis.Ask.rolling(self.fma).mean()
        df_analysis['Prev_Ask_Fast_Avg'] = df_analysis['Ask_Fast_Avg'].shift()
        df_analysis['Ask_Slow_Avg'] = df_analysis.Ask.rolling(self.sma).mean()
        df_analysis['Prev_Ask_Slow_Avg'] = df_analysis['Ask_Slow_Avg'].shift()
        df_analysis['Spread_Fast_Avg'] = df_analysis.Bid_Ask_Spread.rolling(self.fma).mean()
        df_analysis['Prev_Spread_Fast_Avg'] = df_analysis['Spread_Fast_Avg'].shift()
        df_analysis['Spread_Slow_Avg'] = df_analysis.Bid_Ask_Spread.rolling(self.sma).mean()
        df_analysis['Prev_Spread_Slow_Avg'] = df_analysis['Spread_Slow_Avg'].shift()

        # Final analysis dataset creation
        df_analysis_final = df_analysis.dropna()

        # analysis array creation
        arr_ip = [tuple(i) for i in df_analysis_final.values]

        dtyp = np.dtype(list(zip(df_analysis_final.dtypes.index, df_analysis_final.dtypes)))

        global arr_analysis

        arr_analysis = np.array(arr_ip, dtype=dtyp)
'''

class Reporting:
    def __init__(self):
        ##### build arrays #####
        global arr_results
        arr_results = np.empty((1,), dtype=[('TestName', 'S100'), ('TotalNetProfit', 'float64'),
                                            ('TotalTrades', 'int'), ('GrossProfit', 'float64'),
                                            ('GrossLoss', 'float64'), ('ProfitFactor', 'float64'),
                                            ('PercentProfitable', 'float64'), ('WinningTrades', 'int'),
                                            ('LosingTrades', 'int'), ('EvenTrades', 'int'),
                                            ('AvgTradeNetProfit', 'float64'), ('AvgWinningTrade', 'float64'),
                                            ('AvgLosingTrade', 'float64'), ('RatioAvgWinAvgLoss', 'float64'),
                                            ('LargestWinningTrade', 'float64'), ('LargestLosingTrade', 'float64'),
                                            ('MaxConWinTrade', 'int'), ('MaxConLoseTrade', 'int'),
                                            ('AvgBarsInTotalTrades', 'float64'), ('AvgBarsInWinTrades', 'float64'),
                                            ('AvgBarsInLosTrades', 'float64')
                                            ])

        global arr_new_results

        arr_new_results = np.empty_like(arr_results)

    def populate_results(self):
        trades = trading_test.activity[1:, ]

        positive_trades = np.squeeze(np.take(trades, np.where(trades['Profit'] > 0)))
        negative_trades = np.squeeze(np.take(trades, np.where(trades['Profit'] < 0)))
        even_trades = np.squeeze(np.take(trades, np.where(trades['Profit'] == 0)))

        ###### get longest winning and losing trade streak ######

        results_iterator = 0
        curr_max_con_win_trade = 0
        max_con_win_trade = 0

        for i in np.ndenumerate(positive_trades['Trade']):
            if results_iterator != 0:
                prev_trade = results_iterator - 1
                if positive_trades[prev_trade]['Trade'] + 1 == positive_trades[results_iterator]['Trade']:
                    curr_max_con_win_trade += 1
                    if curr_max_con_win_trade > max_con_win_trade:
                        max_con_win_trade = curr_max_con_win_trade
                else:
                    curr_max_con_win_trade = 0

            results_iterator += 1

        results_iterator = 0
        curr_max_con_lose_trade = 0
        max_con_lose_trade = 0

        for i in np.ndenumerate(negative_trades['Trade']):
            if results_iterator != 0:
                prev_trade = results_iterator - 1
                if negative_trades[prev_trade]['Trade'] + 1 == negative_trades[results_iterator]['Trade']:
                    curr_max_con_lose_trade += 1
                    if curr_max_con_lose_trade > max_con_lose_trade:
                        max_con_lose_trade = curr_max_con_lose_trade
                else:
                    curr_max_con_lose_trade = 0

            results_iterator += 1

        ##### populate arr_new_results #####
        arr_new_results['TestName'] = trading_test.name
        arr_new_results['TotalNetProfit'] = trading_test.profit
        arr_new_results['TotalTrades'] = trades.size  # have to subtract the first 0 rowfrom np.zero
        arr_new_results['GrossProfit'] = positive_trades['Profit'].sum()
        arr_new_results['GrossLoss'] = negative_trades['Profit'].sum()
        arr_new_results['ProfitFactor'] = positive_trades['Profit'].sum() / negative_trades['Profit'].sum() if negative_trades.size > 0 else 0
        arr_new_results['PercentProfitable'] = positive_trades.size / trades.size if trades.size > 0 else 0
        arr_new_results['WinningTrades'] = positive_trades.size
        arr_new_results['LosingTrades'] = negative_trades.size
        arr_new_results['EvenTrades'] = even_trades.size
        arr_new_results['AvgTradeNetProfit'] = trading_test.profit / trades.size if trades.size > 0 else 0
        arr_new_results['AvgWinningTrade'] = positive_trades['Profit'].sum() / positive_trades.size if positive_trades.size > 0 else 0
        arr_new_results['AvgLosingTrade'] = negative_trades['Profit'].sum() / negative_trades.size if negative_trades.size > 0 else 0
        arr_new_results['RatioAvgWinAvgLoss'] = (positive_trades['Profit'].sum() / positive_trades.size) / (
                    negative_trades['Profit'].sum() / negative_trades.size) \
            if positive_trades.size > 0 and negative_trades.size > 0 and negative_trades['Profit'].sum() / negative_trades.size > 0 else 0
        arr_new_results['LargestWinningTrade'] = np.amax(positive_trades['Profit']) if positive_trades.size > 0 else 0
        arr_new_results['LargestLosingTrade'] = np.amin(negative_trades['Profit']) if negative_trades.size > 0 else 0
        arr_new_results['MaxConWinTrade'] = max_con_win_trade
        arr_new_results['MaxConLoseTrade'] = max_con_lose_trade
        arr_new_results['AvgBarsInTotalTrades'] = (trades['Bid_TID'].sum() - trades['Ask_TID'].sum()) / trades.size if trades.size > 0 else 0
        arr_new_results['AvgBarsInWinTrades'] = (positive_trades['Bid_TID'].sum() - positive_trades[
            'Ask_TID'].sum()) / positive_trades.size if positive_trades.size > 0 else 0
        arr_new_results['AvgBarsInLosTrades'] = (negative_trades['Bid_TID'].sum() - negative_trades[
            'Ask_TID'].sum()) / negative_trades.size if negative_trades.size > 0 else 0

        global arr_results

        arr_results = np.append(arr_results, arr_new_results)

        # Maximum Adverse Event
        '''
        f, ax = plt.subplots(figsize=(20, 10))
        ax.scatter(abs(positive_trades['Stop_Loss_Max_DD']), abs(positive_trades['Profit']), marker="+", c="g")
        ax.scatter(abs(negative_trades['Stop_Loss_Max_DD']), abs(negative_trades['Profit']), marker="o", c="r")

        ax.plot(min(ax.get_ylim(), ax.get_xlim()), min(ax.get_ylim(), ax.get_xlim()), ls="--", c=".3")
        ax.grid(color='g', linestyle='dashed', linewidth=1)
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        if trading_test.stop_loss is not None:
            plt.axvline(x=trading_test.stop_loss)
            plt.xlabel('Max Drawdown')
            plt.ylabel('Profit ($)')
            ax.set_title("Maximum Adverse Event \n Stop Loss:" + str(trading_test.stop_loss))
        plt.savefig('C:\\Users\\pebaqu\\Desktop\Personal\\Python\\JupyterExports\\' + trading_test.name + '.png')
        '''

    def create_final_report(self):
        # chart the results

        global df_results

        df_results = pd.DataFrame(data=arr_results,  # values
                                  columns=arr_results.dtype.names)  # 1st row as the column names

        df_results.index = df_results['TestName']

        global df_final

        df_final = df_results.iloc[1:, 1:].T


USD_JPY_df = pd.read_pickle(
    "C:/Users/pebaqu/OneDrive - SAS/Profiles for l10c581/l10c581/Desktop/Personal/Python/Datasets/2018-07-01_USD_JPY")

global df_usd_jpy
df_usd_jpy = USD_JPY_df.drop(['Complete', 'Time', 'Volume'], axis=1).copy()
df_usd_jpy.reset_index(inplace=True)
df_usd_jpy['TID'] = df_usd_jpy.index

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

#calcuate slow and fast values
slow_values = [41460, 20730, 34550, 27640]
fast_values = [2073, 10365, 8292, 6219, 9328, 1727, 4146, 9674, 2764]
'''
for length in Length:
    for per in moving_percents:
        slow_value_calc = int(per*length) if int(per*length) is not 0 else 1
        fast_value_calc = int(per * per * length) if int(per * per * length) is not 0 else 1
        if slow_value_calc not in slow_values:
            slow_values.append(slow_value_calc)
        if fast_value_calc not in fast_values:
            fast_values.append(fast_value_calc)
'''

df_usd_jpy['BidAvg'] = df_usd_jpy[['Bid_Open', 'Bid_High', 'Bid_Low', 'Bid_Close']].mean(axis=1)
df_usd_jpy['AskAvg'] = df_usd_jpy[['Ask_Open', 'Ask_High', 'Ask_Low', 'Ask_Close']].mean(axis=1)


#df_usd_jpy['Bid_AO'] = ta.momentum.ao(df_usd_jpy.Bid_High, df_usd_jpy.Bid_Low)
#df_usd_jpy['Ask_AO'] = ta.momentum.ao(df_usd_jpy.Ask_High, df_usd_jpy.Ask_Low)

''' Original MACD calc
df_usd_jpy['Bid_MACD'] = ta.trend.macd(df_usd_jpy.Bid_Close, n_fast=12, n_slow=26)
df_usd_jpy['Ask_MACD'] = ta.trend.macd(df_usd_jpy.Ask_Close, n_fast=12, n_slow=26)


df_usd_jpy['Bid_MACD_Sig'] = ta.trend.macd_signal(df_usd_jpy.Bid_Close, n_fast=12, n_slow=26, n_sign=9)
df_usd_jpy['Ask_MACD_Sig'] = ta.trend.macd_signal(df_usd_jpy.Ask_Close, n_fast=12, n_slow=26, n_sign=9)
'''

df_usd_jpy['Prev_Bid_Close'] = df_usd_jpy['Bid_Close'].shift(-1)
df_usd_jpy['Prev_Ask_Close'] = df_usd_jpy['Ask_Close'].shift(-1)

df_usd_jpy["Bid_ToT_Trend"] = df_usd_jpy.apply(ToT_Trend, axis = 1)

df_usd_jpy["Bid_Trend"] = df_usd_jpy.apply(Trend, axis = 1)

df_usd_jpy.rename(columns={'BidAvg': 'Bid', 'AskAvg': 'Ask'}, inplace=True)

asset_name = 'df_usd_jpy_base'+timestamp

#Write base table to postgress database.
to_pg(df_usd_jpy, asset_name)


for sma in slow_values:
    #df_usd_jpy['Bid_SSMA' + str(sma)] = df_usd_jpy['Bid_Close'].rolling(sma).mean()

    for fma in fast_values:

        #df_usd_jpy['Bid_FSMA' + str(fma_window)] = df_usd_jpy['Bid_Close'].rolling(fma_window).mean()
        if fma < sma:
            df_usd_jpy['Bid_MACD_S' + str(sma) + 'F' + str(fma)] = ta.trend.macd(df_usd_jpy.Bid_Close, n_fast=fma, n_slow=sma)
            df_usd_jpy['Bid_MACD_Sig_S' + str(sma) + 'F' + str(fma)] = ta.trend.macd_signal(df_usd_jpy.Bid_Close, n_fast=fma, n_slow=sma, n_sign=9)


df_usd_jpy.dropna(inplace = True)

arr_ip = [tuple(i) for i in df_usd_jpy.values]

dtyp = np.dtype(list(zip(df_usd_jpy.dtypes.index, df_usd_jpy.dtypes)))

global arr_analysis

arr_analysis = np.array(arr_ip, dtype=dtyp)

report = Reporting()
master_act = Master_Activity()

arr_iterator = 0

#(arr_analysis[arr_analysis_iter]['Bid_SSMA' + str(sma)] < arr_analysis[arr_analysis_iter][
                    #'Bid_FSMA' + str(fma)] and trading_test.buy_status == 1):

#(arr_analysis[arr_analysis_iter]['Bid_SSMA' + str(sma)] > arr_analysis[arr_analysis_iter][
#                    'Bid_FSMA' + str(fma)] and trading_test.sell_status == 1):

for sma in slow_values:

    stop_loss = [None]

    stop_loss_iter = 0

    #fma_val = [int(sma * fma) for fma in fast_values]

    for z in range(len(stop_loss)):

        for fma in fast_values:
            if fma < sma:
                trading_test = Account('test' + str(arr_iterator), 50000, 0)
                #trading_test = Account('test' + str(arr_iterator) + "Bid_SSMA" + str(sma) + "Bid_FSMA" + str(fma) + "SL" + str(
                #    stop_loss[stop_loss_iter]), 50000, 0)
                arr_analysis_iter = 0

                for x in np.ndenumerate(arr_analysis):

                    if (arr_analysis[arr_analysis_iter]['Bid_MACD_S' + str(sma) + 'F' + str(fma)] <
                        arr_analysis[arr_analysis_iter]['Bid_MACD_Sig_S' + str(sma) + 'F' + str(fma)]
                        and trading_test.buy_status == 1):
                        trading_test.buy(1000, arr_analysis_iter, sma, fma)

                    elif (arr_analysis[arr_analysis_iter]['Bid_MACD_S' + str(sma) + 'F' + str(fma)] >
                          arr_analysis[arr_analysis_iter]['Bid_MACD_Sig_S' + str(sma) + 'F' + str(fma)]
                          and trading_test.sell_status == 1):
                        trading_test.sell(arr_analysis_iter, stop_loss[stop_loss_iter])

                    arr_analysis_iter += 1

                master_act.app_activity()

                report.populate_results()
                arr_iterator += 1

            stop_loss_iter += 1

master_act.save_activity()

report.create_final_report()
