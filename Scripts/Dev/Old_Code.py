#Old Trading Test code that didn't have SMA, FMA and stop loss split out
#trading_test = Account('test' + str(arr_iterator) + "Bid_SSMA" + str(sma) + "Bid_FSMA" + str(fma) + "SL" + str(
                #    stop_loss[stop_loss_iter]), 50000, 0)

#MACD Trading Test conditions
if (arr_analysis[arr_analysis_iter]['Bid_MACD_S' + str(sma) + 'F' + str(fma)] >
        arr_analysis[arr_analysis_iter]['Bid_MACD_Sig_S' + str(sma) + 'F' + str(fma)]
        and trading_test.buy_status == 1):

    trading_test.buy(1000, arr_analysis_iter, sma, fma)

elif (arr_analysis[arr_analysis_iter]['Bid_MACD_S' + str(sma) + 'F' + str(fma)] <
      arr_analysis[arr_analysis_iter]['Bid_MACD_Sig_S' + str(sma) + 'F' + str(fma)]
      and trading_test.sell_status == 1):

    trading_test.sell(arr_analysis_iter, stop_loss[stop_loss_iter])

#old MACD calculation used with MACD Trading Test condition
for sma in slow_values:
    # df_usd_jpy['Bid_SSMA' + str(sma)] = df_usd_jpy['Bid_Close'].rolling(sma).mean()

    for fma in fast_values:

        # df_usd_jpy['Bid_FSMA' + str(fma_window)] = df_usd_jpy['Bid_Close'].rolling(fma_window).mean()
        if fma < sma:
            df_usd_jpy['Bid_MACD_S' + str(sma) + 'F' + str(fma)] = ta.trend.macd(df_usd_jpy.Bid_Close, n_fast=fma,
                                                                                 n_slow=sma)
            df_usd_jpy['Bid_MACD_Sig_S' + str(sma) + 'F' + str(fma)] = ta.trend.macd_signal(df_usd_jpy.Bid_Close,
                                                                                            n_fast=fma, n_slow=sma,
                                                                                            n_sign=9)


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

print('test'+str(arr_iterator)+'SMA'+str(sma)+'FMA'+str(fma))