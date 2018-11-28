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