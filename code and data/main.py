# This is the code that execute the data preprocessing and output procedures
# all following packages should have already been installed by PyCharm

import portfolio as pf
import datetime
import pandas as pd
import numpy as np


start_date = datetime.datetime(2018, 1, 2)

bond_purchase_days = [datetime.datetime(2018, 1, 2),
                      datetime.datetime(2018, 7, 2),
                      datetime.datetime(2019, 1, 2),
                      datetime.datetime(2019, 7, 1)]

bond_mature_days = [datetime.datetime(2018, 6, 29),
                    datetime.datetime(2018, 12, 31),
                    datetime.datetime(2019, 6, 28),
                    datetime.datetime(2019, 12, 31)]
# %%
# you can write your code/functions here for carrying out your strategy


def main(current_date, data_stocks, data_test_t, stock_list, in_portfolio, open_price, bond_purchase_days):
    # input: current_date, current date
    #        data_stock, the historical data
    #        data_test_t, the test data up to time current_date
    #        stock_list, the list of stock names
    #        in_portfolio, your portfolio at the end of current_date-1 day
    # you cannot change the above input,
    # you can use the historical data and the test data up to the current date t
    # you can add other input as needed
    # output: buy_action, current_date's buy action
    #         sell_action, current_date's sell action

    # ====================================================================================
    # Your strategy implementation goes here

    # initialize the buy/sell actions
    buy_stock = {}
    sell_stock = {}
    for stock in stock_list:
        buy_stock[stock] = 0.0
        sell_stock[stock] = 0.0
    buy_bond = 0.0

    # if current_date in bond_purchase_days:
    if (current_date == start_date):
        filename = 'close_data_' + current_date.strftime('%Y%m') + '.csv'
        savename = 'opt_weights' + current_date.strftime('%Y%m') + '.csv'
        df_data = pd.read_csv('data/' + filename, index_col="Date", parse_dates=True)

        # build portfolio
        mypf = pf.build_portfolio(data=df_data)

        # Set risk free rate and frequency/time window of the portfolio
        mypf.risk_free_rate = 0.03
        mypf.freq = 252

        opt_weights = mypf.ef_maximum_sharpe_ratio()
        opt_weights.to_csv('data/' + savename,index_label='index')
        stock_value = 0.0
        for stock in opt_weights.index:
            buy_stock[stock] = int(in_portfolio['cash'] * opt_weights.loc[stock]['Allocation']
                                   / open_price[stock])
            stock_value += buy_stock[stock] * open_price[stock]            

        buy_bond = in_portfolio['cash'] - stock_value

    if (current_date in bond_purchase_days and current_date != start_date):

        buy_bond = in_portfolio['cash']

    buy_action = {'stock': buy_stock, 'bond': buy_bond}
    sell_action = {'stock': sell_stock}

    # ====================================================================================
    # Perform a check on portfolio consistency
    # You should not change this part!!!
    assert sum(open_price[stock] * buy_action['stock'][stock] for stock in stock_list) + buy_action['bond'] <= in_portfolio['cash']
    for stock in stock_list:
        assert in_portfolio['stock'][stock] + buy_action['stock'][stock] >= sell_action['stock'][stock]
    if not(current_date in bond_purchase_days):
        assert buy_action['bond'] == 0.0

    # return the portfolio at the end of the day
    return buy_action, sell_action
