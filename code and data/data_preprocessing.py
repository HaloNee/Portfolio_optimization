import numpy as np
import pandas as pd

import os

train_dir = 'train/'    # train data dir
test_dir = 'sample_test/'     # test data dir
data_dir = 'data/'    # ma data dir

filesList = os.listdir('./train/')
stock_list = []
for item in filesList:
    if '.csv' in item:
        stock_list.append(item[:-4])
# print(stock_list)

close_data = pd.read_csv(train_dir + 'B1K' + '.csv', index_col=0,
                         parse_dates=True)  # read train data
close_data = close_data['Close']
for stock in stock_list:
    # print(stock)

    data_train = pd.read_csv(train_dir + stock + '.csv', index_col=0,
                             parse_dates=True)  # read train data
    data2_test = pd.read_csv(test_dir + stock + '.csv', index_col=0,
                             parse_dates=True)  # read test data
    
    if len(data_train) <= 252 * 3:
        print(stock)

    # Filter out stocks that listed later than 2015.01.01
    if len(data_train) > 252 * 3:

        data = pd.concat([data_train, data2_test])  # merge data

        close_data = pd.merge(close_data, data['Close'], how='outer', left_index=True, right_index=True, suffixes=('', str(stock)))


close_data = close_data.drop('Close', axis=1).fillna(0)
close_data.columns = close_data.columns.str.replace('Close', '')

# get the close price from 2015-01 to 2017-12
close_data_1801 = close_data['2015-01':'2017-12']
close_data_1801.to_csv(data_dir + 'close_data_201801.csv')

