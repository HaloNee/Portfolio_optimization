import pathlib
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import portfolio as pf


# read data from files:
df_data_path = pathlib.Path.cwd()/ "data" / "close_data_201801.csv"
df_data = pd.read_csv(df_data_path, index_col="Date", parse_dates=True)

# building a portfolio
mypf = pf.build_portfolio(data=df_data)


# Set risk free rate and frequency/time window of the portfolio
mypf.risk_free_rate = 0.03
mypf.freq = 252
print("mypf.risk_free_rate = {}".format(mypf.risk_free_rate))
print("mypf.freq = {}".format(mypf.freq))


# optimisation for maximum Sharpe ratio
opt_weights = mypf.ef_maximum_sharpe_ratio(verbose=True)
opt_weights = opt_weights[opt_weights['Allocation'] > 1e-6]
opt_weights.to_csv('data/' + 'opt_weights201801.csv', index_label='index')

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1)


# computing and plotting efficient frontier of pf
mypf.ef_plot_efrontier()

# adding markers to optimal solutions
mypf.ef_plot_optimal_portfolios()

# and adding the individual stocks to the plot
mypf.plot_stocks()
ax.legend(loc='best')
plt.grid(linestyle='-.')
ax.set_xlim([0, 0.7])
ax.set_ylim([-0.2, 0.9])

plt.show()
