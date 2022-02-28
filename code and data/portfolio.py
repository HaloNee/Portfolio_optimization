import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from efficient_frontier import EfficientFrontier


class Stock(object):
    # Object that contains information about a stock.
    def __init__(self, data):
        self.data = data


class Portfolio(object):
    # Object that contains information about a investment portfolio.

    def __init__(self):
        self.portfolio = pd.DataFrame()
        self.stocks = {}
        self.data = pd.DataFrame()
        self.risk_free_rate = 0.03
        self.freq = 252

        self.ef = None

    @property
    def freq(self):
        return self.__freq

    @freq.setter
    def freq(self, val):
        self.__freq = val
        # self._update()

    @property
    def risk_free_rate(self):
        return self.__risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, val):
        self.__risk_free_rate = val
        

    def add_stock(self, stock):

        # also add stock data of stock to the dataframe
        self._add_stock_data(stock.data)

    def _add_stock_data(self, df):
        # loop over columns in given dataframe
        for datacol in df.columns:
            cols = len(self.data.columns)
            self.data.insert(loc=cols, column=datacol, value=df[datacol].values)
        # set index correctly
        self.data.set_index(df.index.values, inplace=True)
        # set index name:
        self.data.index.rename("Date", inplace=True)

    def comp_daily_returns(self):
        # Computes the daily returns (percentage change) of all stocks in the portfolio.

        # return daily_returns(self.data)
        results = self.data.pct_change().dropna(how="all").replace([np.inf, -np.inf], np.nan)
        return results

    def comp_mean_returns(self, freq=252):
        # Computes the mean returns based on historical stock price data.

        # return historical_mean_return(self.data, freq=freq)
        return self.comp_daily_returns().mean() * freq

    def comp_stock_volatility(self, freq=252):
        # Computes the Volatilities of all the stocks individually

        return self.comp_daily_returns().std() * np.sqrt(freq)


    def comp_cov(self):
       # Compute  the covariance matrix of the portfolio.

        daily_returns = self.comp_daily_returns()
        return daily_returns.cov()

    # optimising the investments with the efficient frontier class

    def _get_ef(self):
        """If self.ef does not exist, create and return an instance of 
        efficient_frontier.EfficientFrontier, else, return the existing instance.
        """
        if self.ef is None:
            # create instance of EfficientFrontier
            self.ef = EfficientFrontier(
                self.comp_mean_returns(freq=1),
                self.comp_cov(),
                risk_free_rate=self.risk_free_rate,
                freq=self.freq,
            )
        return self.ef

    def ef_maximum_sharpe_ratio(self, verbose=False):
        # Finds the portfolio with the maximum Sharpe Ratio

        ef = self._get_ef()
        # perform optimisation
        opt_weights = ef.maximum_sharpe_ratio()
        return opt_weights

    def ef_efficient_frontier(self, targets=None):
        ef = self._get_ef()
        # perform optimisation
        efrontier = ef.efficient_frontier(targets)
        return efrontier

    def ef_plot_efrontier(self):

        # Plots the Efficient Frontier
        ef = self._get_ef()
        # plot efficient frontier
        ef.plot_efrontier()

    def ef_plot_optimal_portfolios(self):
        ef = self._get_ef()
        # plot the optimal_portfolios
        ef.plot_optimal_portfolios()

    def plot_stocks(self, freq=252):

        # Plots the Expected annual Returns over annual Volatility of
        # the stocks of the portfolio.

        # annual mean returns of all stocks
        stock_returns = self.comp_mean_returns(freq=freq)
        stock_volatility = self.comp_stock_volatility(freq=freq)
        # adding stocks of the portfolio to the plot
        # plot stocks individually:
        plt.scatter(stock_volatility, stock_returns, marker="o", s=50, label="Stocks")
        # adding text to stocks in plot:
        for i, txt in enumerate(stock_returns.index):
            plt.annotate(
                txt,
                (stock_volatility[i], stock_returns[i]),
                xytext=(10, 0),
                textcoords="offset points",
                label=i,
            )
            plt.legend()


def _generate_pf_allocation(names=None, data=None):

    names = data.columns
    weights = [1.0 / len(names) for i in range(len(names))]
    return pd.DataFrame({"Allocation": weights, "Name": names})


def build_portfolio(data, pf_allocation=None, datacolumns=["Adj. Close"]):

    if pf_allocation is None:
        pf_allocation = _generate_pf_allocation(data=data)

    # building portfolio:
    pf = Portfolio()
    for i in range(len(pf_allocation)):
        # get name of stock
        name = pf_allocation.loc[i].Name

        # extract data column(s) of said stock
        stock_data = data.loc[:, [name]].copy(deep=True)

        # create Stock instance and add it to portfolio
        pf.add_stock(Stock(data=stock_data))

    return pf
