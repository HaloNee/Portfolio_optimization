import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pylab as plt
from scipy import linalg


def annualised_portfolio_quantities(weights, means, cov_matrix, risk_free_rate=0.03, freq=252):

    expected_return = np.sum(means * weights) * freq
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(freq)
    sharpe = (expected_return - risk_free_rate) / float(volatility)

    return (expected_return, volatility, sharpe)


def portfolio_volatility(weights, mean_returns, cov_matrix, risk_free_rate=0.03, freq=252):
    # """Calculates the negative Sharpe ratio of a portfolio

    return annualised_portfolio_quantities(weights, mean_returns, cov_matrix)[1]


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.03, freq=252):
    # Calculates the negative Sharpe ratio of a portfolio

    return -annualised_portfolio_quantities(weights, mean_returns, cov_matrix)[2]


def portfolio_return(weights, mean_returns, cov_matrix, risk_free_rate=0.03, freq=252):
    # Calculates the expected annualised return of a portfolio

    return annualised_portfolio_quantities(weights, mean_returns, cov_matrix)[0]


class EfficientFrontier(object):

    def __init__(
        self, mean_returns, cov_matrix, risk_free_rate=0.03, freq=252, method="SLSQP"
    ):
      # instance variables
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.freq = freq
        self.method = method
        self.names = list(mean_returns.index)
        self.num_stocks = len(self.names)

        # set numerical parameters
        bound = (0, 1)
        self.bounds = tuple(bound for stock in range(self.num_stocks))
        self.x0 = np.array(self.num_stocks * [1.0 / self.num_stocks])
        self.constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        # placeholder for optimised values/weights
        self.weights = None
        self.df_weights = None
        self.efrontier = None

    def maximum_sharpe_ratio(self, save_weights=True):
        # Finds the portfolio with the maximum Sharpe Ratio

        args = (self.mean_returns.values, self.cov_matrix.values, self.risk_free_rate)
        result = sco.minimize(
            negative_sharpe_ratio,
            args=args,
            x0=self.x0,
            method=self.method,
            bounds=self.bounds,
            constraints=self.constraints,
        )
        # set optimal weights
        if save_weights:
            self.weights = result["x"]
            self.df_weights = self._dataframe_weights(self.weights)
            self.df_weights = self.df_weights[self.df_weights['Allocation'] > 1e-6]
            return self.df_weights
        else:
            # not setting instance variables, and returning array instead
            # of pandas.DataFrame
            return result["x"]

    def efficient_return(self, target, save_weights=True):
        # Finds the portfolio with the minimum volatility for a given target return.

        if not isinstance(target, (int, float)):
            raise ValueError("target is expected to be an integer or float.")

        args = (self.mean_returns.values, self.cov_matrix.values)
        # here we have an additional constraint:
        constraints = (
            self.constraints,
            {
                "type": "eq",
                "fun": lambda x: portfolio_return(
                    x, self.mean_returns, self.cov_matrix
                )
                - target,
            },
        )
        # optimisation
        result = sco.minimize(
            portfolio_volatility,
            args=args,
            x0=self.x0,
            method=self.method,
            bounds=self.bounds,
            constraints=constraints,
        )
        # set optimal weights
        if save_weights:
            self.weights = result["x"]
            self.df_weights = self._dataframe_weights(self.weights)
            return self.df_weights
        else:
            # not setting instance variables, and returning array instead
            # of pandas.DataFrame
            return result["x"]

    def efficient_frontier(self, targets=None):
        # Gets portfolios for a range of given target returns.

        if targets is not None and not isinstance(targets, (list, np.ndarray)):
            raise ValueError("targets is expected to be a list or numpy.ndarray")
        elif targets is None:
            # set range of target returns from the individual expected
            # returns of the stocks in the portfolio.
            min_return = self.mean_returns.min() * self.freq
            max_return = self.mean_returns.max() * self.freq
            targets = np.linspace(round(min_return, 3), round(max_return, 3), 100)
        # compute the efficient frontier
        efrontier = []
        for target in targets:
            weights = self.efficient_return(target, save_weights=False)
            efrontier.append(
                [
                    annualised_portfolio_quantities(
                        weights, self.mean_returns, self.cov_matrix, freq=self.freq
                    )[1],
                    target,
                ]
            )
        self.efrontier = np.array(efrontier)
        return self.efrontier

    def plot_efrontier(self):
        """Plots the Efficient Frontier."""
        if self.efrontier is None:
            # compute efficient frontier first
            self.efficient_frontier()
        plt.plot(
            self.efrontier[:, 0],
            self.efrontier[:, 1],
            linestyle="-.",
            color="black",
            lw=2,
            label="Efficient Frontier",
        )
        plt.legend()
        plt.title("Efficient Frontier")
        plt.xlabel("Volatility")
        plt.ylabel("Expected Return")
        plt.legend()

    def plot_optimal_portfolios(self):
        # Plots markers of the optimised portfolios for maximum_sharpe_ratio

        max_sharpe_weights = self.maximum_sharpe_ratio(save_weights=False)

        max_sharpe_vals = list(
            annualised_portfolio_quantities(
                max_sharpe_weights, self.mean_returns, self.cov_matrix, freq=self.freq
            )
        )[0:2]
        max_sharpe_vals.reverse()
        plt.scatter(
            max_sharpe_vals[0],
            max_sharpe_vals[1],
            marker="*",
            color="r",
            s=150,
            label="Max Sharpe Ratio",
        )
        plt.legend()

    def _dataframe_weights(self, weights):

        return pd.DataFrame(weights, index=self.names, columns=["Allocation"])

