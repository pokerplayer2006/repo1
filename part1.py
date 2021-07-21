import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean
from scipy.linalg import sqrtm
import cvxpy as cvx

target_securities = ['AAPL', 'ABBV', 'ABC']

prices = pd.read_csv('../data/prices.csv')  # 1
prices = prices[prices['ticker'].isin(target_securities)]  # 2
prices.loc[(prices['close price'] > 10000) | (prices['close price'] < 0), 'close price'] = np.nan  # 3

prices['date'] = pd.to_datetime(prices['date'])
prices['close price'] = prices.sort_values('date').groupby('ticker')['close price'].ffill()  # 4
prices_pivot = prices.pivot_table(values='close price', index='date', columns='ticker')
prices_pivot.to_csv('../results/prices.csv')  # 5

prices_end_of_month = prices_pivot.loc[prices_pivot.groupby(prices_pivot.index.to_period('M')).apply(lambda x: x.index.max())]
returns = prices_end_of_month.pct_change()  # 6

initial_investment = 10000
returns['portfolio_return'] = returns.mean(axis=1)
returns['cumulative_return'] = (1 + returns['portfolio_return']).cumprod()  # used in max drawdown

# 7.1
returns['value'] = initial_investment * returns['cumulative_return']
returns['value'].iloc[0] = initial_investment

# 7.2
geometric_mean = gmean(1 + returns['portfolio_return'].dropna()) - 1
geometric_vol = np.exp(np.std(np.log(1 + returns['portfolio_return'])))

# 7.3
returns['max_return'] = np.fmax.accumulate(returns['cumulative_return'])
returns['drawdown'] = (returns['cumulative_return'] - returns['max_return']) / returns['max_return']

max_drawdown = np.nanmin(returns['drawdown'])
max_drawdown_end = returns['drawdown'].idxmin()
max_drawdown_start = returns.loc[returns['cumulative_return'] == returns.loc[max_drawdown_end]['max_return']].index[0]

# 8.1
ax = returns['value'].plot()
ax.set_ylabel('Portfolio Value')
ax.set_title('Time series of portfolio value')

# 8.2
cov = returns[target_securities].cov()


# 9
def optimization(returns_):
    """
    Estimates expected return and covariance based on historical data
    with objective of minimizing vol.
    
    Parameters
    ----------
    returns_ : pd.DataFrame, shape [n_periods, m_securities]
        Historical returns of securities.

    Returns
    -------
    np.array, shape [1, m_securities]
        Optimized weights for each security based on constraints.
    """

    returns_.dropna(inplace=True)

    mu = np.array(returns_.mean(), ndmin=2)
    sigma = np.array(returns_.cov(), ndmin=2)

    returns_ = np.array(returns_, ndmin=2)
    w = cvx.Variable((mu.shape[1], 1))
    g = cvx.Variable(nonneg=True)
    G = sqrtm(sigma)
    n = returns_.shape[0]

    ret = 1 / n * cvx.sum(cvx.log(1 + returns_ @ w))  # kelly
    A = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    B = np.array([[-1], [-1], [-1], [0.002], [0.002], [0.002]])

    constraints = [cvx.sum(w) == 1,  # total weights sum to 1
                   w <= 1, w * 1000 >= 0,  # long-only
                   A @ w - B >= 0,  # asset weight constraints (0.002 - 1)
                   ret >= -0.1,  # min return
                   #cvx.SOC(g, G.T @ w),  # cone; don't need since variance in objective function but can speed up / increase precision.
                   ]

    risk = g
    objective = cvx.Minimize(risk * 1000)
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver='ECOS')

    return np.array(w.value, ndmin=2).T


last_dates = returns.index[-4:]
optimal_weights = np.concatenate([
                                  optimization(returns[target_securities].loc[:date])
                                  for date in last_dates
                                  ])
