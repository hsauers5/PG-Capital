#%%
from cProfile import label
from calendar import month
import datetime as dt
from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

start_date = dt.datetime(2016,3,1)
end_date = dt.datetime(2021,12,8)
days_between = end_date - start_date
rf = 0.01


df = pd.read_excel('Data\price_data.xlsx')
price_data = df.set_index('Date')
tickers = price_data.columns.tolist()
# Get daily percentage change for each ticker
tickers_pct_change = price_data.pct_change()
# Get yearly standard deviation for each ticker
ann_sd = tickers_pct_change.apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
# Get yearly return for each ticker
yrly_return = ((price_data.pct_change().mean()+1)**252 - 1)

# Create a covariance and correlation matrix for all assets
cov_matrix = tickers_pct_change.apply(lambda x: x*np.sqrt(250)).cov()
corr_matrix = tickers_pct_change.corr()

# Get S&P 500 Portfolio Performance
stocks_data = price_data['SPY']
stocks_pct_change = stocks_data.pct_change()
stock_ann_sd = stocks_pct_change.std() * np.sqrt(250)
stock_yrly_return = ((stocks_data.pct_change().mean()+1)**252 - 1)
stocks_sharpe = ((stock_yrly_return-rf)/stock_ann_sd)
downside_deviation = stocks_pct_change.clip(upper=0).std()
annual_dd = abs(downside_deviation * np.sqrt(250))
stock_sortino = ((stock_yrly_return-rf)/annual_dd)
stocks_rolling_max = stocks_data.cummax()
stocks_daily_drawdown = (stocks_data/stocks_rolling_max) - 1.0
max_drawdown = abs(stocks_daily_drawdown.cummin().min())

stocks_calmar = ((stock_yrly_return-rf)/max_drawdown)

stock_data = {  'Return':stock_yrly_return , 'Standard Deviation': stock_ann_sd, 
                'Sharpe Ratio': stocks_sharpe, 'Downside Deviation': annual_dd, 
                'Sortino Ratio': stock_sortino, 'Maximum Drawdown': max_drawdown,
                'Calmar Ratio': stocks_calmar}

stock_pf_performance = pd.DataFrame(stock_data, index=[0])
print(stock_pf_performance)

# Choose which optimizer to run: A=Sharpe Ratio ; B=Calmar Ratio ; C=Sortino Ratio
# -------------------------------------------------------------------------------
portfolio_optimizer = 'C'
# -------------------------------------------------------------------------------

def optimize_portfolio_sharpe_ratio(tickers):
    num_of_tickers = len(tickers)
    pf_num = 100000

    rf= 0.01
    # Randomly Assign Weights to Each Asset and get the return and vol per portfolio
    ticker_weights = []
    all_pf_returns = []
    all_pf_sharpe = []
    all_pf_std = []
    for i in range(pf_num):
        # Portfolio weights
        weights = np.random.random(num_of_tickers)
        weights = weights/np.sum(weights)
        ticker_weights.append(weights)
        # Portfolio return
        pf_return = np.dot(weights, yrly_return)
        all_pf_returns.append(pf_return)
        # Portfolio sharpe ratio
        pf_var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        pf_sd = np.sqrt(pf_var)
        all_pf_std.append(pf_sd)
        all_pf_sharpe.append((pf_return-rf)/pf_sd)

    data = {'Return':all_pf_returns , 'Standard Deviation': all_pf_std, 'Sharpe Ratio': all_pf_sharpe}

    for counter, symbol in enumerate(tickers):
        data[symbol + ' Weight'] = [w[counter] for w in ticker_weights]


    portfolios = pd.DataFrame(data)
    # Minimum Shapre Ratio Portfolio
    optimal_risk_port = portfolios.iloc[portfolios['Sharpe Ratio'].idxmax()]
    print(optimal_risk_port)


    plt.scatter(y=portfolios['Return'],x=portfolios['Standard Deviation'], marker='*', s=10, alpha=0.3)
    plt.scatter(y=optimal_risk_port['Return'], x=optimal_risk_port['Standard Deviation'], color='r', marker='*', s=400, label='Max Sharpe Ratio')
    plt.scatter(y=stock_data['Return'], x=stock_data['Standard Deviation'], color='b', marker='*', s=400, label='S&P 500 Portfolio')
    plt.title('Simulated Portfolio Optimization based on Sharpe Ratio')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Returns')
    plt.tight_layout()
    plt.legend()
    plt.show()


def optimize_portfolio_sortino(tickers):
    num_of_tickers = len(tickers)
    pf_num = 100000

    rf= 0.01


    # Randomly Assign Weights to Each Asset and get the return and vol per portfolio
    ticker_weights = []
    all_pf_returns = []
    all_pf_dd = []
    for i in range(pf_num):
        # Portfolio weights
        weights = np.random.random(num_of_tickers)
        weights = weights/np.sum(weights)
        ticker_weights.append(weights)
        # Portfolio annual and daily return
        pf_return = np.dot(weights, yrly_return)
        all_pf_returns.append(pf_return)
        # Portfolio sharpe ratio
        pf_daily_returns = tickers_pct_change.mul(weights).sum(axis=1)
        downside_deviation = pf_daily_returns.clip(upper=0).std()
        annual_dd = abs(downside_deviation * np.sqrt(250))
        all_pf_dd.append(annual_dd)

    all_pf_sortino = []
    for i in range(len(all_pf_returns)):
        # Portfolio Sortino Ratio
        pf_sortino = ((all_pf_returns[i]-rf)/all_pf_dd[i])
        all_pf_sortino.append(pf_sortino)

    data = {'Returns':all_pf_returns , 'Downside Deviation': all_pf_dd, 'Sortino Ratio': all_pf_sortino}

    for counter, symbol in enumerate(tickers):
        data[symbol + ' Weight'] = [w[counter] for w in ticker_weights]

    portfolios = pd.DataFrame(data)
    # Minimum Calmar Portfolio
    max_sortino_pf = portfolios.iloc[portfolios['Sortino Ratio'].idxmax()]
    print(max_sortino_pf)

    plt.scatter(y=portfolios['Returns'], x=portfolios['Downside Deviation'],marker='o', s=10, alpha=0.3)
    plt.scatter(y=max_sortino_pf['Returns'], x=max_sortino_pf['Downside Deviation'], color='r', marker='*', s=400, label='Max Sortino Ratio')
    plt.scatter(y=stock_data['Return'], x=stock_data['Downside Deviation'], color='b', marker='*', s=400, label='S&P 500 Portfolio')
    plt.title('Simulated Portfolio Optimization based on Sortino Ratio')
    plt.xlabel('Downside Deviation')
    plt.ylabel('Returns')
    plt.tight_layout()
    plt.legend()
    plt.show()


def optimize_max_drawdown(tickers):
    num_of_tickers = len(tickers)
    pf_num = 10000  

    # Randomly Assign Weights to Each Asset and get the return and vol per portfolio
    ticker_weights = []
    all_pf_returns = []
    all_pf_max_dd = []
    for i in range(pf_num):
        # Portfolio weights
        weights = np.random.random(num_of_tickers)
        weights = weights/np.sum(weights)
        ticker_weights.append(weights)
        # Portfolio annual and daily return
        pf_return = np.dot(weights, yrly_return)
        all_pf_returns.append(pf_return)
        # Portfolio Calmar Ratio
        pf_daily_price = price_data.mul(weights).sum(axis=1)
        rolling_max = pf_daily_price.cummax()
        daily_drawdown = (pf_daily_price/rolling_max) - 1.0
        max_drawdown = abs(daily_drawdown.cummin().min())
        all_pf_max_dd.append(max_drawdown)


    all_pf_calmar = []
    for i in range(len(all_pf_returns)):
        # Portfolio Sortino Ratio
        pf_sortino = ((all_pf_returns[i]-rf)/all_pf_max_dd[i])
        all_pf_calmar.append(pf_sortino)
        
    data = {'Returns':all_pf_returns, 'Maximum Drawdown': all_pf_max_dd, 'Calmar Ratio': all_pf_calmar}

    for counter, symbol in enumerate(tickers):
        data[symbol + ' Weight'] = [w[counter] for w in ticker_weights]

    portfolios = pd.DataFrame(data)
    # Maximum Calmar Portfolio
    max_calmar_portfolio = portfolios.iloc[portfolios['Calmar Ratio'].idxmax()]
    # Maximum Calmar Portfolio with a MDD below 50%
    below_stock_mdd = portfolios.loc[(portfolios['Maximum Drawdown'] < stock_data['Maximum Drawdown']+.1)]
    best_calmar_stock_comp = below_stock_mdd.loc[(below_stock_mdd['Returns'].idxmax())]
    print(max_calmar_portfolio)
    print(best_calmar_stock_comp)

    plt.scatter(y=portfolios['Returns'], x=portfolios['Maximum Drawdown'],marker='o', s=10, alpha=0.3)
    plt.scatter(y=max_calmar_portfolio['Returns'], x=max_calmar_portfolio['Maximum Drawdown'], color='r', marker='*', s=400, label='Max Calmar Ratio')
    plt.scatter(y=best_calmar_stock_comp['Returns'], x=best_calmar_stock_comp['Maximum Drawdown'], color='g', marker='*', s=400, label='Below 50% MDD Max Calmar')
    plt.scatter(y=stock_data['Return'], x=stock_data['Maximum Drawdown'], color='b', marker='*', s=400, label='S&P 500 Portfolio')
    plt.title('Simulated Portfolio Optimization based on Calmar Ratio')
    plt.xlabel('Maximum Drawdown')
    plt.ylabel('Returns')
    plt.tight_layout()
    plt.legend()
    plt.show()
    

if portfolio_optimizer == 'A':
    optimize_portfolio_sharpe_ratio(tickers)
elif portfolio_optimizer == 'B':
    optimize_portfolio_sortino(tickers)
elif portfolio_optimizer == 'C':
        optimize_max_drawdown(tickers)


