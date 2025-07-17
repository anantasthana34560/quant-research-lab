'''
Portfolio strategy: 
- we use monthly data over the span of a year 
- metrics for evaluation: CAGR, Max Drawdown, Sharpe, Volatility
- Basic strategy: start with an empty portfolio of m stocks. Every month, the top x performing stocks are placed into the portfolio, and the bottom x performing stocks are shorted in the portfolio. At most m long positions and x short positions can be held at a given time. 
- We also test the strategy with no shorting. 

Strategy was inspired by Mayank Rasu's Udemy course on algorithmic trading. 
'''

import matplotlib.pyplot as plt 
import copy

# OHLC = Open, High, Low, Close

def CAGR(input_df):
    df = input_df.copy()
    df['Cumulative Return'] = (1 + df['Return']).cumprod()
    n = len(df)/12 # 12-month CAGR
    cagr = (df['Cumulative Return'].iloc[-1]) ** (1 / n) - 1
    return cagr.item()

def volatility(input_df):
    df = input_df.copy()
    vol = df['Return'].std() * (12 ** 0.5)  # Annualized volatility
    return vol

def sharpe(input_df, risk_free_rate=0.025):
    df = input_df.copy()
    sharpe_ratio = (CAGR(df) - risk_free_rate) / volatility(df)  # Annualized Sharpe Ratio
    return sharpe_ratio

def max_dd(input_df):
    df = input_df.copy()
    df['Cumulative Return'] = (1 + df[''Return']).cumprod()
    df['Cumulative Roll Max'] = df['Cumulative Return'].cummax()
    df['Drawdown'] = df['Cumulative Roll Max'] - df['Cumulative Return']
    df['Drawdown Percentage'] = df['Drawdown'] / df['Cumulative Roll Max']
    max_drawdown = df['Drawdown Percentage'].max()
    return max_drawdown

import pandas as pd
import numpy as np
import yfinance as yf 
start_date = '2023-07-01'
end_date = '2024-07-01'
# intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo]
print(f'Dates: {start_date} to {end_date}')

def get_data(tickers, start=start_date, end=end_date):
    ohlc_mon = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, interval='1mo')
        df[''Return'] = df['Adj Close'].pct_change()
        df.dropna(inplace=True)
        ohlc_mon[ticker] = df
    return ohlc_mon

tickers = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'JPM', 'WMT', 'V', 'JNJ', 'HD', 'PG', 'KO']
ohlc_mon = get_data(tickers)
# print(ohlc_mon)
DJI = yf.download('^DJI', start=start_date, end=end_date, auto_adjust=False, interval='1mo')
DJI['Return'] = DJI['Adj Close'].pct_change()
DJI.dropna(inplace=True)
print(f"DJI CAGR: {CAGR(DJI):.4f}")
print(f"DJI Volatility: {volatility(DJI):.4f}")
print(f"DJI Sharpe Ratio: {sharpe(DJI):.4f}")
print(f"DJI Max Drawdown: {max_dd(DJI):.4f}")
# print(DJI)

ohlc_dict = copy.deepcopy(ohlc_mon)
return_df = pd.DataFrame()
for ticker in tickers:
    return_df[ticker] = ohlc_dict[ticker]['Return']

m = 6 # Number of stocks in portfolio
x = 3 # Number of stocks to replace in portfolio each month
y = 2 # Number of stocks to short each month

def pfolio(input_df, m, x):
    df = input_df.copy()
    monthly_ret = [0]
    portfolio = []  # Initialize an empty portfolio
    for i in range(1, len(df)):  # Adjusted to include the last month
        if len(portfolio) > 0:
            poor_performing_stocks = df.iloc[i-1][portfolio].nsmallest(len(portfolio) - x).index.values.tolist() # Get the poor performing stocks
            portfolio = [s for s in portfolio if s not in poor_performing_stocks]
        fill = m - len(portfolio) # Fill the portfolio with new stocks
        new_picks = df[[s for s in tickers if s not in portfolio]].iloc[i-1].nlargest(fill).index.values.tolist()
        portfolio.extend(new_picks)
        monthly_ret.append(df.iloc[i][portfolio].mean()) # Average returns of stocks in portfolio, assuming all are weighted equally 
        print(portfolio)
    monthly_ret_df = pd.DataFrame(monthly_ret, columns=['Return'])
    # Set the index to match the input DataFrame's index for correct date alignment
    monthly_ret_df.index = df.index[:len(monthly_ret_df)]
    return monthly_ret_df

pfolio_return = pfolio(return_df, m, x)

print(f"Portfolio returns over {len(return_df)} months:\n{pfolio_return.head(20)}")
print(f"Portfolio CAGR: {CAGR(pfolio_return):.4f}")
print(f"Portfolio Volatility: {volatility(pfolio_return):.4f}")
print(f"Portfolio Sharpe Ratio: {sharpe(pfolio_return):.4f}")
print(f"Portfolio Max Drawdown: {max_dd(pfolio_return):.4f}")

def pfolio_with_shorting(input_df, m, x, y):
    df = input_df.copy()
    monthly_ret = [0]
    portfolio = []  # Long portfolio
    short_portfolio = []  # Short portfolio
    for i in range(1, len(df)):
        # Update long portfolio as before
        if len(portfolio) > 0:
            # Pick x poorest performers to remove from long portfolio
            poor_performing_stocks = df.iloc[i-1][portfolio].nsmallest(len(portfolio) - x).index.values.tolist()
            portfolio = [s for s in portfolio if s not in poor_performing_stocks]
        fill = m - len(portfolio)
        new_picks = df[[s for s in tickers if s not in portfolio]].iloc[i-1].nlargest(fill).index.values.tolist()
        portfolio.extend(new_picks)
        # Pick y worst performers (not in long portfolio) to short
        short_picks = df[[s for s in tickers if s not in (short_portfolio+portfolio)]].iloc[i-1].nsmallest(y).index.values.tolist()
        short_portfolio = short_picks
        # Calculate returns: long + short
        long_ret = df.iloc[i][portfolio].mean() if portfolio else 0
        short_ret = -df.iloc[i][short_portfolio].mean() if short_portfolio else 0
        if i > 0:
            monthly_ret.append(long_ret + short_ret)
        print(f"Long: {portfolio}, Short: {short_portfolio}")
    monthly_ret_df = pd.DataFrame(monthly_ret, columns=['Return'])
    monthly_ret_df.index = df.index[:len(monthly_ret_df)] # Set the index so the dates match
    return monthly_ret_df

pfolio_with_short_return = pfolio_with_shorting(return_df, m, x, y)

print(f"Portfolio_with_short returns over {len(return_df)} months:\n{pfolio_with_short_return.head(20)}")
print(f"Portfolio_with_short CAGR: {CAGR(pfolio_with_short_return):.4f}")
print(f"Portfolio_with_short Volatility: {volatility(pfolio_with_short_return):.4f}")
print(f"Portfolio_with_short Sharpe Ratio: {sharpe(pfolio_with_short_return):.4f}")
print(f"Portfolio_with_short Max Drawdown: {max_dd(pfolio_with_short_return):.4f}")

plt.figure(figsize=(14, 7))
plt.plot((1 + pfolio_return['Return']).cumprod(), label='Portfolio Returns', color='blue')
plt.plot((1 + pfolio_with_short_return['Return']).cumprod(), label='Portfolio_with_short Returns', color='red')
plt.plot((1 + DJI['Return']).cumprod(), label='DJI Returns', color='orange')
plt.title('Portfolio Returns vs DJI Returns')
plt.xlabel('Months')
plt.ylabel(f'Cumulative Returns (m={m}, x={x}, y={y}) from {start_date} to {end_date}')
plt.legend()
plt.grid()
plt.show()