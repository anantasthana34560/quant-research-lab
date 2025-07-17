import pandas as pd
import numpy as np
import statsmodels.api as sm

### KPIs

def raw_return(input_df):
    df = input_df.copy()
    df['Cumulative Return'] = (1 + df['Return']).cumprod()

def CAGR(input_df, num_candles_per_year = 252):
    df = input_df.copy()
    df['Cumulative Return'] = (1 + df['Return']).cumprod()
    n = len(df)/(num_candles_per_year)  # Assuming num_days trading days per and 78 5-minute candles per day
    cagr = (df['Cumulative Return'].iloc[-1]) ** (1 / n) - 1
    return cagr.item()

def avg_return(input_df, num_candles_per_year = 252):
    return input_df['Return'].mean() * num_candles_per_year  # Annualized average return

def volatility(input_df, num_candles_per_year = 252):
    df = input_df.copy()
    vol = df['Return'].std() * ((num_candles_per_year) ** 0.5)  # Annualized volatility
    return vol

def sharpe(input_df, risk_free_rate=0.025, num_candles_per_year = 252):
    df = input_df.copy()
    if volatility(df) == 0:
        return pd.Series([float('nan') for _ in range(len(df))])
    sharpe_ratio = (avg_return(df, num_candles_per_year) - risk_free_rate) / volatility(df, num_candles_per_year)  # Annualized Sharpe Ratio
    return sharpe_ratio

def raw_sharpe(input_df, risk_free_rate=0.025):
    df = input_df.copy()
    if df['Return'].std() == 0:
        return pd.Series([float('nan') for _ in range(len(df))])
    raw_sharpe_ratio = (df['Return'].mean() - risk_free_rate) / df['Return'].std()  # Raw Sharpe Ratio
    return raw_sharpe_ratio

def max_dd(input_df):
    df = input_df.copy()
    df['Cumulative Return'] = (1 + df['Return']).cumprod()
    df['Cumulative Roll Max'] = df['Cumulative Return'].cummax()
    df['Drawdown'] = df['Cumulative Roll Max'] - df['Cumulative Return']
    df['Drawdown Percentage'] = df['Drawdown'] / df['Cumulative Roll Max']
    max_drawdown = df['Drawdown Percentage'].max()
    return max_drawdown

### TIs 

def ATR(input_df, period=20):
    df = input_df.copy()
    df['High-Low'] = df['High'] - df['Low']
    df['High-Prev Close'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-Prev Close'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-Prev Close', 'Low-Prev Close']].max(axis=1, skipna=True)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df['ATR']

def OBV(input_df):
    df = input_df.copy()
    df['Return'] = df['Adj Close'].pct_change()
    df['Direction'] = np.where(df['Return']>=0,1,-1)
    df['Direction'][0]=0
    df['Adj Volume'] = df['Volume'] * df['Direction']
    df['OBV'] = df['Adj Volume'].cumsum()
    return df['OBV']

### Misc 

def slope(ser, n):
    '''calculates slope of n consecutive points in series ser'''
    slopes = [0 for i in range(n-1)]
    for i in range(n, len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min()) / (y.max() - y.min())
        x_scaled = (x - x.min()) / (x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)