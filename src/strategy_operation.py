#~* utf-8

# Import custom functions and classes for preparing datasets
from generate_dataset import *

# Import libraries for handling HTTP requests and data manipulation
import requests
import numpy as np
import json
from datetime import datetime
import pandas as pd
import time

# Import libraries for technical analysis and financial data processing
from ta import add_all_ta_features
import os.path
import pandas_ta as ta

# Import custom functions and classes for preparing datasets
from generate_dataset import *

# Import tools and strategies for backtesting trading strategies
from backtesting.lib import SignalStrategy, TrailingStrategy, crossover, Strategy
from backtesting import Backtest, Strategy
from backtesting.test import SMA


# Import utility function to clean data and mathematical function
from math import exp
from ta.utils import dropna



def RSI(df, n):
    '''Calculate Relative Strength Index given a dataframe and n days'''
    return df.ta.rsi(n)

def Williams(df, n):
    '''Calculate Williams %R given a dataframe and n days'''
    return df.ta.willr(n)




class JMstrategy(Strategy):
    """ Trading strategy class using the RSI indicator and candlestick patterns for buy/sell signals. """
    #gene = idividual = input_var = fitness_evaluate_function's input
    input_var = None
    stop_lose = None

    def init(self):
        """ Initialize and prepare the strategy with necessary indicators and parameters. """
        super().init()
        # Unpack parameter for RSI and candlestick patterns
        self.parameter = self.input_var[:7] # Extract RSI lengths and Fibonacci level
        self.boolean = self.input_var[7:15]  # Extract boolean flags for pattern checking


        # Set RSI parameters
        self.rsi_length = self.parameter[0] 
        self.rsi_overbought, self.rsi_oversold = self.parameter[1], self.parameter[2]
        
        ### Candlestick Pattern ##4
        # Fibonacci retracement level used for trading calculations ( Fixed or dynamic )
        self.fib_level = self.parameter[3]

        # Williams %R
        self.williams_length = self.parameter[4]
        self.williams_overbought, self.williams_oversold = self.parameter[5], self.parameter[6]


        # Buy and Sell Rules boolean 
        self.hammer_pattern = self.boolean[0]
        self.shooting_star_pattern = self.boolean[1]
        self.bull_engulf_pattern = self.boolean[2]
        self.bear_engulf_pattern = self.boolean[3]
        self.two_green_bar_pattern = self.boolean[4]
        self.two_red_bar_pattern = self.boolean[5]
        self.simple_rsi = self.boolean[6]
        self.simple_williams = self.boolean[7]

#         value =  [ 16,          70,          30,           0.39689154,  13,
#  -26,         -82,           1,   0,           0,
#    1,   0,           1,           1,           1       ]
        # self.rsi_length, self.rsi_overbought, self.rsi_oversold, self.fib_level, self.williams_length, self.williams_overbought, self.williams_oversold, self.hammer_pattern, self.shooting_star_pattern, self.bull_engulf_pattern, self.bear_engulf_pattern, self.two_green_bar_pattern, self.two_red_bar_pattern, self.simple_rsi, self.simple_williams = value



        # Initialize RSI indicator
        self.rsi_moment = self.I(RSI, self.data.df, self.rsi_length) 
        #print("rsi_length", self.rsi_length),print("rsi_overbought", self.rsi_overbought),print("rsi_oversold", self.rsi_oversold),print("rsi_moment", self.rsi_moment)

        #Initialize Williams %R indicator
        self.williams_moment = self.I(Williams, self.data.df, int(self.williams_length))

    def next(self):
        """ Define logic to execute on each candlestick in the data set. """
        super().next()
        #* 1e6      #convert to μBTC 
        # Obtain prices from the latest candlestick
        current_price = self.data.Close[-1] #using closed price of the current candle 
        high_price = self.data.High[-1] # highest price of the current candle
        low_price = self.data.Low[-1] # lowest price of the current candle
        open_price = self.data.Open[-1] # open price of the current candle
        close_price = self.data.Close[-1] # close price of the current candle
        prev_open = self.data.Open[-2]
        prev_close = self.data.Close[-2]
    
        # Calculate Fibonacci retracement levels
        bullFib = (low_price - high_price) * self.fib_level + high_price
        bearFib = (high_price - low_price) * self.fib_level + low_price

        # Determine highest/lowest price of the current candle
        bearCandle = close_price if close_price < open_price else open_price
        bullCandle = close_price if close_price > open_price else open_price

        # Get latest RSI values for overbought and oversold conditions
        rsiOB = self.rsi_moment[-1] >= self.rsi_overbought  
        rsiOS = self.rsi_moment[-1] <= self.rsi_oversold    

        # Get latest Williams %R values for overbought and oversold conditions
        williamsOB = self.williams_moment[-1] >= self.williams_overbought
        williamsOS = self.williams_moment[-1] <= self.williams_oversold

        # Buy Signal 
        buy_signal = [] 

        
        # Check for hammer pattern rule (buy signal)
        if self.hammer_pattern:
            buy_signal.append( (bearCandle >= bullFib) and rsiOS 
                              or (bearCandle >= bullFib) and williamsOS)
        
        # # Bullish Engulfing (buy signal)
        if self.bull_engulf_pattern:    
            buy_signal.append( (close_price > prev_open) and (prev_close < prev_open) and rsiOS 
                              or (close_price > prev_open) and (prev_close < prev_open) and williamsOS)
        # # Check for Two consecutive green bars (buy signal)
        if self.two_green_bar_pattern:
            buy_signal.append( (close_price > open_price) and (prev_close > prev_open) and rsiOS 
                              or (close_price > open_price) and (prev_close > prev_open) and williamsOS)
        # Check for simple RSI
        if self.simple_rsi:
            buy_signal.append( rsiOS )
        # Check for simple William R%
        if self.simple_williams:
            buy_signal.append( williamsOS )

        if len(buy_signal) == 0:
            buy_signal = [False]



        # Sell Signal 
        sell_signal = []


        # Check for shooting star pattern rule (sell signal)        
        if self.shooting_star_pattern:
            sell_signal.append( (bullCandle <= bearFib) and rsiOB 
                               or (bullCandle <= bearFib) and williamsOB )
        # # bearish engulfing
        if self.bear_engulf_pattern:    
            sell_signal.append( (close_price < prev_open) and (prev_close > prev_open) and rsiOB 
                               or (close_price < prev_open) and (prev_close > prev_open) and williamsOB)
        # # Check for Two consecutive red bars
        if self.two_red_bar_pattern:        
            sell_signal.append( (close_price < open_price) and (prev_close < prev_open) and rsiOB 
                               or (close_price < open_price) and (prev_close < prev_open) and williamsOB)   
        # Check for simple rsi
        if self.simple_rsi:
            sell_signal.append( rsiOB ) 
        # Check for simple William R%
        if self.simple_williams:
            sell_signal.append( williamsOB )
        

        if len(sell_signal) == 0:
            sell_signal = [False]

            
        # Buy and Sell Operation    

        #no position + all buy signals are true
        if (not self.position) & all(buy_signal): 
    
            long_sl = current_price*(1-0.5) # 50% stop loss - never give up - to the moon
            self.buy(size = 1, sl=long_sl) # buy 100% of the wallet

        #if (self.position.is_long) & (min(sell_signal)):
        if (self.position.is_long) & all(sell_signal):
            #existing long + all sell signal is true
            self.position.close()
        
        # no position + all sell signals are true
        if (not self.position) & (min(sell_signal)): 
           
            short_sl = current_price*(1+0.5) # 50% stop loss - never give up - to the moon
            self.sell(size = 1, sl=short_sl) # sell 100% of the wallet


        if (self.position.is_short) & all(buy_signal):
            #existing short + all buy signal is true
            self.position.close()   



# Backtesting function
def backtesting(data_frame, gene, stop_lose, wallet=50000, commission=.05, trade_on_close=True):
    """
    Perform backtesting on a given dataset using a specified trading strategy and genetic parameters.
    
    Parameters:
    - data_frame: DataFrame containing historical market data.
    - gene: Parameters for the trading strategy, likely genetic algorithm results.
    - stop_lose: Stop loss level to limit potential losses.
    - wallet: Initial amount of money to start the backtest with.
    - commission: Trading commission per trade, expressed as a decimal.
    - trade_on_close: Boolean indicating whether trades execute at the close of a candle.

    Returns:
    - full_fitness_score: Calculated as the Sharpe ratio of the strategy.
    """
    # Create a Backtest instance with the provided market data and strategy
    back_testing = Backtest(data_frame, JMstrategy,
              cash=wallet, commission=commission,
              trade_on_close=trade_on_close)
    
    # Run the backtest with the strategy parameters and stop loss setting
    output = back_testing.run(input_var = gene, stop_lose=stop_lose)

    # Optionally display the plot and print output if the commission is 1%
    if commission == 0.01:
        back_testing.plot(superimpose=False)
        print(output)

    # Calculate the Sharpe ratio, defined as annual return divided by annual volatility
    full_fitness_score = (output.iloc[8]) / (output.iloc[9] or np.nan)  # sharpe ratio = annual return / annual volatility


    return full_fitness_score # Return the Sharpe ratio as a measure of the strategy's effectiveness




