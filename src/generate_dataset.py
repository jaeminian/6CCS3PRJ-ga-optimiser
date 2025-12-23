#~* utf-8

import pandas as pd  # Library for data manipulation and analysis
import numpy as np  # Library for numerical operations
import time  # Module for time-related tasks

from datetime import datetime  # Module to handle date and time objects
from ta import add_all_ta_features  # Function to add technical analysis features to data frames

import pandas_ta as ta  # Extension of pandas for technical analysis

import warnings  # Module to handle warnings
# Set up to ignore specific FutureWarnings from pandas to maintain clean output
warnings.simplefilter(action='ignore', category=FutureWarning)


# Utility function to convert UNIX timestamp to date string
def unix_to_date_convert(uni):
    """Converts UNIX timestamp to a date string in 'YYYY-MM-DD HH:MM:SS' format."""
    return datetime.utcfromtimestamp(uni).strftime('%Y-%m-%d %H:%M:%S')


# Utility function to convert date string to UNIX timestamp
def date_to_unix_convert(date_time): 
    """Converts date in 'YYYY-MM-DD HH:MM:SS' format to a UNIX timestamp."""
    return int(time.mktime(datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S').timetuple()))


# Function to retrieve data from a CSV file and adjust the dataframe
def Read_data(file_location= 'data/BINANCE_BTCUSDT, 120.csv'):  # SP_SPX, BINANCE_ETHUSDT, BINANCE_BTCUSDT
    """Loads market data from a CSV file and prepares the dataframe."""
    data_frame = pd.read_csv(file_location)
    data_frame.columns =  ['date','open','high','low','close','volume']
    # Convert date from seconds to milliseconds for consistency
    data_frame['date'] = data_frame['date'].apply(lambda x: x * 1000)
    return data_frame

# Function to rename columns in the dataframe
def generate_new_frame(data_frame):
    """Renames columns to standard Open, High, Low, Close format used by TA libraries."""
    data_frame.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close'}, inplace = True)
    return data_frame

# Function to save the modified dataframe to a new CSV file
def complete_data(data_frame):
    """Saves the dataframe to a CSV file."""
    data_frame.to_csv('data/market_prices.csv', index=None)

# Main block to execute the functions
if __name__ == '__main__':
    data_frame = Read_data() # Load data
    data_frame = generate_new_frame(data_frame) # Process data
    complete_data(data_frame) # Save data 
    print('Done. File saved!')  # Confirmation message



