
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import requests
import json


#function to load data from api
def custom_business_week_mean(values):
    # Filter out Saturdays
    working_days = values[values.index.dayofweek != 5]
    return working_days.mean()

#function to read stock data from Nepalipaisa.com api
def stock_dataFrame(stock_symbol,start_date='2020-01-01',weekly=False):
  """
  input : stock_symbol
            start_data set default at '2020-01-01'
            weekly set default at False
  output : dataframe of daily or weekly transactions
  """
  #print(end_date)
  today = datetime.today()
  # Calculate yesterday's date
  yesterday = today - timedelta(days=1)

  # Format yesterday's date
  formatted_yesterday = yesterday.strftime('%Y-%-m-%-d')
  print(formatted_yesterday)


  path = f'https://www.nepalipaisa.com/api/GetStockHistory?stockSymbol={stock_symbol}&fromDate={start_date}&toDate={formatted_yesterday}&pageNo=1&itemsPerPage=10000&pagePerDisplay=5&_=1686723457806'
  df = pd.read_json(path)
  theList = df['result'][0]
  df = pd.DataFrame(theList)
  #reversing the dataframe
  df = df[::-1]

  #removing 00:00:00 time
  #print(type(df['tradeDate'][0]))
  df['Date'] = pd.to_datetime(df['tradeDateString'])

  #put date as index and remove redundant date columns
  df.set_index('Date', inplace=True)
  columns_to_remove = ['tradeDate', 'tradeDateString','sn']
  df = df.drop(columns=columns_to_remove)

  new_column_names = {'maxPrice': 'High', 'minPrice': 'Low', 'closingPrice': 'Close','volume':'Volume','previousClosing':"Open"}
  df = df.rename(columns=new_column_names)

  if(weekly == True):
     weekly_df = df.resample('W').apply(custom_business_week_mean)
     df = weekly_df


  return df

def create_sequences(df, window_size=5):
    """
    Create input-output sequences for time series forecasting.
    
    Parameters:
    - df: pandas DataFrame with 'Close' column
    - window_size: number of days to use as input (default 5)
    
    Returns:
    - X: numpy array of shape (n_samples, window_size) containing input sequences
    - y: numpy array of shape (n_samples,) containing target prices
    """
    close_prices = df['Close'].values
    X = []
    y = []
    print("****************************************")
    # Create sliding windows
    for i in range(len(close_prices) - window_size):
        # Get the window of features
        window = close_prices[i:i+window_size]
        X.append(window)
        
        # Get the target (next day's close)
        target = close_prices[i+window_size]
        y.append(target)
    
    return np.array(X), np.array(y)

def inference(X):
    # Define the input JSON payload
    
    
    payload = json.dumps({"data": X.tolist()})

    # Get the deployment endpoint
    scoring_uri = "http://052b8af2-5fc7-4af2-adc6-4e7dff223ca8.eastus2.azurecontainer.io/score"
    headers = {"Content-Type": "application/json"}

    # Send request to the deployed model
    response = requests.post(scoring_uri, data=payload, headers=headers)

    # Print response
    # print("Response:", response.json())
    
    return response.json()