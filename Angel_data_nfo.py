#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pytz
import time
import json
import pyotp
import requests
import numpy as np
import pandas as pd
# import xlwings as xw

from threading import Thread
from datetime import datetime,timedelta
from SmartApi import SmartConnect
from concurrent.futures import ThreadPoolExecutor, as_completed


# In[2]:


password='4524'
user_id='P854692'
api_key='dAZqTvM5'
totp_token='7KBMGSVRW3BRBTLGKTBG37QIRY'
secret_key='453aadf1-4b1d-4ebb-91df-06296c5f7edf'


# In[3]:


# login
obj=SmartConnect(api_key=api_key)
totp = pyotp.TOTP(totp_token).now()
data = obj.generateSession(user_id,password,totp)
auth_token=data['data']['jwtToken']
refreshToken= data['data']['refreshToken']
feedToken=obj.getfeedToken()
userProfile= obj.getProfile(refreshToken)


# In[4]:


headers ={
'X-PrivateKey': obj.api_key,
'Accept': obj.accept,
'X-SourceID': obj.sourceID,
'X-ClientLocalIP': obj.clientLocalIp,
'X-ClientPublicIP': obj.clientPublicIP,
'X-MACAddress': obj.clientMacAddress,
'X-UserType': obj.userType,
'Authorization': auth_token,
'Content-Type': 'application/json'
}


# In[5]:


# URL of the JSON file
url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

# Fetching the JSON data
response = requests.get(url)

# Checking if the request was successful
if response.status_code == 200:
    # Load the JSON data into a DataFrame
    df_master = pd.json_normalize(response.json())
    
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")


# In[6]:


# Function to get market data
def fetch_data(chunk,exch):
    exchangeTokens = {exch: chunk}
    result = obj.getMarketData("FULL", exchangeTokens)
    return pd.DataFrame(result['data']['fetched'])

def data_process(chunks,exch):
    # Use ThreadPoolExecutor for parallel processing
    start = datetime.now()

    df_list = []
    total_data = len(chunks)  # Total number of chunks
    completed_data = 0        # Counter to track completion

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_data, chunk,exch): chunk for chunk in chunks}

        # As tasks complete, collect the results and update the counter
        for future in as_completed(futures):
            try:
                df_list.append(future.result())
            except Exception as exc:
                print(f"Error occurred: {exc}")

            completed_data += 1
            print(f"Completed {completed_data} out of {total_data} tokens", end='\r')

    # Concatenate all the dataframes at once
    df = pd.concat(df_list, ignore_index=True)

    end = datetime.now()
    print(f"\nTime taken: {end - start}")
    return df


# In[7]:


def filter_selected_strikes_for_all_stocks(nfo_data, df_nse):
    # Create a dictionary to store selected strikes for each stock
    selected_strikes_dict = {}
    
    # Precompute LTP for all stocks to avoid repeated filtering
    stock_ltp_map = df_nse.set_index('name')['ltp'].to_dict()
    
    # Iterate over each unique stock in nfo_data
    for stock in nfo_data['name'].unique():
        # Get available strikes and sort once for the stock
        stock_data_nfo = nfo_data[nfo_data.name == stock]
        available_strikes = np.sort(stock_data_nfo['strike'].unique())
        
        # Get the LTP (Last Traded Price) for the stock (precomputed)
        ltp = stock_ltp_map[stock]
        
        # Find the ATM (At-the-Money) strike
        atm_strike = available_strikes[np.abs(available_strikes - ltp).argmin()]
        
        # Get the index of ATM strike
        atm_index = np.searchsorted(available_strikes, atm_strike)
        
        # Select 5 ITM and 5 OTM strikes using slicing
        itm_strikes = available_strikes[max(0, atm_index - 5):atm_index]
        otm_strikes = available_strikes[atm_index + 1:atm_index + 6]
        
        # Combine selected strikes
        selected_strikes = np.concatenate([itm_strikes, [atm_strike], otm_strikes])
        
        # Store the selected strikes for the stock in the dictionary
        selected_strikes_dict[stock] = selected_strikes
    
    # Apply filtering in bulk using `isin()` over a grouped mask
    filtered_nfo_data = nfo_data[
        nfo_data.apply(lambda row: row['strike'] in selected_strikes_dict.get(row['name'], []), axis=1)
    ]
    
    return filtered_nfo_data


# In[8]:


nfo_lis=df_master[(df_master.exch_seg=='NFO')&(df_master.expiry=='31OCT2024')&(df_master.instrumenttype.str.contains('OPTSTK'))]
nse_list=df_master[(df_master['exch_seg'] == 'NSE') & (df_master['name'].isin(list(nfo_lis.name.unique()))) & (df_master.symbol.str.endswith('-EQ'))]


# In[9]:


nse_token_lis=list(nse_list.token.values)
chunk_size = 50
nse_chunks = [nse_token_lis[i:i + chunk_size] for i in range(0, len(nse_token_lis), chunk_size)]
nse_data=data_process(nse_chunks,'NSE')

nfo_lis.loc[:,'strike']=nfo_lis['strike'].astype('float').astype('int')/100
nfo_data=nfo_lis.sort_values(['name','strike'])
nfo_data['type']=nfo_data.symbol.str[-2:]

df_nse=pd.merge(nse_data,nse_list[['token','name']],how='left',left_on='symbolToken',right_on='token').drop('token',axis=1)
df_nse=df_nse[['name','tradingSymbol','symbolToken','ltp','exchFeedTime', 'exchTradeTime']]

df_to_process=filter_selected_strikes_for_all_stocks(nfo_data,df_nse)

df_to_process.reset_index(drop=True,inplace=True)


# In[10]:


nfo_token_lis=list(df_to_process.token.values)
chunk_size = 50
nfo_chunks = [nfo_token_lis[i:i + chunk_size] for i in range(0, len(nfo_token_lis), chunk_size)]
nfo_data=data_process(nfo_chunks,'NFO')


# In[11]:


nfo_data['exchFeedTime']=pd.to_datetime(nfo_data['exchFeedTime'])
nfo_data['exchTradeTime']=pd.to_datetime(nfo_data['exchTradeTime'])
nfo_data['time_diff']=nfo_data['exchFeedTime']-nfo_data['exchTradeTime']
nfo_test=nfo_data[(nfo_data['time_diff']<timedelta(minutes=5))&(nfo_data.opnInterest>50000)]


# In[ ]:


# nfo_test.to_csv(r"C:\Users\priti\OneDrive\Desktop\OI Strategy builder\Experiments\Zerodha\nfo_data.csv",index=False)


# In[ ]:




