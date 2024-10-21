#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pytz
import time
import json
import pyotp
import aiohttp
import asyncio
import warnings
import requests
import nest_asyncio

import numpy as np
import pandas as pd
import xlwings as xw
import pandas_ta as ta

from io import StringIO
from threading import Thread
from datetime import datetime
from SmartApi import SmartConnect
from datetime import datetime,timedelta
from IPython.display import clear_output
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress all warnings
warnings.filterwarnings('ignore')
nest_asyncio.apply()


# In[2]:


def login_with_credentials(userid, password, twofa):
    reqsession = requests.Session()
    r = reqsession.post('https://kite.zerodha.com/api/login', data={
        "user_id": userid,
        "password": password,'type':'user_id'
    })

    r2 = reqsession.post('https://kite.zerodha.com/api/twofa', data={
        "request_id": r.json()['data']['request_id'],
        "twofa_value": twofa,
        "user_id": r.json()['data']['user_id']
    })
    if 'enctoken' in reqsession.cookies.get_dict():
        enctoken = reqsession.cookies.get_dict()['enctoken']
    else:
        enctoken = ''
    return enctoken


# In[3]:


def get_tokens(header,angel_nfo_data):
    session=requests.session()
    resp=session.get("https://api.kite.trade/instruments",headers=header)
    csv_data = StringIO(resp.text)
    df = pd.read_csv(csv_data)
    df_nfo=df[df.segment=='NFO-OPT'].reset_index(drop=True)

    df_nfo=df_nfo[(~df_nfo.name.isin(['BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTY', 'NIFTYNXT50']))]
    df_nfo=df_nfo[(df_nfo.expiry==sorted(df_nfo.expiry)[0])]
    
    df_nfo=df_nfo.reset_index(drop=True)

#     angel_nfo_data=pd.read_csv('nfo_data.csv')
    df_proc=df_nfo[df_nfo.exchange_token.isin(angel_nfo_data.symbolToken.unique())].reset_index(drop=True)
    
    return df_proc


# In[4]:


def get_cpr(df):
    df_test=df[df.time.dt.date==(datetime.now()-timedelta(days=1)).date()]
    if df_test.empty:
        return[0,0,0]
    
    high=df_test.high.max()
    low=df_test.low.min()
    close=df_test.close.iloc[-1]

    pp=(high+close+low)/3
    bc=(low+high)/2
    tc=(pp-bc)+pp
    return [tc,pp,bc]


# In[5]:


# Function to filter stocks based on the provided conditions on the latest data
def filter_latest_stock_data(df):
    """
    Filter the latest row for each stock (token) based on the provided conditions:
    - OI crossing below OI_20SMA
    - Volume is highest of the day
    - Close is higher than VWAP
    - Fisher signal is higher than fisher trigger
    - RSI is higher than 60
    - Close crossing BC (Bottom Central Pivot) upwards
    
    df: pandas DataFrame containing stock data, with columns:
        - 'token', 'OI', 'OI_20SMA', 'Volume', 'Close', 'VWAP'
        - 'FISHERT_signal', 'FISHERT_trigger', 'RSI', 'BC', 'time'
    """

    # Step 1: Sort the DataFrame by 'time' within each token group
    df = df.sort_values(['token', 'time'])

    # Step 2: Get the latest row for each token
    latest_rows = df.groupby('token').tail(1).copy()
    
    # Step 3: Get the previous row for each token by shifting and then aligning with latest rows
    previous_rows = df.groupby('token').shift(1)  # This gives all previous rows
    previous_rows = previous_rows.loc[latest_rows.index]  # Align previous rows with the latest row indices

    # Step 4: Calculate conditions using the latest and previous rows

    # 1. OI crossing below OI_20SMA (check if the latest row crossed below the SMA using previous row)
    latest_rows['oi_cross_down'] = (
        previous_rows['oi'] >= previous_rows['oi_20sma']
    ) & (latest_rows['oi'] < latest_rows['oi_20sma'])

    # 2. Volume is highest of the day for the stock
    latest_rows['highest_volume'] = latest_rows['volume'].values == df.groupby('token')['volume'].max().values

    # 3. Close is higher than VWAP
    latest_rows['close_above_vwap'] = latest_rows['close'] > latest_rows['vwap']

    # 4. Fisher signal is higher than Fisher trigger
    latest_rows['fisher_signal_above_trigger'] = latest_rows['FISHERT_signal'] > latest_rows['FISHERT_trigger']

    # 5. RSI is higher than 60
    latest_rows['rsi_above_60'] = latest_rows['RSI_14'] > 60

    # 6. Close crossing BC upwards (check if it crossed upwards using previous close and current close)
    latest_rows['close_cross_bc_up'] = (
        previous_rows['close'] <= previous_rows['bc']
    ) & (latest_rows['close'] > latest_rows['bc'])

    # Step 5: Filter the rows where all conditions are met
    filtered_stocks = latest_rows[
        latest_rows['oi_cross_down'] &
        latest_rows['highest_volume'] &
        latest_rows['close_above_vwap'] &
        latest_rows['fisher_signal_above_trigger'] &
        latest_rows['rsi_above_60'] &
        latest_rows['close_cross_bc_up']
    ]

    # Return the filtered DataFrame with stocks that meet the conditions
    return filtered_stocks,latest_rows




# In[10]:


def read_user_credentials(user_data):
    users = []
    for index,row in user_data.iterrows():
        TOKEN = row['token']
        try:
            factor2 = pyotp.TOTP(TOKEN).now()
        except:
            print('Error in TOTP')
        enc=login_with_credentials(row['user'],row['password'],factor2)
        if enc!='':
            users.append({'user': row['user'], 'enc': enc})
        else:
            u=row['user']
            print(f'{u} could not be logged in. Check credentials...')
    return users

# Asynchronous fetch function
async def fetch_data(session, token, exch, param, tf, header):
    url = f'https://kite.zerodha.com/oms/instruments/historical/{token}/{tf}'
    
    try:
        async with session.get(url, headers=header, params=param) as response:
            if response.status == 200:
                data = await response.json()
                if 'data' in data and "candles" in data['data'] and len(data['data']["candles"]) > 0:
                    df = pd.DataFrame(data['data']["candles"], columns=['time', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['token'] = token
                    
                    # Data processing steps
                    df_mid = df.iloc[:-1]
                    df_mid['time'] = pd.to_datetime(df_mid.time)
                    df_mid.set_index("time", inplace=True)
                    df_mid['vwap'] = ta.vwap(df_mid.high, df_mid.low, df_mid.close, df_mid.volume).round(2)
                    df_mid = pd.concat([df_mid, ta.fisher(df_mid.high, df_mid.low, 9).round(2), ta.rsi(df_mid.close).round(2)], axis=1)
                    df_mid['oi_20sma'] = df_mid.oi.rolling(20).mean()
                    df_mid = df_mid.reset_index()
                    df_mid['tc'], df_mid['pp'], df_mid['bc'] = get_cpr(df_mid)
                    df_mid1 = df_mid[df_mid.time.dt.date == (datetime.now()).date()]
                    df_mid = pd.concat([df_mid[df_mid.time.dt.date == (datetime.now() - timedelta(days=1)).date()].iloc[-1:], df_mid1])
                    
                    # Update the global success counter
                    total_success_counter[0] += 1  # Increment the total success counter
                    return df_mid.reset_index(drop=True)
                else:
                    return pd.DataFrame()
            elif response.status == 429:  # Rate limit error
                print(f"Rate limit hit for token {token}. Retrying after delay...")
                await asyncio.sleep(2)  # Short sleep before retrying
                return await fetch_data(session, token, exch, param, tf, header)  # Retry the same request
            else:
                print(f'Failed to fetch data. Status: {response.status} for token {token}')
                return pd.DataFrame()  # Return empty DataFrame for non-200 responses
    except Exception as e:
        print(f"Error fetching token {token}: {e}")
        return pd.DataFrame()

# Asynchronous function to fetch data for multiple tokens with a specific user
async def fetch_data_for_user(user, tokens, exch, param, tf, batch_size=20, delay=2):
    results = []
    
    header = {
    "Authorization": f'enctoken {user["enc"]}',  # Replace with your actual enctoken
    "Content-Type": "application/json"}
    
    start = datetime.now()
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i + batch_size]
            tasks = [fetch_data(session, token, exch, param, tf, header) for token in batch_tokens]
            
            # Gather results for the current batch
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Print the total number of tokens processed so far
            clear_output(wait=True)
            print(f"Total tokens processed so far: {total_success_counter[0]} out of {len(tokens)}")

            # Throttle by waiting for some time after each batch
            if i + batch_size < len(tokens):
                end = datetime.now()
                print(f"Waiting for {delay} seconds before processing the next batch for user {user['user']}...")
                print(f"Time elapsed for user {user['user']}: {end - start}")
                await asyncio.sleep(delay)

    # Concatenate all DataFrames together
    final_df = pd.concat(results, ignore_index=True)
    
    return final_df

# Function to divide tokens among users
def divide_tokens_among_users(tokens, users):
    num_users = len(users)
    chunk_size = len(tokens) // num_users
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

# Main function to read users and fetch data
def fetch_data_with_multiple_users(users, df_proc, exch, param, tf):
    

    tokens = list(df_proc.instrument_token.unique())
    # Divide tokens among users
    token_batches = divide_tokens_among_users(tokens, users)

    # Run the fetch process in parallel for each user
    loop = asyncio.get_event_loop()
    tasks = [
        fetch_data_for_user(user, token_batches[i], exch, param, tf)
        for i, user in enumerate(users)
    ]
    
    # Gather all results from parallel execution
    results = loop.run_until_complete(asyncio.gather(*tasks))
    
    # Concatenate the results from all users
    all_results = pd.concat(results, ignore_index=True)
    
    return all_results



# In[11]:


# Function to check if a date is a trading day (not a weekend or holiday)
def is_trading_day(date):
    url = "https://api.upstox.com/v2/market/holidays/"

    payload={}
    headers = {
      'Accept': 'application/json'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    holiday=pd.DataFrame(response.json()['data'])
    holiday['date']=pd.to_datetime(holiday.date)
    holiday=holiday[holiday.closed_exchanges.apply(lambda x: 'NFO' in x)].reset_index(drop=True)
    nfo_holiday = [d.date() for d in holiday['date']]
    return date.weekday() < 5 and date not in nfo_holiday

# Function to get the latest trading day before the given date
def get_previous_trading_day(date):
    while not is_trading_day(date):
        date -= timedelta(days=1)
    return date

# Function to adjust 'from_date' and 'to_date'
def adjust_trading_dates():
    today = datetime.now().date()

    # Adjust 'to_date' to the latest trading day (today if trading day)
    to_date = get_previous_trading_day(today)

    # Start with 'from_date' as one day before 'to_date'
    from_date = to_date - timedelta(days=1)

    # Find the last valid trading day for 'from_date'
    from_date = get_previous_trading_day(from_date)

    return from_date, to_date



def send_message_to_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'  # Optional: if you want to format the message with Markdown
    }
    response = requests.post(url, data=payload)
    return response.json()

def send_list_to_telegram(stock_df):
    strings_list = [f"{row.Candle_time} ----> {row.Stock} {row.strike} {row.type}" for i,row in stock_df.iterrows()]
    message = "\n".join(strings_list)
    # Send the message to the bot
    response = send_message_to_telegram(message)
    if response.get("ok"):
        print("Message sent successfully!")
    else:
        print("Failed to send message", response)

        
def is_weekday_and_within_time():
    import datetime as dt
    
    current_day = datetime.today().weekday()  
    current_time = datetime.now().time()  

    # Define the time range
    start_time = dt.time(9, 19)  
    end_time = dt.time(15, 31)  
    
    # Check if it's a weekday (Monday to Friday)
    weekday = current_day < 5

    # Check if the current time is before, after, or within market hours
    before_market = current_time < start_time
    after_market = current_time > end_time
    within_time = start_time <= current_time <= end_time
        
    
    return weekday,before_market,after_market,within_time







