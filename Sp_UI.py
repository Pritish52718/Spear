#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
from zerodha import *

def get_candle_time():
    current_time = datetime.now()
    rounded_minutes = (current_time.minute // 5) * 5
    rounded_time = current_time.replace(minute=rounded_minutes, second=0, microsecond=0)
    final_time = rounded_time - timedelta(minutes=5)
    formatted_time = final_time.strftime("%I:%M %p")
    return(formatted_time)


# Streamlit app
# st.title("SPEAR")
st.markdown("""<h1 style='text-align: center; margin-bottom: -30px;'>Spear</h1>
    <h3 style='text-align: center; margin-top: -30px; margin-bottom: 30px;'>A Stock Warrior weapon</h3>""", unsafe_allow_html=True)
# st.markdown("<h2 style='text-align: center;'>A stock warrior weapon</h2>", unsafe_allow_html=True)

col1,col2,col3=st.columns([4,2,2])
# Step 1: Upload a file
uploaded_file = col1.file_uploader("Upload the user credential file. (csv or xlsx)", type=["csv", "xlsx"])


    

BOT_TOKEN=col2.text_input("Telegram Details (optional)","Telegram Bot Token")
CHAT_ID=col2.text_input("Enter Telegram Chat ID","Telegram Chat ID",label_visibility='collapsed') 
#BOT_TOKEN = '7462509066:AAFUuNInGSnv1B7KZf_mzU3UnXDUNnRNGXE'
# CHAT_ID = '1105000514'
clock_placeholder = st.empty()
data_placeholder = st.empty()

if 'run' not in st.session_state:
    st.session_state.run = False
if 'over' not in st.session_state:
    st.session_state.over = False
if 't' not in st.session_state:
    st.session_state.t = None  # To store last candle time
if 'stock_df' not in st.session_state:
    st.session_state.stock_df = pd.DataFrame()  # To store last DataFrame


# Inject custom CSS to reduce font size
st.markdown(
    """
    <style>
    .small-font {
        font-size:15px;
        margin-bottom: 3px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use st.write() with reduced font size by applying the custom CSS class
# col3.write('<p class="small-font">Run or Stop</p>', unsafe_allow_html=True)


if col3.button('Run 🚀'):
    st.session_state.run = True

if col3.button('Stop 🛑'):
    if st.session_state.run:
        st.session_state.over = True
    st.session_state.run = False
# Reset block
if col3.button('Reset 🔄'):
    st.session_state.run = False
    st.session_state.over = False
    st.session_state.t = None
    st.session_state.stock_df = pd.DataFrame()  # Reset to an empty DataFrame

    

# while st.session_state.run:
#     st.session_state.t=get_candle_time()
    
#     clock_placeholder.markdown(f"<h4 style='text-align: center;'>Data is as of {st.session_state.t}  candle</h4>", unsafe_allow_html=True)
    
#     #Sample
#     sample_results = [['HINDALCO', '24', 'OCT', '750', 'CE', '04:50 PM'],
#  ['HINDALCO', '24', 'OCT', '750', 'CE', '04:50 PM'],
#  ['HINDALCO', '24', 'OCT', '750', 'CE', '04:50 PM'],
#  ['SIEMENS', '24', 'OCT', '7700', 'CE', '04:50 PM']]
#     stock_df=pd.DataFrame(sample_results,columns=['Stock','year','month','strike','type','Candle_time'])
#     stock_df=stock_df[['Stock','strike','type','Candle_time']]
#     with data_placeholder.container():
#         st.dataframe(stock_df,hide_index=True,use_container_width=True)
#     st.session_state.stock_df = stock_df
# if st.session_state.over and (not st.session_state.run):
#     st.write(st.session_state.over)
#     st.write(st.session_state.run)
#     clock_placeholder.markdown(f"<h4 style='text-align: center;'>Data is as of {st.session_state.t}  candle</h4>", unsafe_allow_html=True)
#     with data_placeholder.container():
#             st.dataframe(st.session_state.stock_df,hide_index=True,use_container_width=True)

############################### Main Code ##########################################
current_min=0
from_date, to_date = adjust_trading_dates()
print(from_date,to_date)
# user_data=pd.read_csv('user_credentials.csv')

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        user_data = pd.read_csv(uploaded_file)
        st.write("CSV file uploaded successfully!")
        users = read_user_credentials(user_data)
    elif uploaded_file.name.endswith('.xlsx'):
        user_data = pd.read_excel(uploaded_file)
        st.write("Excel file uploaded successfully!")
        users = read_user_credentials(user_data)
    else:
        st.error("Unsupported file format! Only accepted 'csv' or 'xslsx'")
else:
    st.info("Please upload a user credential file to proceed")


if 'users' in globals():    
    header = {
    "Authorization": f'enctoken {users[0]["enc"]}',  # Replace with your actual enctoken
    "Content-Type": "application/json"}

    angel_nfo_data = __import__('Angel_data_nfo')
    nfo_df=angel_nfo_data.nfo_test.reset_index(drop=True)
    nfo_df['symbolToken']=nfo_df.symbolToken.astype('int')
    print('###############################')
    print(header)
    print(nfo_df.head())
    print('###############################')
    df_proc=get_tokens(header,nfo_df)#.head(100)
results=[]


while st.session_state.run:
    if uploaded_file is None:
        st.error("Credential file not uploaded. Please upload a user credential file to proceed")
        st.session_state.run = False
        break;
    weekday,before_market,after_market,within_time=is_weekday_and_within_time()
    
    if weekday&within_time:
        st.session_state.t=get_candle_time()
        stock_df=pd.DataFrame(columns=['Stock','year','month','strike','type','Candle_time'])
        clock_placeholder.markdown(f"<h4 style='text-align: center;'>Data is as of {st.session_state.t}  candle</h4>", unsafe_allow_html=True)

        if (datetime.now().minute%5==0) & (datetime.now().minute!=current_min):
            total_success_counter = [0]        
            current_min=datetime.now().minute

            param = {
            'oi': 1,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d')}

            exch = 'NFO'
            tf = '5minute'

            # Fetch data with multiple users
            start = datetime.now()
            df_all_tokens = fetch_data_with_multiple_users(users, df_proc, exch, param, tf)
            df_all_tokens.rename(columns={'FISHERT_9_1':'FISHERT_signal','FISHERTs_9_1':'FISHERT_trigger'}, inplace=True)
            end = datetime.now()

            print(f"\nTime taken: {end - start}")


            # Call the function to filter stocks based on the conditions
            filtered_stocks,filtered_data = filter_latest_stock_data(df_all_tokens)

            if not filtered_stocks.empty:
                # Send the list to Telegram bot
                stock_lis=list(df_proc[df_proc.instrument_token.isin(filtered_stocks.token.unique())].tradingsymbol.unique())
                
                # Use regular expression to match the pattern
                pattern = r'([A-Z]+)(\d{2})([A-Z]{3})(\d+)([A-Z]+)'
                for string in lis:
                    match = re.match(pattern, string)
                    if match:
                        result = list(match.groups())
                        result.append(get_candle_time())
                        results.append(result)
                stock_df=pd.DataFrame(results,columns=['Stock','year','month','strike','type','Candle_time'])
                stock_df=stock_df[['Stock','strike','type','Candle_time']]
                
                if BOT_TOKEN & CHAT_ID:
                    send_list_to_telegram(stock_df)

        with data_placeholder.container():
            st.dataframe(stock_df,hide_index=True,use_container_width=True)
        st.session_state.stock_df = stock_df
        
        if (datetime.now().minute==59)&(datetime.now().hour<15)& (datetime.now().minute!=current_min):

            current_min=datetime.now().minute

            angel_nfo_data = __import__('Angel_data_nfo')
            nfo_df=angel_nfo_data.nfo_test.reset_index(drop=True)
            nfo_df['symbolToken']=nfo_df.symbolToken.astype('int')
            df_proc=get_tokens(header,nfo_df)
            
    
    else: #weekday,before_market,after_market,within_time
        if not weekday:
            print('Its weekend. Market is closed')
            break
        elif weekday & before_market:
            market_open_time = datetime.now().replace(hour=9, minute=19, second=0, microsecond=0)
            time_left = market_open_time - datetime.now()

            # Extract hours and minutes from the time difference
            hours, remainder = divmod(time_left.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours>0:
                print(f'Market is not open yet, or data not started Time left to open: {hours} hours and {minutes} minutes.')
            else:
                print(f'Market is not open yet, or data not started Time left to open: {minutes} minutes.')
            continue
        elif weekday & after_market:
            print('Market is closed. Its post 3:30PM')
            break
            
if st.session_state.over and (not st.session_state.run):
    clock_placeholder.markdown(f"<h4 style='text-align: center;'>Data is as of {st.session_state.t}  candle</h4>", unsafe_allow_html=True)
    with data_placeholder.container():
            st.dataframe(st.session_state.stock_df,hide_index=True,use_container_width=True)
            

