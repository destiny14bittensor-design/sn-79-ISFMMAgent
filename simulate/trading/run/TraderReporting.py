# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import json
import traceback
from arch import arch_model
from scipy.stats import levy_stable
from decimal import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import textwrap
from statsmodels.graphics.tsaplots import plot_acf

def convert_to_number(string_number):
    if string_number == '':
        return None
    try:
        #getcontext().prec = 10 
        return float(string_number)
    except ValueError:
        raise ValueError(f"Cannot convert '{string_number}' to a number")
    
def calculate_parkinson_volatility(self,data):
    """
    Calculate the Parkinson volatility estimator.

    Parameters:
    - data: DataFrame containing 'high' and 'low' prices for each time period.

    Returns:
    - parkinson_vol: A float representing the Parkinson volatility.
    """
    Low = data['Price'].min()
    High = data['Price'].max()
    # Ensure the data contains the required columns
    if pd.isna(High) or pd.isna(Low):
        raise ValueError("Data must contain 'high' and 'low' columns")
    print(High)
    print(Low)
    # Compute the log of high/low price ratios
    log_hl = np.log(High / Low)

    # Calculate the Parkinson volatility estimator
    parkinson_vol = (1 / (4 * np.log(2))) * (log_hl ** 2)
    print(parkinson_vol)
    return parkinson_vol


def calculate_vpin(self,df, num_buckets):
    """
    Calculate VPIN for trading data.

    Parameters:
    - df: pandas DataFrame with columns ['Price', 'BookId', 'Volume', 'Direction', 'Timestamp'].
    - bucket_volume: int, the volume size of each bucket.
    - trading_period_seconds: int, duration of a trading period in seconds.

    Returns:
    - pandas DataFrame with columns ['TradingPeriod', 'VPIN'].
    """
    # Ensure input DataFrame has required columns
    required_columns = {'Price', 'BookId', 'Volume', 'Direction', 'Timestamp'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")
    
    # Assign trades to trading periods
    TradingPeriod = ((self.currentTimestamp-self.trading_period_seconds) // self.trading_period_seconds)
    print(type(TradingPeriod))

    if 'Volume' not in df.columns:
        raise ValueError("Data must contain a 'Volume' column")

    total_volume = df['Volume'].sum()
    bucket_volume = total_volume/num_buckets
    
    if 'Volume' not in df.columns:
        raise ValueError("Data must contain a 'Volume' column")

    total_volume = df['Volume'].sum()
    bucket_volume = total_volume/num_buckets
        
    # Divide into volume buckets
    buy_volume, sell_volume = 0, 0
    imbalances = []
    current_bucket_volume = 0

    for _, row in df.iterrows():
        
        remaining_volume = float(row["Volume"])
        
        # Allocate volume to buckets
        while remaining_volume > 0:
            # Determine how much volume can fit in the current bucket
            volume_to_fill = min(bucket_volume - current_bucket_volume, remaining_volume)
            
            # Add to buy or sell volume based on direction
            if float(row["Direction"]) == 0:
                buy_volume += volume_to_fill
            else:
                sell_volume += volume_to_fill
            
            # Update current bucket and remaining trade volume
            current_bucket_volume += volume_to_fill
            remaining_volume -= volume_to_fill
            #print(f"{current_bucket_volume}|{buy_volume}|{sell_volume}")
            
            # If the bucket is full, calculate imbalance and reset bucket
            if current_bucket_volume == bucket_volume:
                # Calculate imbalance
                imbalance = abs(buy_volume - sell_volume) / (buy_volume + sell_volume)
                imbalances.append(imbalance)
                
                # Reset bucket
                current_bucket_volume = 0
                buy_volume, sell_volume = 0, 0
    
    # Step 3: Calculate VPIN for the trading period
    if imbalances:  # Avoid division by zero
        vpin = np.mean(imbalances)
    else:
        vpin = np.nan  # No buckets were formed in this period
        

    return vpin, TradingPeriod


    
def process_l3_file(l3_file, offset=0):
    aggressing_agent_id = []
    resting_agent_id = []
    trade_id = []
    volume = []
    price = []
    timestamp = []
    with open(l3_file, 'r', encoding='utf-8') as file:
        file.seek(offset)
        #lines = file.readlines()    
        #for i in range(len(lines)-1):
        #for line in file:
        #    _, _, json_part = lines[i].split(',',2)
        while True:
            line = file.readline()
            if not line:
                break
            offset = file.tell()  # Update offset after reading a line.        
            try:
                parts = line.split(',', 2)
                if len(parts) < 3:
                    continue
                json_part = parts[2]
                log_entry = json.loads(json_part)
                if 'trade' in log_entry:
                    # Extract the relevant fields
                    aggressing_agent_id.append(log_entry.get('logContext', {}).get('aggressingAgentId'))
                    resting_agent_id.append(log_entry.get('logContext', {}).get('restingAgentId'))
                    trade_id.append(log_entry['trade'].get('tradeId'))
                    volume.append(log_entry['trade'].get('volume')) 
                    price.append(log_entry['trade'].get('price')) 
                    timestamp.append(log_entry['trade'].get('timestamp'))
            except json.JSONDecodeError:
                #print(f"Invalid JSON: {lines[i]}")
                print(f"Invalid JSON: {line.strip()}")
    new_offset = offset
    return aggressing_agent_id, resting_agent_id, trade_id, volume, price, timestamp, new_offset


def process_l2_file(l2_file, N, offset=0):
    # Initialize lists to store processed data
    date = []
    time = []
    symbol = []
    market = []
    bidvol= []
    bidprice = []
    askvol = []
    askprice = []
    vol_imbalance = []
    bidlevels = []
    asklevels = []
    bidLiquidity = []
    askLiquidity = []
    
    # Open and read the file
    with open(l2_file, 'r') as file:
        file.seek(offset)
        #lines=file.readlines()
        #for i in range(len(lines)):        
        #for line in file:
        while True:
            line = file.readline()
            if not line:
                break
            offset = file.tell()
            # Split the line by comma
            #line_splits = lines[i].strip().split(',')
            #if len(line_splits) < 13 or i == len(lines) - 1: continue
            line_splits = line.strip().split(',')
            # Ensure we have enough fields and skip any footer or incomplete lines
            if len(line_splits) < 13:
                continue
            
            # Extract the first 11 components
            date.append(line_splits[0])
            h, m, s = line_splits[1].split(':')
            seconds = int(h) * 3600 + int(m) * 60 + float(s)
            time.append(seconds)
            symbol.append(line_splits[2])
            market.append(line_splits[3])
            bidvol.append(convert_to_number(line_splits[4]))
            bidprice.append(convert_to_number(line_splits[5]))
            askvol.append(convert_to_number(line_splits[6]))
            askprice.append(convert_to_number(line_splits[7]))


            # Extract the last component containing tuples
            BidLevels_split = line_splits[11].split()
            AskLevels_split = line_splits[12].split()
            bidlevels = []
            if len(BidLevels_split) > 0:
                for bidlevel in BidLevels_split:
                    quantity, price = bidlevel.strip('()').split('@')
                    bidlevels.append([convert_to_number(quantity),convert_to_number(price)])
            asklevels = []
            if len(AskLevels_split) > 0:
                for asklevel in AskLevels_split:
                    quantity, price = asklevel.strip('()').split('@')
                    asklevels.append([convert_to_number(quantity),convert_to_number(price)])
            
            bidlevels.sort(key=lambda x: x[1],reverse=True)
            asklevels.sort(key=lambda x: x[1])
            #print(f"{bidlevels=}|{asklevels=}")
            levels_count = min(len(bidlevels),len(asklevels),N)
            numerator = 0
            denominator = 0 
            bidVolLevels = 0
            askVolLevels = 0

            if bidprice[-1] == None or askprice[-1] == None :
                vol_imbalance.append(None) 
            else: 
                for (bvol,_),(avol,_) in zip(bidlevels[:levels_count],asklevels[:levels_count]):
                    numerator += (bvol-avol)
                    denominator += (bvol+avol)
                    bidVolLevels += bvol
                    askVolLevels += avol
                #print(f"{numerator=}|{denominator=}")
                bidLiquidity.append(bidVolLevels)
                askLiquidity.append(askVolLevels)
                vol_imbalance.append(numerator/denominator if denominator != 0 else None)
            
    # Return new data along with the new file offset.
    new_offset = offset
    return date, time, symbol, market, bidvol, bidprice, askvol, askprice, bidlevels, asklevels, vol_imbalance, bidLiquidity, askLiquidity, new_offset
    #return date, time, symbol, market, bidvol, bidprice, askvol, askprice, bidlevels, asklevels, vol_imbalance

def generate_trade_file(self, aggressing_agent_id, resting_agent_id, trade_id, volume, price, timestamp):
    trade_volume_type_feed = {"Timestamp": [], "Volume": [], "Price":[], "Type" : []}  
    trade_price_feed = {"Timestamp": [], "TradeId" : [], "Price":[], }  
    for i in range(len(trade_id)):
        trade_price_feed["Timestamp"].append(timestamp[i])
        trade_price_feed["TradeId"].append(trade_id[i])
        trade_price_feed["Price"].append(price[i])
    trade_price_feed = pd.DataFrame(trade_price_feed)

    for i in range(len(aggressing_agent_id)):
        if aggressing_agent_id[i] < 0 and aggressing_agent_id[i] >= -self.cCount and resting_agent_id[i] < 0 and resting_agent_id[i] >= -self.cCount:
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("CC")

        elif (aggressing_agent_id[i] < 0 and aggressing_agent_id[i] >= -self.cCount and resting_agent_id[i] < -self.cCount and resting_agent_id[i] >= -(self.cCount + self.fCount)) or (aggressing_agent_id[i] < -self.cCount and  aggressing_agent_id[i] >= -(self.cCount + self.fCount) and resting_agent_id[i] < 0 and resting_agent_id[i] >= -self.cCount):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("CF")

        elif (aggressing_agent_id[i] < 0 and aggressing_agent_id[i] >= -self.cCount and resting_agent_id[i] < -(self.cCount + self.fCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount)) or (aggressing_agent_id[i] < -(self.cCount + self.fCount) and aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount) and resting_agent_id[i] < 0 and resting_agent_id[i] >= -self.cCount):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("CH")

        elif (aggressing_agent_id[i] < 0 and aggressing_agent_id[i] >= -self.cCount and resting_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount)) or (aggressing_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount) and aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and resting_agent_id[i] < 0 and resting_agent_id[i] >= - self.cCount):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("CN")

        elif (aggressing_agent_id[i] < 0 and aggressing_agent_id[i] >= -self.cCount and resting_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount + self.sCount)) or (aggressing_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount + self.sCount) and resting_agent_id[i] < 0 and resting_agent_id[i] >= -self.cCount):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("CS")

        elif aggressing_agent_id[i] < -self.cCount and  aggressing_agent_id[i] >= -(self.cCount + self.fCount) and resting_agent_id[i] < -self.cCount and resting_agent_id[i] >= -(self.cCount + self.fCount):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("FF")

        elif (aggressing_agent_id[i] < -self.cCount and  aggressing_agent_id[i] >= -(self.cCount + self.fCount) and resting_agent_id[i] < -(self.cCount + self.fCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount)) or (aggressing_agent_id[i] < -(self.cCount + self.fCount) and  aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount) and resting_agent_id[i] < -self.cCount and resting_agent_id[i] >= -(self.cCount + self.fCount)):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("FH")

        elif (aggressing_agent_id[i] < -self.cCount and  aggressing_agent_id[i] >= -(self.cCount + self.fCount) and resting_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount)) or \
                (aggressing_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount) and  aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and resting_agent_id[i] < -self.cCount and resting_agent_id[i] >= -(self.cCount + self.fCount) \
            ):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("FN")
        
        elif (aggressing_agent_id[i] < -self.cCount and  aggressing_agent_id[i] >= -(self.cCount + self.fCount) and resting_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount + self.sCount)) or (aggressing_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and  aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount + self.sCount) and resting_agent_id[i] < -self.cCount and resting_agent_id[i] >= -(self.cCount + self.fCount)):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("FS")

        elif aggressing_agent_id[i] < -(self.cCount + self.fCount) and  aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount) and resting_agent_id[i] < -(self.cCount + self.fCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("HH")

        elif (aggressing_agent_id[i] < -(self.cCount + self.fCount) and  aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount) and resting_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount)) or \
                (aggressing_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount) and  aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and resting_agent_id[i] < -(self.cCount + self.fCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount) \
            ):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("HN")
        
        elif (aggressing_agent_id[i] < -(self.cCount + self.fCount) and  aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount) and resting_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount + self.sCount)) or (aggressing_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and  aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount + self.sCount) and resting_agent_id[i] < -(self.cCount + self.fCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount)):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("HS")
 
        elif aggressing_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount) and aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and resting_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("NN")

        elif (aggressing_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount ) and aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and resting_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount + self.sCount)) or (aggressing_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and  aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount + self.sCount) and resting_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount)):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("NS")

        elif aggressing_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and aggressing_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount + self.sCount) and resting_agent_id[i] < -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount) and resting_agent_id[i] >= -(self.cCount + self.fCount + self.hCount + self.iCount + self.nCount + self.sCount):
            trade_volume_type_feed["Timestamp"].append(timestamp[i])
            trade_volume_type_feed["Volume"].append(volume[i])
            trade_volume_type_feed["Price"].append(price[i])
            trade_volume_type_feed["Type"].append("SS")

    trade_volume_type_feed = pd.DataFrame(trade_volume_type_feed)
    return trade_volume_type_feed, trade_price_feed
    #for i in range(len(aggressing_agent_id)):
    #    if aggressing_agent_id[i] < -self.iCount and aggressing_agent_id[i] >= -(self.iCount + self.nCount):
    #        trade_volume_type_feed["Timestamp"].append(timestamp[i])
    #        trade_volume_type_feed["Volume"].append(volume[i])
    #        trade_volume_type_feed["Price"].append(price[i])
    #        trade_volume_type_feed["Type"].append("Noise")
    #    elif aggressing_agent_id[i] < -(self.iCount + self.nCount) and  aggressing_agent_id[i] >= -(self.iCount + self.nCount + self.fCount):
    #        trade_volume_type_feed["Timestamp"].append(timestamp[i])
    #        trade_volume_type_feed["Volume"].append(volume[i])
    #        trade_volume_type_feed["Price"].append(price[i])
    #        trade_volume_type_feed["Type"].append("Fundamentalist")
    #    elif aggressing_agent_id[i] < -(self.iCount + self.nCount + self.fCount) and aggressing_agent_id[i] >= -(self.iCount + self.nCount + self.fCount + self.cCount):
    #        trade_volume_type_feed["Timestamp"].append(timestamp[i])
    #        trade_volume_type_feed["Volume"].append(volume[i])
    #        trade_volume_type_feed["Price"].append(price[i])
    #        trade_volume_type_feed["Type"].append("Chartist")
    #    elif aggressing_agent_id[i] < -(self.iCount + self.nCount + self.fCount + self.cCount) and aggressing_agent_id[i] >= -(self.iCount + self.nCount + self.fCount + self.cCount + self.hCount):
    #        trade_volume_type_feed["Timestamp"].append(timestamp[i])
    #        trade_volume_type_feed["Volume"].append(volume[i])
    #        trade_volume_type_feed["Price"].append(price[i])
    #        trade_volume_type_feed["Type"].append("HighFrequency")
        
    #    if resting_agent_id[i] < -self.iCount and resting_agent_id[i] >= -(self.iCount + self.nCount):
    #        trade_volume_type_feed["Timestamp"].append(timestamp[i])
    #        trade_volume_type_feed["Volume"].append(volume[i])
    #        trade_volume_type_feed["Price"].append(price[i])
    #        trade_volume_type_feed["Type"].append("Noise")
    #    elif resting_agent_id[i] < -(self.iCount + self.nCount) and resting_agent_id[i] >= -(self.iCount + self.nCount + self.fCount):
    #        trade_volume_type_feed["Timestamp"].append(timestamp[i])
    #        trade_volume_type_feed["Volume"].append(volume[i])
    #        trade_volume_type_feed["Price"].append(price[i])
    #        trade_volume_type_feed["Type"].append("Fundamentalist")
    #    elif resting_agent_id[i] < -(self.iCount + self.nCount + self.fCount) and resting_agent_id[i] >= -(self.iCount + self.nCount + self.fCount + self.cCount):
    #        trade_volume_type_feed["Timestamp"].append(timestamp[i])
    #        trade_volume_type_feed["Volume"].append(volume[i])
    #        trade_volume_type_feed["Price"].append(price[i])
    #        trade_volume_type_feed["Type"].append("Chartist")
    #    elif resting_agent_id[i] < -(self.iCount + self.nCount + self.fCount + self.cCount) and resting_agent_id[i] >= -(self.iCount + self.nCount + self.fCount + self.cCount + self.hCount):
    #        trade_volume_type_feed["Timestamp"].append(timestamp[i])
    #        trade_volume_type_feed["Volume"].append(volume[i])
    #        trade_volume_type_feed["Price"].append(price[i])
    #        trade_volume_type_feed["Type"].append("HighFrequency")
    #trade_volume_type_feed = pd.DataFrame(trade_volume_type_feed)
    #return trade_volume_type_feed, trade_price_feed

def generate_plots(self, out_dir, l2_file, time, timestamp_trade,bestbidvol,bestbidprice,bestaskvol,bestaskprice,bidlevels,asklevels,bidliquidity,askliquidity,vol_imbalance, trade_volume_type_feed, trade_price_feed, bookId, messagetype):
    # Calculate and plot the relative spread, midquote, and volume imbalance
    _, relative_spread, midquote, weighted_midquote, _ = lob_stats(bestaskprice,bestbidprice,bestaskvol,bestbidvol)

    agent_type = 'CH+HFT'
    try:
        self.fCount
    except NameError:
        self.fCount = 0    
    try:
        self.cCount
    except NameError:
        self.cCount =0
    try:
        self.nCount
    except NameError:
        self.nCount =0
    try:
        self.sCount
    except NameError:
        self.sCount =0
    try:
        self.hCount
    except NameError:
        self.hCount =0
        
    total_count = self.fCount + self.cCount + self.nCount + self.sCount + self.hCount
    type_proportion = {}
    type_proportion['Fundamentalist'] = self.fCount / total_count
    type_proportion['Chartist'] = self.cCount /total_count
    type_proportion['Noise'] = self.nCount /total_count
    type_proportion['Stylized'] = self.sCount /total_count
    type_proportion['HighFrequency'] = self.hCount /total_count
    #print(f"{type_proportion['Fundamentalist']=}|{type_proportion['Chartist']=}|{type_proportion['Noise']=}|{type_proportion['Stylized']=}|{type_proportion['HighFrequency']=}")
    # print(f"{len(relative_spread)=}")


    #Generate a plot showing inventory evolution throughout simulation time

    df_aggregateInventory = pd.read_csv(os.path.join(self.logDir,'output_inventory.csv'))

    df_filtered = df_aggregateInventory[df_aggregateInventory['BookId'] == int(bookId)]
    #print(f"{type(df_aggregateInventory['BookId'][0])=}") 
    #print(f"{type(df_aggregateInventory['BookId'][1])=}") 
    #print(f'{df_filtered=}')
    #print(f'{df_filtered=}')
    #print(f'{df_filtered=}')
    #print(f'{df_filtered=}')
    #print(f'{df_filtered=}')
    #print(f"{type(df_filtered['Timestamp'])=}")
    #print(f"{type(df_filtered['Inventory'])=}")
    #print(f"{type(df_filtered['Timestamp'])=}")
    #print(f"{type(df_filtered['Inventory'])=}")
    plt.figure(figsize=(12,8))
    plt.plot(df_filtered['Timestamp']*self.timeScale, df_filtered['Inventory'])
    plt.xlabel('Time (sec)')
    plt.ylabel('HFT Aggregate Inventory')
    plt.savefig(os.path.join(out_dir,'aggregateInventory.png'))
    plt.clf()
    plt.close()

    # Generate a plot with three panels corresponding to relative spread, midquote, and volume imbalance
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    trim_len = 100
    axs[0].plot(time[trim_len:], relative_spread[trim_len:])
    axs[0].set_ylabel('relative spread')
    if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
        axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    else:
        axs[0].set_title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))

    axs[1].plot(time[trim_len:], midquote[trim_len:])
    axs[1].set_ylabel('mid-quote')
    axs[2].plot(time[trim_len:], vol_imbalance[trim_len:])
    axs[2].set_xlabel('Time (sec)')
    axs[2].set_ylabel('volume imbalance')
    plt.savefig(os.path.join(out_dir,'smv.png'))
    plt.clf()
    plt.close()


    plt.figure(figsize=(12,8))
    plt.plot(time[trim_len:], bidliquidity[trim_len:])
    plt.plot(time[trim_len:], askliquidity[trim_len:])
    plt.xlabel('Time (sec)')
    plt.ylabel('Bid-Ask Liquidity')
    if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    else:
        plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))
    plt.legend(labels=['Bid', 'Ask'])
    plt.savefig(os.path.join(out_dir,'bidask_liquidity.png'))
    plt.clf()
    plt.close('all')

    #print('Hello smv')
    # Generate the relative spread plot
    plt.figure(figsize=(12,8))
    plt.plot(time[trim_len:], relative_spread[trim_len:])
    plt.xlabel('Time (sec)')
    plt.ylabel('relative spread')
    if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    else:
        plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))
    plt.savefig(os.path.join(out_dir,'relative_spread.png'))
    plt.clf()
    plt.close('all')
    del relative_spread
    #print('Hello relative_spread')

    # Generate the midquote plot
    plt.figure(figsize=(12,8))
    plt.plot(time[trim_len:], midquote[trim_len:])
    plt.xlabel('Time (sec)')
    plt.ylabel('mid-quote')
    if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    else:
        plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count)) 
    plt.savefig(os.path.join(out_dir,'midquote.png'))
    plt.clf()
    plt.close('all')
    #print('Hello midquote')
    
    # Generate the weighted midquote plot
    plt.figure(figsize=(12,8))
    plt.plot(time[trim_len:], weighted_midquote[trim_len:])
    plt.xlabel('Time (sec)')
    plt.ylabel('weighted-mid-quote')
    if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    else:
        plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count)) 
    plt.savefig(os.path.join(out_dir,'weighted_midquote.png'))
    plt.clf()
    plt.close('all')
    #print('Hello weighted_midquote')

    # Load trade timeseries
    df_return = trade_price_feed.copy()
    df_return['Return'] = (np.log(trade_price_feed.Price) - np.log(trade_price_feed.Price.shift(1))).fillna(0)

    # Generate the trade plot
    if not df_return['Return'].empty:
        plt.figure(figsize=(12,8))
        plt.plot(timestamp_trade * self.timeScale, trade_price_feed['Price'])
        plt.xlabel('Time (sec)')
        plt.ylabel('trade price')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count)) 
        plt.savefig(os.path.join(out_dir,'trade_price.png'))
        plt.clf()
        plt.close('all')
    else:
        print('No trade has yet occurred')
    #print('Hello return')
    # Generate the volume imbalance plot
    plt.figure(figsize=(12,8))
    plt.plot(time[trim_len:], vol_imbalance[trim_len:])
    plt.xlabel('Time (sec)')
    plt.ylabel('volume imbalance')
    if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    else:
        plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count)) 
    plt.savefig(os.path.join(out_dir,'volume_imbalance.png'))
    plt.clf()
    plt.close('all')
    del vol_imbalance
    #print('Hello volume imbalance')

    # Generate the midquote and weighted midquote timeseries where the average of prices at each timestamp defines the price.
    df_midquote = pd.DataFrame({'MidQuote': midquote, 'Timestamp':time})
    df_weighted_midquote = pd.DataFrame({'Weighted_MidQuote': weighted_midquote, 'Timestamp':time})
    midquote_time_series=df_midquote.groupby('Timestamp')['MidQuote'].last().reset_index()
    weighted_midquote_time_series=df_weighted_midquote.groupby('Timestamp')['Weighted_MidQuote'].last().reset_index()
    #print('computing weighted midquote')
    # Load sampled fundamental price
    #if os.path.exists(os.path.join(self.logDir,'output_priceFP.csv')):
    #    df_Pf = pd.read_csv(os.path.join(self.logDir,'output_priceFP.csv'), header=0)
    #    df_Pf.set_index('Timestamp')
    #else:
    df_Pf = pd.read_csv(os.path.join(self.logDir,'fundamental.csv'), header=0)
    time_fundamental = df_Pf['Timestamp'][:int(midquote_time_series['Timestamp'].iloc[-1])] * self.timeScale
    df_Pf.set_index('Timestamp')

    # Generate midquote plot aligned with fundamental price
    plt.figure(figsize=(12,8))
    plt.plot(midquote_time_series['Timestamp'] * self.timeScale, midquote_time_series['MidQuote'])       
    plt.plot(time_fundamental, df_Pf[str(bookId)][:int(midquote_time_series['Timestamp'].iloc[-1])])
    if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    else:
        plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count)) 
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(labels=['Midquote', 'Fundamental'])
    plt.savefig(os.path.join(out_dir,'midquote_averaged.png'))
    plt.clf()
    plt.close('all')
    #print('Hello midquote_pf')

# Generate fundamental return plot 
    fundamental_return = np.log(df_Pf[str(bookId)][1:].reset_index(drop=True)/df_Pf[str(bookId)][:-1].reset_index(drop=True))
    fundamental_lag1 = fundamental_return.autocorr(lag=1)
    plt.figure(figsize=(12,8))
    plt.plot(df_Pf['Timestamp'][1:]*self.timeScale, fundamental_return)
    plt.ylabel('fundamental price return')
    #axs[0].set_title('T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, S = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],type_proportion['Noise'], type_proportion['Stylized'], total_count))
    if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    else:
        plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))     
    plt.xlabel('Time (sec)')
    plt.ylabel('Fundamental price return')
    plt.savefig(os.path.join(out_dir,'fundamental_return.png'))
    plt.clf()
    plt.close('all')

    # Generate fundamental reutrn autocorrelation plot
    plt.figure(figsize=(20,14))
    plot_acf(fundamental_return, lags =min(len(fundamental_return),20))
    plt.xlabel('lags')
    plt.ylabel('fundamental price return SACF')
    #plt.title('T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, S = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],type_proportion['Noise'], type_proportion['Stylized'], total_count))
    if self.fCount == 0 and self.cCount == 0 and self.nCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, fundamental_lag1)
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        itle = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, fundamental_lag1)
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, fundamental_lag1)
    else:
        title = 'T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count, fundamental_lag1)
    wrapped_title = "\n".join(textwrap.wrap(title, width=70))  # Adjust width as needed
    plt.title(wrapped_title, fontsize=10, pad=0)
    plt.ylim((-2, 2))
    plt.savefig(os.path.join(out_dir,'fundamental_return_autocorrelation.png'))
    plt.clf()
    plt.close('all')

    if messagetype == "EVENT_SIMULATION_STOP":
        data_scaled = (fundamental_return - fundamental_return.mean()) / fundamental_return.std()
        model_fundamental = arch_model(data_scaled, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
        garch_fit_fundamental = model_fundamental.fit(disp='off')
        beta_param = garch_fit_fundamental.params['beta[1]']
        alpha_param = 0
        #params = levy_stable.fit(fundamental_return)
        #alpha_param = params[0]
    else:
        beta_param = 0
        alpha_param = 0

    # Generate fundamental return histogram
    plt.figure(figsize=(12,8))
    plt.hist( fundamental_return, bins=10, color='blue', alpha=0.7)
    plt.xlabel('fundamental return')
    plt.ylabel('Frequency')
    if self.fCount == 0 and self.cCount == 0 and self.nCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        itle = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    else:
        title = 'T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    wrapped_title = "\n".join(textwrap.wrap(title, width=70))  # Adjust width as needed
    plt.title(wrapped_title, fontsize=10, pad=0)
    plt.savefig(os.path.join(out_dir,'fundamental_return_histogram.png'))
    plt.clf()
    plt.close('all')
    del fundamental_return

    try:
        plt.figure(figsize=(12,8))
        plt.plot(
            midquote_time_series[(midquote_time_series['Timestamp'] >= (self.currentTimestamp-self.reportInterval / self.timeScale)) & (midquote_time_series['Timestamp'] <= self.currentTimestamp)]['Timestamp'] * self.timeScale,\
            midquote_time_series[(midquote_time_series['Timestamp'] >= (self.currentTimestamp-self.reportInterval / self.timeScale)) & (midquote_time_series['Timestamp'] <= self.currentTimestamp)]['MidQuote']
        )       
        plt.plot(
            df_Pf['Timestamp'][(df_Pf['Timestamp'] >= (self.currentTimestamp-self.reportInterval / self.timeScale)) & (df_Pf['Timestamp'] <= self.currentTimestamp)] * self.timeScale,
            df_Pf[str(bookId)][(df_Pf['Timestamp'] >= (self.currentTimestamp-self.reportInterval / self.timeScale)) & (df_Pf['Timestamp'] <= self.currentTimestamp)]
        )
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))        
        plt.xlabel('Time (sec)')
        plt.ylabel('Price')
        plt.legend(labels=['Midquote', 'Fundamental'])
        plt.savefig(os.path.join(out_dir,f'midquote_averaged_{self.checkpoints[0] if self.checkpoints != [] else self.duration}.png'))
        plt.clf()
        plt.close('all')
    except Exception as e:
        # Print an error message if plotting fails
        print("An error occurred while trying to plot:")
        traceback.print_exc()




    # Generate weighted midquote plot aligned with fundamental price
    plt.figure(figsize=(12,8))
    plt.plot(weighted_midquote_time_series['Timestamp'], weighted_midquote_time_series['Weighted_MidQuote'])
    plt.plot(time_fundamental, df_Pf[str(bookId)][:int(midquote_time_series['Timestamp'].iloc[-1])])
    if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
    else:
        plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count)) 
    plt.xlabel('Time (sec)')
    plt.ylabel('Price')
    plt.legend(labels=['Weighted Midquote', 'Fundamental'])
    plt.savefig(os.path.join(out_dir,'weighted_midquote_averaged.png'))
    plt.clf()
    plt.close('all')

    #print('Hello weighted_midquote_pf')
    # Generate the trade plot aligned with fundamental price where the average of all prices at each timestamp determines the price.
    trade_time_series=trade_price_feed.groupby('Timestamp')['Price'].last().reset_index()
    if not trade_time_series.empty:
        plt.figure(figsize=(12,8))
        plt.plot(trade_time_series['Timestamp'] * self.timeScale, trade_time_series['Price'])
        plt.plot(time_fundamental, df_Pf[str(bookId)][:int(midquote_time_series['Timestamp'].iloc[-1])])
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))
        plt.xlabel('Time (sec)')
        plt.ylabel('Price')
        plt.legend(labels=['Trade', 'Fundamental'])
        plt.savefig(os.path.join(out_dir,'trade_price_averaged.png'))
        plt.clf()
        plt.close('all')
    else:
        print('No trade has yet occurred')
    #print('Hello trade_average')

    try:
        plt.figure(figsize=(12,8))
        plt.plot(
            trade_time_series[(trade_time_series['Timestamp'] >= (self.currentTimestamp-self.reportInterval / self.timeScale)) & (trade_time_series['Timestamp'] <= self.currentTimestamp)]['Timestamp'] * self.timeScale, \
            trade_time_series[(trade_time_series['Timestamp'] >= (self.currentTimestamp-self.reportInterval / self.timeScale)) & (trade_time_series['Timestamp'] <= self.currentTimestamp)]['Price']
        )       
        plt.plot(
            df_Pf['Timestamp'][(df_Pf['Timestamp'] >= (self.currentTimestamp-self.reportInterval / self.timeScale)) & (df_Pf['Timestamp'] <= self.currentTimestamp)] * self.timeScale,\
            df_Pf[str(bookId)][(df_Pf['Timestamp'] >= (self.currentTimestamp-self.reportInterval / self.timeScale)) & (df_Pf['Timestamp'] <= self.currentTimestamp)]
        )
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))        
        plt.xlabel('Time (sec)')
        plt.ylabel('Price')
        plt.legend(labels=['Trade', 'Fundamental'])
        plt.savefig(os.path.join(out_dir,f'trade_price_averaged_{self.checkpoints[0] if self.checkpoints != [] else self.duration}.png'))
        plt.clf()
        plt.close('all')

    except Exception as e:
        # Print an error message if plotting fails
        print("An error occurred while trying to plot:")
        traceback.print_exc()

    del df_midquote
    del df_weighted_midquote
    del weighted_midquote
    del midquote

    # Generate the return plot aligned with absolute return
    if not df_return['Return'].empty:
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))
        axs[0].plot(df_return['Timestamp'] * self.timeScale, df_return['Return'])
        axs[0].set_ylabel('return')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            axs[0].set_title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))

        axs[1].plot(df_return['Timestamp'] * self.timeScale, abs(df_return['Return']))
        axs[1].set_xlabel('Time (sec)')
        axs[1].set_ylabel('Asolute return (trade)')
        axs[1].set_title('Absolute Return')
        plt.savefig(os.path.join(out_dir,'return.png'))
        plt.clf()
        plt.close('all')
    else:
        print('No trade has yet occurred')
    #print('Hello return_absolute_return')
    #print(f"{len(df_return)=}")
    # Generate the return SACF (which is the sampled version of midquote by agents)
    if not df_return['Return'].empty:
        plt.figure(figsize=(12,8))
        plot_acf(df_return['Return'], lags =min(len(df_return),20))
        plt.xlabel('lags')
        plt.ylabel('return SACF')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))
        plt.ylim((-2, 2))
        plt.savefig(os.path.join(out_dir,'return_autocorrelation.png'))
        plt.clf()
        plt.close('all')
    else:
        print('No trade has yet occurred')
    del df_return

    #print('Hello return_autocorrelation')
    # Generate the midquote return SACF (averaging prices at each timestamp)
    midquote_return = np.log(midquote_time_series['MidQuote'][1:].reset_index(drop=True)/midquote_time_series['MidQuote'][:-1].reset_index(drop=True))
    midquote_lag1 = midquote_return.autocorr(lag=1)
    if messagetype == "EVENT_SIMULATION_STOP":
        data_scaled = (midquote_return - midquote_return.mean()) / midquote_return.std()
        print(f'number of NaN in midquote: {data_scaled.isna().sum()}')  # Check NaN
        print(f'number of Inf in midquote: {np.isinf(data_scaled).sum()}') 
        data_scaled = data_scaled.dropna()
        data_scaled = data_scaled[~np.isinf(data_scaled)]
        assert data_scaled.isna().sum() == 0
        assert np.isinf(data_scaled).sum() == 0
        model_midquote = arch_model(data_scaled, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
        garch_fit_midquote = model_midquote.fit(disp='off')
        beta_param = garch_fit_midquote.params['beta[1]']
        alpha_param = 0
        #params = levy_stable.fit(midquote_return)
        #alpha_param = params[0]
    else:
        beta_param = 0
        alpha_param = 0

    plt.figure(figsize=(12,8))
    plot_acf(midquote_return, lags =min(len(midquote_return)-1,20))
    plt.xlabel('lags')
    plt.ylabel('midquote return SACF')
    if self.fCount == 0 and self.cCount == 0 and self.nCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, midquote_lag1)
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        itle = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, midquote_lag1)
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, midquote_lag1)
    else:
        title = 'T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count, midquote_lag1)
    wrapped_title = "\n".join(textwrap.wrap(title, width=70))  # Adjust width as needed
    plt.title(wrapped_title, fontsize=10, pad=0)
    plt.ylim((-2, 2))
    plt.savefig(os.path.join(out_dir,'midquote_return_autocorrelation.png'))
    plt.clf()
    plt.close('all')
    #print('Hello midquote_return_autocorrelation')

    plt.figure(figsize=(12,8))
    plt.hist( midquote_return, bins=10, color='blue', alpha=0.7)
    plt.xlabel('midquote return')
    plt.ylabel('Frequency')
    if self.fCount == 0 and self.cCount == 0 and self.nCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        itle = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    else:
        title = 'T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    wrapped_title = "\n".join(textwrap.wrap(title, width=70))  # Adjust width as needed
    plt.title(wrapped_title, fontsize=10, pad=0)  
    plt.savefig(os.path.join(out_dir,'midquote_return_histogram.png'))
    plt.clf()
    plt.close('all')
    del midquote_return, midquote_time_series



    # Generate the weighted midquote return SACF
    weighted_midquote_return = np.log(weighted_midquote_time_series['Weighted_MidQuote'][1:].reset_index(drop=True)/weighted_midquote_time_series['Weighted_MidQuote'][:-1].reset_index(drop=True))
    weighted_midquote_lag1 = weighted_midquote_return.autocorr(lag=1)
    if messagetype == "EVENT_SIMULATION_STOP":
        data_scaled = (weighted_midquote_return - weighted_midquote_return.mean()) / weighted_midquote_return.std()
        print(f'number of NaN in weighted midquote: {data_scaled.isna().sum()}')  # Check NaN
        print(f'number of Inf in weighted midquote: {np.isinf(data_scaled).sum()}') 
        data_scaled = data_scaled.dropna()
        data_scaled = data_scaled[~np.isinf(data_scaled)]
        assert data_scaled.isna().sum() == 0
        assert np.isinf(data_scaled).sum() == 0
        model_weighted_midquote = arch_model(data_scaled, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
        garch_fit_weighted_midquote = model_weighted_midquote.fit(disp='off')
        beta_param = garch_fit_weighted_midquote.params['beta[1]']
        alpha_param = 0
        #params = levy_stable.fit(weighted_midquote_return)
        #alpha_param = params[0]
    else:
        beta_param = 0
        alpha_param = 0
    plt.figure(figsize=(18,10))
    plot_acf(weighted_midquote_return, lags =min(len(weighted_midquote_return)-1,20))
    plt.xlabel('lags')
    if self.fCount == 0 and self.cCount == 0 and self.nCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, weighted_midquote_lag1)
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        itle = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, weighted_midquote_lag1)
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, weighted_midquote_lag1)
    else:
        title = 'T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
        agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count, weighted_midquote_lag1)
    wrapped_title = "\n".join(textwrap.wrap(title, width=70))  # Adjust width as needed
    plt.title(wrapped_title, fontsize=10, pad=0)
    plt.ylim((-2, 2))
    plt.savefig(os.path.join(out_dir,'weighted_midquote_return_autocorrelation.png'))
    plt.close('all')

    plt.figure(figsize=(12,8))
    plt.hist( weighted_midquote_return, bins=10, color='blue', alpha=0.7)
    plt.xlabel('weight midquote return')
    plt.ylabel('Frequency')
    if self.fCount == 0 and self.cCount == 0 and self.nCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
        itle = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
        title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
        type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    else:
        title = 'T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
        agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
    wrapped_title = "\n".join(textwrap.wrap(title, width=70))  # Adjust width as needed
    plt.title(wrapped_title, fontsize=10, pad=0)  
    plt.savefig(os.path.join(out_dir,'weighted_midquote_return_histogram.png'))
    plt.clf()
    plt.close('all')
    del weighted_midquote_time_series, weighted_midquote_return

    # Generate the trade SACF where the average of all prices at each timestamp determines the price.
    trade_return = np.log(trade_time_series['Price'][1:].reset_index(drop=True)/trade_time_series['Price'][:-1].reset_index(drop=True))
    trade_lag1 = trade_return.autocorr(lag=1)
    if not trade_return.empty:
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))
        axs[0].plot(trade_time_series['Timestamp'][1:] * self.timeScale, trade_return)
        axs[0].set_ylabel('trade return')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            axs[0].set_title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))
        axs[1].plot(trade_time_series['Timestamp'][1:] * self.timeScale, abs(trade_return))
        axs[1].set_xlabel('Time (sec)')
        axs[1].set_ylabel('Asolute return (trade)')
        axs[1].set_title('Absolute Return')
        plt.savefig(os.path.join(out_dir,'return_averaged.png'))
        plt.clf()
        plt.close('all')  
    else:
        print('No trade has yet occurred') 

    if messagetype == "EVENT_SIMULATION_STOP":
        data_scaled = (trade_return - trade_return.mean()) / trade_return.std()
        print(f'number of NaN in trade: {data_scaled.isna().sum()}')  # Check NaN
        print(f'number of Inf in trade: {np.isinf(data_scaled).sum()}') 
        data_scaled = data_scaled.dropna()
        data_scaled = data_scaled[~np.isinf(data_scaled)]
        assert data_scaled.isna().sum() == 0
        assert np.isinf(data_scaled).sum() == 0
        model_trade = arch_model(data_scaled, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
        garch_fit_trade = model_trade.fit(disp='off')
        beta_param = garch_fit_trade.params['beta[1]']
        #params = levy_stable.fit(trade_return)
        #alpha_param = params[0]
        alpha_param = 0
        beta_alpha = pd.DataFrame({'Beta': [beta_param],'Alpha': [alpha_param]})
        lag1_SACF = pd.DataFrame({'Lag1': [trade_lag1]})
    else:
        beta_param = 0
        alpha_param = 0
        beta_alpha = pd.DataFrame({'Beta': [0],'Alpha': [0]})
        lag1_SACF = pd.DataFrame({'Lag1': [0]})

    if not trade_return.empty:
        plt.figure(figsize=(12,8))
        plot_acf(trade_return, lags=min(len(trade_return)-1, 20))
        plt.xlabel('lags')
        plt.ylabel('trade return SACF')
        if self.fCount == 0 and self.cCount == 0 and self.nCount != 0:
            title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
            agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
            type_proportion['Noise'], type_proportion['HighFrequency'], total_count, trade_lag1)
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
            agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
            type_proportion['Noise'], type_proportion['HighFrequency'], total_count, trade_lag1)
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
            agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
            type_proportion['Noise'], type_proportion['HighFrequency'], total_count, trade_lag1)
        else:
            title = 'T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}, Lag1 = {:.2f}'.format(
            agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count, trade_lag1)
        wrapped_title = "\n".join(textwrap.wrap(title, width=70))  # Adjust width as needed
        plt.title(wrapped_title, fontsize=10, pad=0)  
        plt.ylim((-2, 2))
        plt.savefig(os.path.join(out_dir,'trade_return_autocorrelation.png'))
        plt.clf()
        plt.close('all')
    else:
        print('No trade has yet occurred')

    if not trade_return.empty:
        plt.figure(figsize=(14,8))
        plt.hist(trade_return, bins=10, color='blue', alpha=0.7)
        plt.xlabel('trade return')
        plt.ylabel('Frequency')
        if self.fCount == 0 and self.cCount == 0 and self.nCount != 0:
            title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
            agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
            type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
            agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
            type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            title = 'T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
            agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],
            type_proportion['Noise'], type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
        else:
            title = 'T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}, B={:.2f}, Alpha={:.2f}'.format(
            agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count, beta_param, alpha_param)
        wrapped_title = "\n".join(textwrap.wrap(title, width=70))  # Adjust width as needed
        plt.title(wrapped_title, fontsize=10, pad=0) 
        plt.savefig(os.path.join(out_dir,'trade_return_histogram.png'))
        plt.clf()
        plt.close('all')
    else:
        print('No trade has yet occurred')
    del trade_time_series, trade_price_feed, trade_return
    
    #plt.figure()
    #plt.hist( midquote_return, bins=10, color='blue', alpha=0.7)
    #plt.title('T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, S = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],type_proportion['Noise'], type_proportion['Stylized'], type_proportion['HighFrequency'], total_count))
    #plt.xlabel('midquote return')
    #plt.ylabel('histogram of midquote return')
    #plt.savefig(os.path.join(out_dir,'hist_midquote_return.png'))
    #plt.close('all')

    #plt.figure()
    #plt.hist( weighted_midquote_return, bins=10, color='blue', alpha=0.7)
    #plt.title('T = {:s}, F = {:.2f}, C = {:.2f}, N = {:.2f}, S = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],type_proportion['Noise'], type_proportion['Stylized'], type_proportion['HighFrequency'], total_count))
    #plt.xlabel('weighted midquote return')
    #plt.ylabel('histogram of weighted midquote return')
    #plt.savefig(os.path.join(out_dir,'hist_weighted_midquote_return.png'))
    #plt.close('all')


    volume_time_series_sum = trade_volume_type_feed.groupby('Timestamp')['Volume'].sum().reset_index()
    if not volume_time_series_sum.empty:
        plt.figure(figsize=(12,8))
        plt.plot(volume_time_series_sum['Timestamp'] * self.timeScale, volume_time_series_sum['Volume'])
        plt.xlabel('Time (sec)')
        plt.ylabel('Volume')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))    
        plt.savefig(os.path.join(out_dir,f'volume_time_series.png'))
        plt.clf()
        plt.close('all')
    else:
        print('No trade has yet occurred')

    try:
        plt.figure(figsize=(12,8))
        plt.plot(
            volume_time_series_sum[(volume_time_series_sum['Timestamp'] >= (self.currentTimestamp-self.reportInterval / self.timeScale)) & (volume_time_series_sum['Timestamp'] <= self.currentTimestamp)]['Timestamp'] * self.timeScale, 
            volume_time_series_sum[(volume_time_series_sum['Timestamp'] >= (self.currentTimestamp-self.reportInterval / self.timeScale)) & (volume_time_series_sum['Timestamp'] <= self.currentTimestamp)]['Volume']
        )        
        plt.xlabel('Time (sec)')
        plt.ylabel('Volume')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],type_proportion['Noise'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],type_proportion['Noise'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = 0.00, N = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],type_proportion['Noise'], total_count))
        else:
            plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, total_count))    
        plt.savefig(os.path.join(out_dir,f'volume_time_series_{self.checkpoints[0] if self.checkpoints != [] else self.duration}.png'))
        plt.clf()
        plt.close('all')
    except Exception as e:
        # Print an error message if plotting fails
        print("An error occurred while trying to plot:")
        traceback.print_exc()
    del volume_time_series_sum

    target_agents = ['HH', 'HN', 'HC', 'HF', 'HS']
    volume_type_time_series=trade_volume_type_feed.groupby(['Timestamp', 'Type'])['Volume'].sum().reset_index()
    df_volume_pivot = volume_type_time_series.pivot(index='Timestamp', columns='Type', values='Volume').fillna(0)
    df_volume_percentage = df_volume_pivot.div(df_volume_pivot.sum(axis=1), axis=0)
    print('Tik')
    #df_volume_percentageH = df_volume_percentage[target_agents]
    df_volume_percentageH = df_volume_percentage.reindex(columns=target_agents, fill_value=0)
    print('Tok')
    if not df_volume_percentageH.empty:
        plt.figure(figsize=(12,8))
        plt.stackplot(df_volume_percentageH.index * self.timeScale, df_volume_percentageH.T, labels=df_volume_percentageH.columns, alpha=0.8)
        #plt.plot(volume_type_time_series[volume_type_time_series['Type']=="Noise"]['Timestamp'],volume_type_time_series[volume_type_time_series['Type']=="Noise"]['Volume'])
        #plt.plot(volume_type_time_series[volume_type_time_series['Type']=="Chartist"]['Timestamp'],volume_type_time_series[volume_type_time_series['Type']=="Chartist"]['Volume'])
        #plt.plot(volume_type_time_series[volume_type_time_series['Type']=="Fundamentalist"]['Timestamp'],volume_type_time_series[volume_type_time_series['Type']=="Fundamentalist"]['Volume'])
        #plt.plot(volume_type_time_series[volume_type_time_series['Type']=="HighFrequency"]['Timestamp'],volume_type_time_series[volume_type_time_series['Type']=="HighFrequency"]['Volume'])
        plt.xlabel('Time (sec)')
        plt.ylabel('HFT Percentage of Total Trade Volume')
        plt.legend(loc="upper right")
        #plt.legend(["N","C","F","H"], loc="lower right")
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count)) 
        plt.savefig(os.path.join(out_dir,'trade_volume_plot_hft.png'))
        plt.clf()
        plt.close('all')
    else: 
        print('No trade has yet occurred')

    df_volume_percentageO = df_volume_percentage.drop(columns=target_agents, errors='ignore')
    if not df_volume_percentageO.empty:
        plt.figure(figsize=(12,8))
        plt.stackplot(df_volume_percentageO.index * self.timeScale, df_volume_percentageO.T, labels=df_volume_percentageO.columns, alpha=0.8)
        #plt.plot(volume_type_time_series[volume_type_time_series['Type']=="Noise"]['Timestamp'],volume_type_time_series[volume_type_time_series['Type']=="Noise"]['Volume'])
        #plt.plot(volume_type_time_series[volume_type_time_series['Type']=="Chartist"]['Timestamp'],volume_type_time_series[volume_type_time_series['Type']=="Chartist"]['Volume'])
        #plt.plot(volume_type_time_series[volume_type_time_series['Type']=="Fundamentalist"]['Timestamp'],volume_type_time_series[volume_type_time_series['Type']=="Fundamentalist"]['Volume'])
        #plt.plot(volume_type_time_series[volume_type_time_series['Type']=="HighFrequency"]['Timestamp'],volume_type_time_series[volume_type_time_series['Type']=="HighFrequency"]['Volume'])
        plt.xlabel('Time (sec)')
        plt.ylabel('NON-HFT Percentage of Total Trade Volume')
        plt.legend(loc="upper right")
        #plt.legend(["N","C","F","H"], loc="lower right")
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            plt.title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            plt.title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count)) 
        plt.savefig(os.path.join(out_dir,'trade_volume_plot_nhft.png'))
        plt.clf()
        plt.close('all')
    else:
        print('No trade has yet occurred')
    del volume_type_time_series


    #print("before_pricevol_Hello")

    # NEw
    df_results = lsSeries(l2_file, self.currentTimestamp * self.timeScale, self.reportInterval)
    if not df_results.empty:
        plt.figure(figsize=(12,8))
        plt.plot(df_results['Timestamp'], df_results['m_bid'], label='Bid Slope', color='blue', marker='o', markersize=4, linestyle='-')
        plt.plot(df_results['Timestamp'], df_results['m_ask'], label='Ask Slope', color='orange', marker='o', markersize=4, linestyle='-')
        plt.title('Bid and Ask Slope over Time')
        plt.xlabel('Time (sec)')
        plt.ylabel('Slope')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,f'slope_plot_{self.currentTimestamp}.png'))
        plt.clf()
        plt.close('all')
    else:
        print('LOB is empty')

    # Generate Liquidity-Price plot for the top 21 levels aligned with Liquidity-Price plot for the orders falling within 10 percent of best bid/ask
    bidlevelsprice_count, bidlevelsvol_count, asklevelsprice_count, asklevelsvol_count = pricevol_count(l2_file, 21)
    #print("mid_pricevol_Hello")
    _, bidlevelsprice_percent, bidlevelsvol_percent, _, asklevelsprice_percent, asklevelsvol_percent = pricevol(l2_file, 0.1)
    #print("pricevol_Hello")
    fig, axs = plt.subplots(2, 1, figsize=(8, 12))
    if not asklevelsprice_count:
        m_bid_count, c_bid_count = np.linalg.lstsq(np.vstack([bidlevelsprice_count, np.ones(len(bidlevelsprice_count))]).T, bidlevelsvol_count, rcond=None)[0]
        axs[0].scatter(bidlevelsprice_count, bidlevelsvol_count)
        axs[0].plot(bidlevelsprice_count, m_bid_count*np.array(bidlevelsprice_count) + c_bid_count, linestyle='--', label='Fitted Line - Bid')
        axs[0].set_ylabel('Liquidity (for the top 21 levels)')
        axs[0].annotate(f"BidS = {m_bid_count:.2f}", xy=(bidlevelsprice_count[-1], m_bid_count*np.array(bidlevelsprice_count[-1]) + c_bid_count), xytext=(bidlevelsprice_count[-1] + 0.2, m_bid_count*np.array(bidlevelsprice_count[-1]) + c_bid_count + 0.1),
            arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
        axs[0].annotate(f"AskS = {m_ask_count:.2f}", xy=(asklevelsprice_count[-1], m_ask_count*np.array(asklevelsprice_count[-1]) + c_ask_count), xytext=(asklevelsprice_count[-1] - 0.7, m_ask_count*np.array(asklevelsprice_count[-1]) + c_ask_count + 0.1),
            arrowprops=dict(arrowstyle='<-', color='blue'), color='blue')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            axs[0].set_title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count)) 
    
    elif not bidlevelsprice_count:
        m_ask_count, c_ask_count = np.linalg.lstsq(np.vstack([asklevelsprice_count, np.ones(len(asklevelsprice_count))]).T, asklevelsvol_count, rcond=None)[0]
        axs[0].scatter(asklevelsprice_count, asklevelsvol_count)
        axs[0].plot(asklevelsprice_count, m_ask_count*np.array(asklevelsprice_count) + c_ask_count, linestyle='--', label='Fitted Line - Bid')
        axs[0].set_ylabel('Liquidity (for the top 21 levels)')
        axs[0].annotate(f"BidS = {m_bid_count:.2f}", xy=(bidlevelsprice_count[-1], m_bid_count*np.array(bidlevelsprice_count[-1]) + c_bid_count), xytext=(bidlevelsprice_count[-1] + 0.2, m_bid_count*np.array(bidlevelsprice_count[-1]) + c_bid_count + 0.1),
            arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
        axs[0].annotate(f"AskS = {m_ask_count:.2f}", xy=(asklevelsprice_count[-1], m_ask_count*np.array(asklevelsprice_count[-1]) + c_ask_count), xytext=(asklevelsprice_count[-1] - 0.7, m_ask_count*np.array(asklevelsprice_count[-1]) + c_ask_count + 0.1),
            arrowprops=dict(arrowstyle='<-', color='blue'), color='blue')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            axs[0].set_title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count)) 

    elif not asklevelsprice_count and not bidlevelsprice_count:
        print(f"The liquidity-price can be plotted as the book is empty")

    else:
        m_bid_count, c_bid_count = np.linalg.lstsq(np.vstack([bidlevelsprice_count, np.ones(len(bidlevelsprice_count))]).T, bidlevelsvol_count, rcond=None)[0]
        m_ask_count, c_ask_count = np.linalg.lstsq(np.vstack([asklevelsprice_count, np.ones(len(asklevelsprice_count))]).T, asklevelsvol_count, rcond=None)[0] 
        axs[0].scatter(bidlevelsprice_count, bidlevelsvol_count)
        axs[0].scatter(asklevelsprice_count, asklevelsvol_count)
        axs[0].plot(bidlevelsprice_count, m_bid_count*np.array(bidlevelsprice_count) + c_bid_count, linestyle='--', label='Fitted Line - Bid')
        axs[0].plot(asklevelsprice_count, m_ask_count*np.array(asklevelsprice_count) + c_ask_count, linestyle='--', label='Fitted Line - Ask')
        axs[0].set_ylabel('Liquidity (for the top 21 levels)')
        axs[0].annotate(f"BidS = {m_bid_count:.2f}", xy=(bidlevelsprice_count[-1], m_bid_count*np.array(bidlevelsprice_count[-1]) + c_bid_count), xytext=(bidlevelsprice_count[-1] + 0.2, m_bid_count*np.array(bidlevelsprice_count[-1]) + c_bid_count + 0.1),
            arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
        axs[0].annotate(f"AskS = {m_ask_count:.2f}", xy=(asklevelsprice_count[-1], m_ask_count*np.array(asklevelsprice_count[-1]) + c_ask_count), xytext=(asklevelsprice_count[-1] - 0.7, m_ask_count*np.array(asklevelsprice_count[-1]) + c_ask_count + 0.1),
            arrowprops=dict(arrowstyle='<-', color='blue'), color='blue')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            axs[0].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            axs[0].set_title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))  

    if not bidlevelsprice_percent:
        m_bid_percent, c_bid_percent = np.linalg.lstsq(np.vstack([bidlevelsprice_percent, np.ones(len(bidlevelsprice_percent))]).T, bidlevelsvol_percent, rcond=None)[0]
        axs[1].scatter(bidlevelsprice_percent, bidlevelsvol_percent)
        axs[1].plot(bidlevelsprice_percent, m_bid_percent*np.array(bidlevelsprice_percent) + c_bid_percent, linestyle='--', label='Fitted Line - Bid')
        axs[1].set_xlabel('Price')
        axs[1].set_ylabel('Liquidity (for price ranges within 1 percent of best ask/bid)')
        axs[1].annotate(f"BidS = {m_bid_percent:.2f}", xy=(bidlevelsprice_percent[-1], m_bid_percent*np.array(bidlevelsprice_percent[-1]) + c_bid_percent), xytext=(bidlevelsprice_percent[-1] + 0.2, m_bid_percent*np.array(bidlevelsprice_percent[-1]) + c_bid_percent + 0.1),
            arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
        axs[1].annotate(f"AskS = {m_ask_percent:.2f}", xy=(asklevelsprice_percent[-1], m_ask_percent*np.array(asklevelsprice_percent[-1]) + c_ask_percent), xytext=(asklevelsprice_percent[-1] - 0.7, m_ask_percent*np.array(asklevelsprice_percent[-1]) + c_ask_percent + 0.1),
            arrowprops=dict(arrowstyle='<-', color='blue'), color='blue')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            axs[1].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            axs[1].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            axs[1].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            axs[1].set_title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))  

    elif not asklevelsprice_percent:
        m_ask_percent, c_ask_percent = np.linalg.lstsq(np.vstack([asklevelsprice_percent, np.ones(len(asklevelsprice_percent))]).T, asklevelsvol_percent, rcond=None)[0]
        axs[1].scatter(asklevelsprice_percent, asklevelsvol_percent)
        axs[1].plot(asklevelsprice_percent, m_ask_percent*np.array(asklevelsprice_percent) + c_ask_percent, linestyle='--', label='Fitted Line - Ask')
        axs[1].set_xlabel('Price')
        axs[1].set_ylabel('Liquidity (for price ranges within 1 percent of best ask/bid)')
        axs[1].annotate(f"BidS = {m_bid_percent:.2f}", xy=(bidlevelsprice_percent[-1], m_bid_percent*np.array(bidlevelsprice_percent[-1]) + c_bid_percent), xytext=(bidlevelsprice_percent[-1] + 0.2, m_bid_percent*np.array(bidlevelsprice_percent[-1]) + c_bid_percent + 0.1),
            arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
        axs[1].annotate(f"AskS = {m_ask_percent:.2f}", xy=(asklevelsprice_percent[-1], m_ask_percent*np.array(asklevelsprice_percent[-1]) + c_ask_percent), xytext=(asklevelsprice_percent[-1] - 0.7, m_ask_percent*np.array(asklevelsprice_percent[-1]) + c_ask_percent + 0.1),
            arrowprops=dict(arrowstyle='<-', color='blue'), color='blue')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            axs[1].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            axs[1].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            axs[1].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            axs[1].set_title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))  
    
    elif not asklevelsprice_percent and not bidlevelsprice_percent:
        print(f"The liquidity-price can be plotted as the book is empty")
    
    else:
        m_bid_percent, c_bid_percent = np.linalg.lstsq(np.vstack([bidlevelsprice_percent, np.ones(len(bidlevelsprice_percent))]).T, bidlevelsvol_percent, rcond=None)[0]
        m_ask_percent, c_ask_percent = np.linalg.lstsq(np.vstack([asklevelsprice_percent, np.ones(len(asklevelsprice_percent))]).T, asklevelsvol_percent, rcond=None)[0]
        axs[1].scatter(bidlevelsprice_percent, bidlevelsvol_percent)
        axs[1].scatter(asklevelsprice_percent, asklevelsvol_percent)
        axs[1].plot(bidlevelsprice_percent, m_bid_percent*np.array(bidlevelsprice_percent) + c_bid_percent, linestyle='--', label='Fitted Line - Bid')
        axs[1].plot(asklevelsprice_percent, m_ask_percent*np.array(asklevelsprice_percent) + c_ask_percent, linestyle='--', label='Fitted Line - Ask')
        axs[1].set_xlabel('Price')
        axs[1].set_ylabel('Liquidity (for price ranges within 1 percent of best ask/bid)')
        axs[1].annotate(f"BidS = {m_bid_percent:.2f}", xy=(bidlevelsprice_percent[-1], m_bid_percent*np.array(bidlevelsprice_percent[-1]) + c_bid_percent), xytext=(bidlevelsprice_percent[-1] + 0.2, m_bid_percent*np.array(bidlevelsprice_percent[-1]) + c_bid_percent + 0.1),
            arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
        axs[1].annotate(f"AskS = {m_ask_percent:.2f}", xy=(asklevelsprice_percent[-1], m_ask_percent*np.array(asklevelsprice_percent[-1]) + c_ask_percent), xytext=(asklevelsprice_percent[-1] - 0.7, m_ask_percent*np.array(asklevelsprice_percent[-1]) + c_ask_percent + 0.1),
            arrowprops=dict(arrowstyle='<-', color='blue'), color='blue')
        if self.fCount == 0 and self.cCount == 0 and self.nCount !=0:
            axs[1].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.fCount == 0 and self.nCount == 0 and self.cCount != 0:
            axs[1].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        elif self.cCount == 0 and self.nCount == 0 and self.fCount != 0:
            axs[1].set_title('T = {:s}, F = {:.2f} , C = {:.2f}, N = {:.2f}, H = {:.2f}, Count = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'], type_proportion['Noise'], type_proportion['HighFrequency'], total_count))
        else:
            axs[1].set_title('T = {:s}, fS = {:.2f}, cS = {:.2f}, nS = {:.2f}, Hp = {:.2f}, Count = {:d}'.format(agent_type, self.fSD, self.cSD, self.nSD, type_proportion['HighFrequency'], total_count))   
    plt.savefig(os.path.join(out_dir,'price_volume.png'))
    plt.clf()
    plt.close('all')

    # # Generate the sampled fundamental price plot
    # plt.figure()
    # plt.plot(df_pf[0],df_pf[1])
    # plt.xlabel('Timestamp')
    # plt.ylabel('fundamental price')
    # plt.title('Type = {:s}, F_ratio = {:.2f}, C_ratio = {:.2f}, N_ratio = {:.2f}, Total = {:d}'.format(agent_type, type_proportion['Fundamentalist'], type_proportion['Chartist'],type_proportion['Noise'], total_count))
    # plt.savefig(os.path.join(out_dir,'fundamental_price.png'))
    # plt.close('all')

    return df_results, beta_alpha, lag1_SACF

def lsSeries(l2_file, currentTimestamp, reportInterval):
    data_records = []


    with open(l2_file, 'r') as file:
        lines=file.readlines()

        # Iterate in reverse order
        for i in range(len(lines)-1,-1,-1):
            # Split the line by comma
            line_splits = lines[i].strip().split(',')
            if len(line_splits) < 13 or i == len(lines) - 1:
                continue

            # Convert time to seconds
            h, m, s = line_splits[1].split(':')
            seconds = int(h) * 3600 + int(m) * 60 + float(s)

             # Check if the entry is within the time interval
            if seconds >= currentTimestamp-reportInterval: 
                # Parse Bid and Ask levels
                BidLevels_split = line_splits[11].split()
                AskLevels_split = line_splits[12].split()

                # Extract price and volume, skipping zeros
                bidlevelsprice = []
                bidlevelsvol = []
                asklevelsprice = []
                asklevelsvol = []
                
                for _, bidlevel in enumerate(BidLevels_split):
                    quantity, price = bidlevel.strip('()').split('@')
                    quantity, price = convert_to_number(quantity), convert_to_number(price)
                    if quantity and price:
                        bidlevelsprice.append(price)
                        bidlevelsvol.append(quantity)
                
                for _, asklevel in enumerate(AskLevels_split):
                    quantity, price = asklevel.strip('()').split('@')
                    quantity, price = convert_to_number(quantity), convert_to_number(price)
                    if quantity and price:
                        asklevelsprice.append(price)
                        asklevelsvol.append(quantity)

                                # Sort ask levels in ascending order
                ask_sorted_indices = np.argsort(asklevelsprice)
                asklevelsprice_sorted = np.array(asklevelsprice)[ask_sorted_indices]
                asklevelsvol_sorted = np.array(asklevelsvol)[ask_sorted_indices]
                asklevelsvol_cumsum = np.cumsum(asklevelsvol_sorted)  # Cumulative sum for ask volume

                # Sort bid levels in descending order
                bid_sorted_indices = np.argsort(bidlevelsprice)[::-1]
                bidlevelsprice_sorted = np.array(bidlevelsprice)[bid_sorted_indices]
                bidlevelsvol_sorted = np.array(bidlevelsvol)[bid_sorted_indices]
                bidlevelsvol_cumsum = np.cumsum(bidlevelsvol_sorted)  # Cumulative sum for bid volume

                
                # Perform least-squares regression if non-zero data points exist
                if bidlevelsprice_sorted.size > 1:
                    m_bid_count, c_bid_count = np.linalg.lstsq(
                        np.vstack([bidlevelsprice_sorted, np.ones(len(bidlevelsprice_sorted))]).T,
                        bidlevelsvol_cumsum, rcond=None)[0]
                    #m_bid_count_list.append(m_bid_count)
                    #c_bid_count_list.append(c_bid_count)
                else:
                    m_bid_count, c_bid_count = np.nan, np.nan

                if asklevelsprice_sorted.size > 1:
                    m_ask_count, c_ask_count = np.linalg.lstsq(
                        np.vstack([asklevelsprice_sorted, np.ones(len(asklevelsprice_sorted))]).T,
                        asklevelsvol_cumsum, rcond=None)[0]
                    #m_ask_count_list.append(m_ask_count)
                    #c_ask_count_list.append(c_ask_count)
                else:
                    m_ask_count, c_ask_count = np.nan, np.nan
                                # Append the results as a row in data_records
                #data_records.append({
                #    'Timestamp': seconds,
                #    'm_bid': m_bid_count,
                #    'c_bid': c_bid_count,
                #    'm_ask': m_ask_count,
                #    'c_ask': c_ask_count,
                #    'b_bidP':convert_to_number(line_splits[5]),
                #    'b_bidV': convert_to_number(line_splits[4]),
                #    'l_idx': i
                #})
                data_records.append({
                    'Timestamp': seconds,
                    'm_bid': m_bid_count,
                    'c_bid': c_bid_count,
                    'm_ask': m_ask_count,
                    'c_ask': c_ask_count,
                    'l_idx': i
                })

            else:
                break

    # Create a DataFrame from the collected records
    df_results = pd.DataFrame(data_records)
    #return m_bid_count_list, c_bid_count_list, m_ask_count_list, c_ask_count_list
    return df_results

def pricevol_count(l2_file, count):
    if count < 1:
        count = 1
        print(f"The number of LOB levels chosen should be at least 1")

    with open(l2_file, 'r') as file:
        lines=file.readlines()
        line_splits = lines[-2].strip().split(',')

        # Extract the last component containing tuples
        BidLevels_split = line_splits[11].split()
        AskLevels_split = line_splits[12].split()
        bidlevelsprice_count = []
        bidlevelsvol_count = []
        asklevelsprice_count = []
        asklevelsvol_count = []

        idx = 0
        for idx, bidlevel in enumerate(BidLevels_split):
            #if idx < count:
                quantity, price = bidlevel.strip('()').split('@')
                bidlevelsprice_count.append(convert_to_number(price))
                bidlevelsvol_count.append(convert_to_number(quantity))

        zipped_pairs = list(zip(bidlevelsprice_count, bidlevelsvol_count))
        zipped_pairs.sort(key=lambda x: x[0], reverse=True)
        bidlevelsprice_count[:], bidlevelsvol_count[:] = zip(*zipped_pairs)

        idx = 0
        for idx, asklevel in enumerate(AskLevels_split):
            #if idx < count:
                quantity, price = asklevel.strip('()').split('@')
                asklevelsprice_count.append(convert_to_number(price))
                asklevelsvol_count.append(convert_to_number(quantity))
        
        zipped_pairs = list(zip(asklevelsprice_count, asklevelsvol_count))
        zipped_pairs.sort(key=lambda x: x[0])
        asklevelsprice_count[:], asklevelsvol_count[:] = zip(*zipped_pairs)

        return bidlevelsprice_count[:count], np.cumsum(bidlevelsvol_count[:count]), asklevelsprice_count[:count], np.cumsum(asklevelsvol_count[:count])

def pricevol(l2_file, percentage):
    if percentage <= 0:
        percentage = 1e-6
        print(f"The percentage chosen should be greater than 0")

    with open(l2_file, 'r') as file:
        lines=file.readlines()
        line_splits = lines[-2].strip().split(',')

        bestbidprice = convert_to_number(line_splits[5])
        bestaskprice = convert_to_number(line_splits[7])
        bidprice_percent = bestbidprice-percentage*bestbidprice
        askprice_percent = bestaskprice+percentage*bestaskprice

        # Extract the last component containing tuples
        BidLevels_split = line_splits[11].split()
        AskLevels_split = line_splits[12].split()
        bidlevels = []
        for bidlevel in BidLevels_split:
            quantity, price = bidlevel.strip('()').split('@')
            bidlevels.append([convert_to_number(quantity),convert_to_number(price)])
        bidlevels.sort(key=lambda x: x[1],reverse=True) 
        bidlevelsvol_percent = [A for A, B in bidlevels if B >= bidprice_percent]
        bidlevelsprice_percent = [B for A, B in bidlevels if B >= bidprice_percent]

        asklevels = []
        for asklevel in AskLevels_split:
            quantity, price = asklevel.strip('()').split('@')
            asklevels.append([convert_to_number(quantity),convert_to_number(price)])
        asklevels.sort(key=lambda x: x[1])
        asklevelsvol_percent = [A for A, B in asklevels if B <= askprice_percent]
        asklevelsprice_percent = [B for A, B in asklevels if B <= askprice_percent]

        return bidlevels, bidlevelsprice_percent, np.cumsum(bidlevelsvol_percent), asklevels, asklevelsprice_percent, np.cumsum(asklevelsvol_percent)

def lob_stats(bestaskprice,bestbidprice,bestaskvol,bestbidvol):
    midquote = []
    relative_spread=[]
    spread=[]
    for i,j in zip(bestaskprice,bestbidprice):
        if i==None or j==None:
            spread.append(None)
            relative_spread.append(None)
            midquote.append(None)
        else:
            spread.append(i-j)
            relative_spread.append((i-j)/(0.5*(i+j)))
            midquote.append(0.5*(i+j))

    vol_imbalance =[]
    for i,j in zip(bestaskvol,bestbidvol):
        if i==None or j==None:
            vol_imbalance.append(None)
        else:
            vol_imbalance.append((j-i)/(j+i))

    weighted_midquote=[]
    for i,j,k,l in zip(bestaskprice,bestbidprice,bestaskvol,bestbidvol):
        if i==None or j==None:
            weighted_midquote.append(None)
        else:
            weighted_midquote.append((i*k+j*l)/(k+l))

    return spread, relative_spread, midquote, weighted_midquote, vol_imbalance

class DummySTLogger:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

import sys
if __name__ == '__main__':
    # Get the current working directory
    cdir = os.getcwd()
    search_pattern = os.path.join(cdir, 'logs/*')
    dirs = [p for p in glob.glob(search_pattern) if os.path.isdir(p)]
    if not dirs:
        print("No directories found matching the pattern.")
    dirs.sort(key=os.path.getmtime, reverse=True)    
    if len(sys.argv) == 1:
        latest_dir = dirs[0]
    else:
        latest_dir = sys.argv[1]
    search_pattern = os.path.join(latest_dir, '*L2*')
    l2_files = sorted(glob.glob(search_pattern))
    search_pattern = os.path.join(latest_dir, '*L3*')
    l3_files = sorted(glob.glob(search_pattern))
    xml = ET.parse(os.path.join(latest_dir, 'config.xml')).getroot()
    N_levels = 21
    for child in xml.find("Agents"):
        if child.tag == 'InitializationAgent':
            iCount = int(child.attrib['instanceCount'])
        if child.tag == 'StylizedTraderAgent':
            if (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaC'])!= 0) or (float(child.attrib['sigmaN'])!= 0 and float(child.attrib['sigmaC'])!= 0) or (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaN'])!= 0) or (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaN'])!= 0 and float(child.attrib['sigmaC'])!= 0): 
                sCount = int(child.attrib['instanceCount'])
            elif float(child.attrib['sigmaF']) != 0:
                fCount = int(child.attrib['instanceCount'])
            elif float(child.attrib['sigmaC']) != 0:
                cCount = int(child.attrib['instanceCount'])
            elif float(child.attrib['sigmaN']) != 0:
                nCount = int(child.attrib['instanceCount'])
        if child.tag == 'HighFrequencyTraderAgent':
            hCount = int(child.attrib['instanceCount'])
    dummy_agent = DummySTLogger(logDir=latest_dir, iCount=iCount, fCount=fCount, cCount=cCount, nCount=nCount, hCount=hCount, sCount=sCount)
    for l2_file,l3_file in zip(l2_files,l3_files):
        bookId = int(l2_file.split('.')[-2].split('-')[-1])
        out_dir = os.path.join(latest_dir,f"book_{bookId}")
        #print(f'Processing File: {l2_file}')
        #print(f'Out Dir: {out_dir}')
        _,time,_,_,bestbidvol,bestbidprice,bestaskvol,bestaskprice,bidlevels,asklevels, vol_imbalance = process_l2_file(l2_file, N_levels)
        aggressing_agent_id, resting_agent_id, trade_id, volume, price, timestamp = process_l3_file(l3_file)
        trade_volume_type_feed, trade_price_feed = generate_trade_file(dummy_agent, aggressing_agent_id, resting_agent_id, trade_id, volume, price, timestamp)
        generate_plots(dummy_agent, out_dir, l2_file, time,bestbidvol,bestbidprice,bestaskvol,bestaskprice,bidlevels,asklevels, vol_imbalance, trade_volume_type_feed, trade_price_feed,bookId )