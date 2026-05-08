# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, RAYLEIGH RESEARCH OY. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Trader Logger Class."""

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from thesimulator import *
from TraderReporting import *
from Texifier import compile_latex, generate_tex_with_texsoup

#----------------------------------------------------------------------------
class TraderLogger:
    def log(self, message):
        print(f'T={self.currentTimestamp} : {self.name()} | {message}')

    def configure(self, simulation, params):
        print(f" --- Configuring {self.name()}--- ")
        self.currentTimestamp = simulation.currentTimestamp()
        self.logDir = simulation.logDir()
        self.duration = simulation.duration()
        self.bookCount = simulation.bookCount()
        self.xml = ET.parse(os.path.join(self.logDir, 'config.xml')).getroot()
        self.timeScale =  self.timeScaleConverter(self.xml.get('timescale'))  
        self.serverFlag = bool(int(params['serverFlag'])) 
        self.ask_lowest = dict() 
        self.bid_highest = dict()            
        self.TA_config = []
        for child in self.xml.find("Agents"):
            if child.tag == 'InitializationAgent':
                self.iCount = int(child.attrib['instanceCount'])
            if child.tag == 'StylizedTraderAgent':
                self.fCount = 0
                self.cCount = 0
                self.nCount = 0
                if (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaC'])!= 0) or (float(child.attrib['sigmaN'])!= 0 and float(child.attrib['sigmaC'])!= 0) or (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaN'])!= 0) or (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaN'])!= 0 and float(child.attrib['sigmaC'])!= 0): 
                    self.sCount = int(child.attrib['instanceCount'])
                    self.fSD = float(child.attrib['sigmaF'])
                    self.cSD = float(child.attrib['sigmaC'])
                    self.nSD = float(child.attrib['sigmaN'])
                elif float(child.attrib['sigmaF']) != 0:
                    self.fCount = int(child.attrib['instanceCount'])
                elif float(child.attrib['sigmaC']) != 0:
                    self.cCount = int(child.attrib['instanceCount'])
                elif float(child.attrib['sigmaN']) != 0:
                    self.nCount = int(child.attrib['instanceCount'])
                self.TA_config.append(child.attrib)
            if child.tag == 'HighFrequencyTraderAgent':
                self.hCount = int(child.attrib['instanceCount'])
        
        self.exchange = str(params['exchange'])
        self.reportInterval = int(params['reportInterval']) if 'reportInterval' in params else 1000
        self.checkpoints = [block for block in range(max(self.reportInterval,self.currentTimestamp + (self.reportInterval - (self.currentTimestamp % self.reportInterval))),self.duration,self.reportInterval)] if self.reportInterval > 0 else [] 
        print(f'{self.checkpoints=}')
        self.num_buckets = float(params['num_buckets'])
        self.trading_period_seconds = float(params['trading_period_seconds'])
        self.Tinit= self.duration
        self.trades_file = open(os.path.join(self.logDir,'output_trade.csv'), 'a')
        self.trades_file.write(f"Timestamp,BookId,TradeId,Volume,Direction,Price\n")
        #self.trades_file.flush()
        self.trades_volume_period = open(os.path.join(self.logDir,'output_trade_volume_period.csv'), 'a')
        self.trades_volume_period.write(f"TradePeriod,BookId,Count,Sum,Mean,MeanTime,CountTime\n")
        #self.trades_volume_period.flush()
        self.prices= open(os.path.join(self.logDir,'output_price.csv'), 'a')
        self.prices.write(f"Timestamp,BookId,Price\n")
        #self.prices.flush()
        self.parkinsonVolatility = open(os.path.join(self.logDir,'output_parkinson.csv'), 'a')
        self.parkinsonVolatility.write(f"TradePeriod,BookId,ParkVol\n")
        #self.parkinsonVolatility.flush()
        self.VPIN = open(os.path.join(self.logDir,'output_vpin.csv'), 'a')
        self.VPIN.write(f"TradePeriod,BookId,VPIN\n")
        #self.VPIN.flush()
        #self.slope_feed = pd.DataFrame({"BookId": [], "Timestamp": [] ,"m_bid": [],"c_bid": [],"m_ask": [], "c_ask": [], "b_bidP": [], "b_bidV": [], "l_idx":[]})
        self.slope_feed = pd.DataFrame({"BookId": [], "Timestamp": [] ,"m_bid": [],"c_bid": [],"m_ask": [], "c_ask": [], "l_idx":[]})
        self.beta_alpha_param = pd.DataFrame({"BookId": [], "Beta": [], "Alpha": []})
        self.lag1_param = pd.DataFrame({"BookId": [], "Lag1": []})
        self.N_levels = 21
        self.stopSignal = 0
        self.aggregateInventory = open(os.path.join(self.logDir,'output_inventory.csv'), 'a')
        self.aggregateInventory.write(f"Timestamp,BookId,Inventory\n")
        self.l2Offsets = {i : 0 for i in range(self.bookCount)}
        self.l3Offsets = {i : 0 for i in range(self.bookCount)}
        self.l2Cumulative = {}
        self.l3Cumulative = {}

    # receiveMessage method, providing communication with the simulator.
    def receiveMessage(self, simulation, messagetype, payload):
        self.currentTimestamp = simulation.currentTimestamp()
        # Subscribing to trade events occcurring within the simulator.
        match messagetype:
            case "EVENT_SIMULATION_START":
                self.log(f"-----SIMULATION STARTED----")
                simulation.dispatchMessage(self.currentTimestamp, 1, self.name(), self.exchange, "SUBSCRIBE_EVENT_TRADE", EmptyPayload())
                for i in range(self.bookCount):
                    L1Payload = RetrieveL1Payload()
                    L1Payload.bookId = i
                    simulation.dispatchMessage(self.currentTimestamp, 1, self.name(), self.exchange, "RETRIEVE_L1", L1Payload)
            case "RESPONSE_SUBSCRIBE_EVENT_TRADE":
                self.log(f"-----Subscribed to trade events----")
                for i in range(self.bookCount):
                    L1Payload = RetrieveL1Payload()
                    L1Payload.bookId = i
                    simulation.dispatchMessage(self.currentTimestamp, 1, self.name(), self.exchange, "RETRIEVE_L1", L1Payload)
            case "RESPONSE_RETRIEVE_L1":
                if self.currentTimestamp > (self.duration-1):
                        self.prices.close()
                        self.aggregateInventory.close()
                else:
                    bookId = payload.bookId        
                    self.ask_lowest[bookId] = float(payload.bestAskPrice)
                    self.bid_highest[bookId] = float(payload.bestBidPrice)
                    MidPrice = (self.ask_lowest[bookId]+self.bid_highest[bookId])/2
                    if (self.ask_lowest[bookId] != 0 and self.bid_highest[bookId] != 0):
                        self.prices.write(f"{self.currentTimestamp},{payload.bookId},{MidPrice}\n")
                        self.prices.flush()
                    L1Payload = RetrieveL1Payload()
                    L1Payload.bookId = bookId
                    simulation.dispatchMessage(self.currentTimestamp, 1, self.name(), self.exchange, "RETRIEVE_L1", L1Payload)
                    # self.aggregateInventory.write(f"{self.currentTimestamp},{bookId},{simulation.totalHFTInventory(bookId)}\n")
                    # self.aggregateInventory.flush()
            case "EVENT_TRADE":
                if self.currentTimestamp > (self.duration-1):
                    self.trades_file.close()
                else:
                    self.trades_file.write(f"{self.currentTimestamp},{payload.bookId},{payload.trade.id()},{float(payload.trade.volume())},{float(payload.trade.direction())},{float(payload.trade.price())}\n")
                    self.trades_file.flush()
            case _ :
                pass

        if messagetype == "EVENT_SIMULATION_STOP" or (self.checkpoints != [] and self.currentTimestamp >= self.checkpoints[0]):
            if messagetype == "EVENT_SIMULATION_STOP":
                self.log(f"-----The simulation ends now----")
                print(f"{dir(payload)=}")
                volumes = pd.DataFrame({"Timestamp": [], "Volume": [], "Price":[], "Type" : [], "BookId" : []})  
                df_trades = pd.read_csv(os.path.join(self.logDir,'output_trade.csv'))               
                cointegration_test = {"TraceVal": [], "CriticalVal": [], "BookId": []}
                for i in range(self.bookCount):
                    df_Pf = pd.read_csv(os.path.join(self.logDir,'fundamental.csv'), header=0)
                    df_Pf.set_index('Timestamp')
                    df_trades['Timestamp'] = pd.to_numeric(df_trades['Timestamp'], errors='coerce')
                    df_trades = df_trades.dropna(subset=['Timestamp'])
                    df_cointegration = df_trades[(df_trades['Timestamp'] <= self.currentTimestamp) & (df_trades['BookId'] == i)].copy()
                    df_cointegration = df_cointegration.groupby('Timestamp')['Price'].last().reset_index()
                    df_cointegration['PriceF'] = df_cointegration['Timestamp'].map(lambda t: df_Pf[str(i)][t])
                    coint_test_result = coint_johansen(df_cointegration[['Price','PriceF']], det_order=0, k_ar_diff=1) 
                    cointegration_test["BookId"].append(i)
                    cointegration_test["TraceVal"].append(coint_test_result.lr1[0])
                    cointegration_test["CriticalVal"].append(coint_test_result.cvt[0,1]) 
                    #if (coint_test_result.lr1 > coint_test_result.cvt[:,1]).all():
                    if (coint_test_result.lr1[0] > coint_test_result.cvt[0,1]):
                        print(f"Fundamental price and trade price are cointegrated.")
                    else:
                        print(f"Fundamental price and trade price are NOT cointegrated")
                    df_trades_period = df_trades[(df_trades['Timestamp']>= (self.currentTimestamp-self.trading_period_seconds))&(df_trades['Timestamp']<= (self.currentTimestamp))&(df_trades['BookId'] == i)]
                    ParkVol = calculate_parkinson_volatility(self,df_trades_period)
                    self.parkinsonVolatility.write(f"{self.currentTimestamp},{i},{ParkVol}\n")
                    self.parkinsonVolatility.flush()
                    vpin, tradePer = calculate_vpin(self, df_trades_period, self.num_buckets)
                    self.VPIN.write(f"{tradePer},{i},{vpin}\n")
                    self.VPIN.flush()
                    trades_count_period = df_trades_period['Volume'].count()
                    trades_sum_period = df_trades_period['Volume'].sum()
                    trades_mean_period = df_trades_period['Volume'].mean()
                    trades_mean_timestamp_period = df_trades_period.groupby('Timestamp')['Volume'].sum().reset_index()['Volume'].mean()
                    trades_count_timestamp_period = df_trades_period.groupby('Timestamp')['Volume'].count().reset_index()['Volume'].mean()
                    self.trades_volume_period.write(f"{tradePer},{i},{trades_count_period},{trades_sum_period},{trades_mean_period},{trades_mean_timestamp_period},{trades_count_timestamp_period}\n")
                    self.trades_volume_period.flush()
                    del df_trades_period
                existed = os.path.exists(os.path.join(self.logDir,'output_cointegration_test.csv'))
                pd.DataFrame(cointegration_test).to_csv(os.path.join(self.logDir,'output_cointegration_test.csv'), header=not existed, index=False, mode='a')          
            else:
                self.log(f"-----Generating Reports at Step {self.currentTimestamp}----")
              
            search_l2_pattern = os.path.join(self.logDir, '*L2*')
            l2_files = sorted(glob.glob(search_l2_pattern))
            search_l3_pattern = os.path.join(self.logDir, '*L3*')
            l3_files = sorted(glob.glob(search_l3_pattern))
            self.log(f'Processing Files:')
            self.log(','.join(l2_files))
            self.log(','.join(l3_files))

            # Initialize local lists to accumulate data from file processing
            slope_data_list=[]
            beta_alpha_list = []
            lag1_list = []
            volumes_list = [] 

            for l2_file,l3_file in zip(l2_files,l3_files):
                out_dir = os.path.join(self.logDir,f"book_{l2_file.split('.')[-2].split('-')[-1]}")
                os.makedirs(out_dir,exist_ok=True)  
                bookId = int(l2_file.split('.')[-2].split('-')[-1]) 
                 
                # Process the L2 log file 
                try:
                    active_file = l2_file
                    _,time,_,_,bestbidvol,bestbidprice,bestaskvol,bestaskprice,bidlevels,asklevels,vol_imbalance,bidliquidity,askliquidity,offset = process_l2_file(l2_file, self.N_levels, self.l2Offsets[bookId])
                    self.l2Offsets[bookId] = offset
                    if bookId not in self.l2Cumulative:
                        self.l2Cumulative[bookId] = {
                        #    "date": [],
                            "time": [],
                        #    "symbol": [],
                        #    "market": [],
                            "bestbidvol": [],
                            "bestbidprice": [],
                            "bestaskvol": [],
                            "bestaskprice": [],
                            "bidliquidity": [],
                            "askliquidity": [],
                            "vol_imbalance": []
                        }

                    # Append the new data to the cumulative store.
                    #self.l2_cumulative[bookId]["date"].extend(date)
                    self.l2Cumulative[bookId]["time"].extend(time)
                    #self.l2_cumulative[bookId]["symbol"].extend(symbol)
                    #self.l2_cumulative[bookId]["market"].extend(market_new)
                    self.l2Cumulative[bookId]["bestbidvol"].extend(bestbidvol)
                    self.l2Cumulative[bookId]["bestbidprice"].extend(bestbidprice)
                    self.l2Cumulative[bookId]["bestaskvol"].extend(bestaskvol)
                    self.l2Cumulative[bookId]["bestaskprice"].extend(bestaskprice)
                    self.l2Cumulative[bookId]["bidliquidity"].extend(bidliquidity)
                    self.l2Cumulative[bookId]["askliquidity"].extend(askliquidity)
                    self.l2Cumulative[bookId]["vol_imbalance"].extend(vol_imbalance)


                    # Offload if the cumulative data grows too large.
                    self.offload_cumulative_data(bookId, "L2", threshold=10000)
                    full_data_df_l2 = self.get_full_data_for_plotting(bookId, "L2")

                    time = full_data_df_l2["time"].tolist()
                    bestbidvol = full_data_df_l2["bestbidvol"].tolist()
                    bestbidprice = full_data_df_l2["bestbidprice"].tolist()
                    bestaskvol = full_data_df_l2["bestaskvol"].tolist()
                    bestaskprice = full_data_df_l2["bestaskprice"].tolist()
                    bidliquidity = full_data_df_l2["bidliquidity"].tolist()
                    askliquidity = full_data_df_l2["askliquidity"].tolist()
                    vol_imbalance = full_data_df_l2["vol_imbalance"].tolist()
                    active_file = l3_file
                    aggressing_agent_id, resting_agent_id, trade_id, volume, price, timestamp, offset = process_l3_file(l3_file, self.l3Offsets[bookId])
                    self.l3Offsets[bookId] = offset

                    if bookId not in self.l3Cumulative:
                        self.l3Cumulative[bookId] = {
                            "timestamp": [],
                            "price": [],
                            "volume": [],
                            "tradeId": [],
                            "restingAgentId": [],
                            "aggressingAgentId": []
                        }

                    # Append the new data to the cumulative store.
                    self.l3Cumulative[bookId]["timestamp"].extend(timestamp)
                    self.l3Cumulative[bookId]["price"].extend(price)
                    self.l3Cumulative[bookId]["volume"].extend(volume)
                    self.l3Cumulative[bookId]["tradeId"].extend(trade_id)
                    self.l3Cumulative[bookId]["restingAgentId"].extend(resting_agent_id)
                    self.l3Cumulative[bookId]["aggressingAgentId"].extend(aggressing_agent_id)


                    # Offload if the cumulative data grows too large.
                    self.offload_cumulative_data(bookId, "L3", threshold=10000)
                    full_data_df_l3 = self.get_full_data_for_plotting(bookId, "L3")

                    timestamp = full_data_df_l3["timestamp"].tolist()
                    price = full_data_df_l3["price"].tolist()
                    volume = full_data_df_l3["volume"].tolist()
                    trade_id = full_data_df_l3["tradeId"].tolist()
                    resting_agent_id = full_data_df_l3["restingAgentId"].tolist()
                    aggressing_agent_id = full_data_df_l3["aggressingAgentId"].tolist()

                    trade_volume_type_feed, trade_price_feed = generate_trade_file(self, aggressing_agent_id, resting_agent_id, trade_id, volume, price, timestamp)
                    slope_data, beta_alpha_data, lag1_data = generate_plots(self, out_dir, l2_file, time, timestamp,bestbidvol,bestbidprice,bestaskvol,bestaskprice,bidlevels,asklevels,bidliquidity,askliquidity, vol_imbalance, trade_volume_type_feed, trade_price_feed, str(l2_file.split('.')[-2].split('-')[-1]), messagetype)
                    slope_data ['BookId'] = l2_file.split('.')[-2].split('-')[-1]                   
                    slope_data_list.append(slope_data)
                    #self.slope_feed = pd.concat([self.slope_feed , slope_data], ignore_index=True)
                    #del slope_data
                    
                    if messagetype == "EVENT_SIMULATION_STOP":
                        trade_volume_type_feed['BookId'] = l2_file.split('.')[-2].split('-')[-1]
                        volumes_list.append(trade_volume_type_feed)
                        #volumes = pd.concat([volumes, trade_volume_type_feed], ignore_index=True)
                        beta_alpha_data ['BookId'] = l2_file.split('.')[-2].split('-')[-1]
                        beta_alpha_list.append(beta_alpha_data)
                        #self.beta_alpha_param = pd.concat([self.beta_alpha_param , beta_alpha_data], ignore_index=True)
                        lag1_data ['BookId'] = l2_file.split('.')[-2].split('-')[-1]
                        lag1_list.append(lag1_data)
                        #self.lag1_param = pd.concat([self.lag1_param , lag1_data], ignore_index=True)

                    del time, bestbidvol, bestbidprice, bestaskvol, bestaskprice, bidliquidity, askliquidity
                    del bidlevels, asklevels, vol_imbalance, aggressing_agent_id, resting_agent_id
                    del trade_id, volume, price, timestamp, trade_volume_type_feed, trade_price_feed
                    del slope_data, beta_alpha_data, lag1_data
                    del full_data_df_l2, full_data_df_l3

                except Exception as ex:
                    self.log(f"Failed to process file {active_file} : {ex}")                    
                    traceback.print_exc()
                    continue
            #self.slope_feed = pd.concat(slope_data_list, ignore_index=True)
            
            if slope_data_list:
                self.slope_feed = pd.concat(slope_data_list, ignore_index=True)
                self.slope_feed.to_csv(os.path.join(self.logDir, 'slopes.csv'), index=False, mode='a')
                self.slope_feed = pd.DataFrame({"BookId": [], "Timestamp": [], "m_bid": [], "c_bid": [], "m_ask": [], "c_ask": [], "l_idx": []})
            else:
                print('The slope data is empty')
            del slope_data_list

            if messagetype == "EVENT_SIMULATION_STOP":
                if beta_alpha_list:
                    self.beta_alpha_param = pd.concat(beta_alpha_list, ignore_index=True)
                    self.beta_alpha_param.to_csv(os.path.join(self.logDir, 'betaalpha.csv'), index=False, mode='a')
                    self.beta_alpha_param = pd.DataFrame({"BookId": [], "Beta": [], "Alpha": []})
                del beta_alpha_list

                if lag1_list:
                    self.lag1_param = pd.concat(lag1_list, ignore_index=True)
                    self.lag1_param.to_csv(os.path.join(self.logDir, 'lag1.csv'), index=False, mode='a')
                    self.lag1_param = pd.DataFrame({"BookId": [], "Lag1": []})
                del lag1_list

                if volumes_list:
                    volumes = pd.concat(volumes_list, ignore_index=True)
                    volumes.to_csv(os.path.join(self.logDir, 'volumes.csv'), index=False, mode='w')
                del volumes_list



                #volumes.to_csv(os.path.join(self.logDir,'volumes.csv'), index=False, mode='w')
                #existed = os.path.exists(os.path.join(self.logDir,'betaalpha.csv'))
                #self.beta_alpha_param.to_csv(os.path.join(self.logDir,'betaalpha.csv'), header=not existed, index=False, mode='a') 
                #existed = os.path.exists(os.path.join(self.logDir,'lag1.csv'))
                #self.lag1_param.to_csv(os.path.join(self.logDir,'lag1.csv'), header=not existed, index=False, mode='a')    

                if self.serverFlag:
                    for l2_file,l3_file in zip(l2_files,l3_files):
                        out_dir = os.path.join(self.logDir,f"book_{l2_file.split('.')[-2].split('-')[-1]}")
                        template = 'rayleigh-template.tex' # Add folder path later
                        figures_paths = sorted(glob.glob(out_dir + '/*.png'))
                        tex_name = f"rayleigh-sim-report-{l2_file.split('.')[-2].split('-')[-1]}"# Replace with desired output .tex file name
                        output_tex_file = os.path.join(out_dir,f'{tex_name}.tex') 
                        generate_tex_with_texsoup(template, figures_paths, self.xml,self.logDir , output_tex_file)
                        compile_latex(output_tex_file) # Remove if latexmk or pdflatex is not working

                # At the end of simulation, perform cleanup to close all open file handles
                self.cleanup()
            
            self.checkpoints = self.checkpoints[1:]

    def cleanup(self):
        """Close all open file handles to free up system resources."""
        self.log("Cleaning up file handles...")
        #for attr in ['trades_file', 'trades_volume_period', 'prices', 'parkinsonVolatility', 'VPIN', 'aggregateInventory']:
        for attr in ['trades_volume_period', 'parkinsonVolatility', 'VPIN']:
            file_handle = getattr(self, attr, None)
            if file_handle and not file_handle.closed:
                try:
                    file_handle.close()
                    self.log(f"Closed {attr}")
                except Exception as e:
                    self.log(f"Error closing {attr}: {e}")

    def offload_cumulative_data(self, bookID, filetype, threshold=10000):
        """
        Offloads cumulative data for a given L2 file if the number of accumulated rows exceeds a threshold.
        The offloaded data is appended to a CSV file on disk.
        """
        if filetype == "L2":
            cum_data = self.l2Cumulative[bookID]
            filename = "l2_offload"
        elif filetype == "L3":
            cum_data = self.l3Cumulative[bookID]
            filename = "l3_offload"

        if cum_data is None:
            return

        #num_rows = len(cum_data["time"])
        num_rows = len(next(iter(cum_data.values())))
        if num_rows < threshold:
            return  # No need to offload yet.

        # Convert cumulative data to a DataFrame.
        df = pd.DataFrame(cum_data)

        # Define the offload file path.
        offload_file = os.path.join(self.logDir, f"{filename}_{bookID}.csv")
    
        # Append the DataFrame to disk.
        header = not os.path.exists(offload_file)
        df.to_csv(offload_file, mode='a', index=False, header=header)

        # Option 1: Clear all cumulative data for this file.
        if filetype == "L2":
            self.l2Cumulative[bookID] = {k: [] for k in cum_data}
        elif filetype == "L3":
            self.l3Cumulative[bookID] = {k: [] for k in cum_data}
        # Option 2: Or, trim to keep only the most recent rows:
        # keep_rows = 1000
        # for key in cum_data:
        #     self.l2_cumulative[l2_file][key] = cum_data[key][-keep_rows:]
    
        self.log(f"Offloaded {num_rows} rows from {filename}_{bookID} to disk.")


    def load_offloaded_data(self, bookID, filetype):
        """
        Loads offloaded data from disk for a given L2 file.
        Returns a DataFrame containing the previously offloaded data.
        """
        if filetype == "L2":
            filename = "l2_offload"
        elif filetype == "L3":
            filename = "l3_offload"


        offload_file = os.path.join(self.logDir, f"{filename}_{bookID}.csv")
        if os.path.exists(offload_file):
            return pd.read_csv(offload_file)
        else:
            return pd.DataFrame()
        
    def get_full_data_for_plotting(self, bookID, filetype):
        """
        Combines offloaded data on disk and the current cumulative in-memory data
        for a given L2 file and returns a single DataFrame for plotting.
        """
        df_offloaded = self.load_offloaded_data(bookID,filetype)
        if filetype == "L2":
            cum_data = self.l2Cumulative.get(bookID, {})
        elif filetype == "L3":
            cum_data = self.l3Cumulative.get(bookID, {})
        df_cum = pd.DataFrame(cum_data) if cum_data else pd.DataFrame()
        # Concatenate offloaded and in-memory data.
        full_df = pd.concat([df_offloaded, df_cum], ignore_index=True)
        return full_df
    
    def timeScaleConverter(self, timeScale):
        scales = {
            "ns": 1e-9,
            "us": 1e-6,
            "ms": 1e-3,
            "s": 1
        }
        try:
            return scales[timeScale]
        except KeyError:
            raise ValueError("The specified time-scale has not been defined")