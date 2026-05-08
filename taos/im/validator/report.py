# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Prometheus metrics reporting service: publishes validator, simulation, miner,
book, trade, and GenTRX gauges via a FastAPI endpoint over POSIX IPC.
"""
import os
import sys
import traceback
import time
import torch
import psutil
import asyncio
import bittensor as bt
import pandas as pd
import posix_ipc
import mmap
import struct
import msgpack
import argparse    

from typing import Dict
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from taos.im.protocol.models import TradeInfo, MarketSimulationConfig
from taos.im.protocol.events import TradeEvent

from taos.common.utils.prometheus import prometheus
from taos.im.utils import duration_from_timestamp
from prometheus_client import Counter, Gauge, Info, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI
from fastapi.responses import Response
import uvicorn
import threading

class ReportingService:
    def __init__(self, config):
        """
        Initialise the reporting service, setting up IPC channels and Prometheus metrics.

        Creates POSIX message queues and shared memory segments for receiving
        publish requests from the validator, initialises all Prometheus registries,
        and starts the FastAPI metrics server in a background thread.

        Args:
            config: Configuration object with wallet, netuid, and prometheus settings.
        """
        self.config = config
        self.wallet = bt.Wallet(
            path=self.config.wallet.path,
            name=self.config.wallet.name,
            hotkey=self.config.wallet.hotkey
        )
        self.running = True
        self.prometheus_initialized = False
        self.current_sim_id = None
        
        self.request_queue = posix_ipc.MessageQueue(
            "/validator-report-req",
            flags=posix_ipc.O_CREAT,
            max_messages=2,
            max_message_size=1024
        )
        self.response_queue = posix_ipc.MessageQueue(
            "/validator-report-res",
            flags=posix_ipc.O_CREAT,
            max_messages=2,
            max_message_size=1024
        )
        self.request_shm = posix_ipc.SharedMemory(
            "/validator-report-data",
            flags=posix_ipc.O_CREAT,
            size=200 * 1024 * 1024
        )
        self.response_shm = posix_ipc.SharedMemory(
            f"/validator-report-response-data",
            flags=posix_ipc.O_CREAT,
            size=100 * 1024 * 1024
        )
        
        self.request_mem = mmap.mmap(self.request_shm.fd, self.request_shm.size)
        self.response_mem = mmap.mmap(self.response_shm.fd, self.response_shm.size)
        
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.report_executor = ThreadPoolExecutor(max_workers=1)        
        self._init_prometheus()
        
    def _start_metrics_server(self):
        """
        Start a FastAPI server exposing per-registry Prometheus metric endpoints.

        Binds to `config.prometheus.port` on all interfaces. Runs in a daemon
        thread so it shuts down automatically when the process exits.
        """
        app = FastAPI()

        @app.get("/metrics")
        def all_metrics():
            """All metrics combined (backwards compatibility)"""
            output = b''.join([generate_latest(r) for r in self.registries.values()])
            return Response(content=output, media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/validator")
        def validator_metrics():
            """Validator-specific metrics: counters, validator_gauges, neuron_info"""
            return Response(content=generate_latest(self.registry_validator), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/simulation")
        def simulation_metrics():
            """Simulation metrics: simulation_gauges"""
            return Response(content=generate_latest(self.registry_simulation), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/miner")
        def miner_metrics():
            """Miner metrics: miner_gauges, miners"""
            return Response(content=generate_latest(self.registry_miner), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/agent")
        def agent_metrics():
            """Miner metrics: agent_gauges"""
            return Response(content=generate_latest(self.registry_agent), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/books")
        def book_metrics():
            """Book metrics: book_gauges, books"""
            return Response(content=generate_latest(self.registry_books), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/trades")
        def trade_metrics():
            """Trade metrics: trades, miner_trades"""
            return Response(content=generate_latest(self.registry_trades), media_type=CONTENT_TYPE_LATEST)

        @app.get("/metrics/gentrx")
        def gentrx_metrics():
            """GenTRX distributed-training metrics: pool allocation, per-miner EMA scores."""
            return Response(content=generate_latest(self.registry_gentrx), media_type=CONTENT_TYPE_LATEST)

        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=self.config.prometheus.port, log_level="debug")

        self.metrics_server_thread = threading.Thread(target=run_server, daemon=True)
        self.metrics_server_thread.start()
        bt.logging.success(f"Prometheus metrics server started on port {self.config.prometheus_port}")
    
    def _init_prometheus(self):
        """
        Initialise all Prometheus collector registries and metric objects.

        Creates separate registries for validator, simulation, miner, agent, books,
        trades, and gentrx metrics, then starts the metrics HTTP server.
        """
        prometheus(
            config=self.config,
            port=self.config.prometheus.port,
            level=None,
            start_server=False
        )
        self.registry_validator = CollectorRegistry()
        self.registry_simulation = CollectorRegistry()
        self.registry_miner = CollectorRegistry()
        self.registry_agent = CollectorRegistry()
        self.registry_books = CollectorRegistry()
        self.registry_trades = CollectorRegistry()
        self.registry_gentrx = CollectorRegistry()

        self.registries = {
            'validator': self.registry_validator,
            'simulation': self.registry_simulation,
            'miner': self.registry_miner,
            'agent': self.registry_agent,
            'books': self.registry_books,
            'trades': self.registry_trades,
            'gentrx': self.registry_gentrx,
        }

        self.prometheus_counters = Counter('counters', 'Counter summaries for the running validator.', ['wallet', 'netuid', 'sim_id', 'timestamp', 'counter_name'], registry=self.registry_validator)
        self.prometheus_simulation_gauges = Gauge('simulation_gauges', 'Gauge summaries for global simulation metrics.', ['wallet', 'netuid', 'sim_id', 'simulation_gauge_name'], registry=self.registry_simulation)
        self.prometheus_validator_gauges = Gauge('validator_gauges', 'Gauge summaries for validator-related metrics.', ['wallet', 'netuid', 'sim_id', 'validator_gauge_name'], registry=self.registry_validator)
        self.prometheus_miner_gauges = Gauge('miner_gauges', 'Gauge summaries for miner-related metrics.', ['wallet', 'netuid', 'sim_id', 'agent_id', 'miner_gauge_name'], registry=self.registry_miner)
        self.prometheus_book_gauges = Gauge('book_gauges', 'Gauge summaries for book-related metrics.', ['wallet', 'netuid', 'sim_id', 'book_id', 'level', 'book_gauge_name'], registry=self.registry_books)
        self.prometheus_agent_gauges = Gauge('agent_gauges', 'Gauge summaries for agent-related metrics.', ['wallet', 'netuid', 'sim_id', 'book_id', 'agent_id', 'agent_gauge_name'], registry=self.registry_agent)
        self.prometheus_trades = Gauge('trades', 'Gauge summaries for trade metrics.', [
            'wallet', 'netuid', 'sim_id', 'timestamp', 'timestamp_str', 'book_id', 'agent_id', 'trade_id',
            'aggressing_order_id', 'aggressing_agent_id', 'resting_order_id', 'resting_agent_id',
            'maker_fee', 'taker_fee',
            'price', 'volume', 'side', 'trade_gauge_name'], registry=self.registry_trades)
        self.prometheus_miner_trades = Gauge('miner_trades', 'Gauge summaries for agent trade metrics.', [
            'wallet', 'netuid', 'sim_id', 'timestamp', 'timestamp_str', 'book_id', 'uid',
            'role', 'price', 'volume', 'side', 'fee',
            'miner_trade_gauge_name'], registry=self.registry_trades)
        self.prometheus_books = Gauge('books', 'Gauge summaries for book snapshot metrics.', [
            'wallet', 'netuid', 'sim_id', 'timestamp', 'timestamp_str', 'book_id',
            'bid_5', 'bid_vol_5', 'bid_4', 'bid_vol_4', 'bid_3', 'bid_vol_3', 'bid_2', 'bid_vol_2', 'bid_1', 'bid_vol_1',
            'ask_5', 'ask_vol_5', 'ask_4', 'ask_vol_4', 'ask_3', 'ask_vol_3', 'ask_2', 'ask_vol_2', 'ask_1', 'ask_vol_1',
            'book_gauge_name'
        ], registry=self.registry_books)
        self.prometheus_miners = Gauge('miners', 'Gauge summaries for miner metrics.', [
            'wallet', 'netuid', 'sim_id', 'timestamp', 'timestamp_str', 'agent_id',
            'placement', 'base_balance', 'base_loan', 'base_collateral', 'quote_balance', 'quote_loan', 'quote_collateral',
            'inventory_value', 'inventory_value_change', 'pnl', 'pnl_change', 'total_realized_pnl',
            'total_daily_volume', 'min_daily_volume', 'average_daily_volume',
            'total_roundtrip_volume', 'min_roundtrip_volume', 'average_roundtrip_volume',
            'activity_factor', 'pnl_factor',
            'kappa', 'kappa_penalty', 'kappa_score',
            'pnl_score', 'combined_score',
            'unnormalized_score', 'score',
            'miner_gauge_name'
        ], registry=self.registry_miner)
        self.prometheus_info = Info('neuron_info', "Info summaries for the running validator.", ['wallet', 'netuid', 'sim_id'], registry=self.registry_validator)
        self.prometheus_gentrx_gauges = Gauge('gentrx_gauges', 'GenTRX distributed-training validator metrics.', ['wallet', 'netuid', 'sim_id', 'gentrx_gauge_name'], registry=self.registry_gentrx)
        self.prometheus_gentrx_miner_scores = Gauge('gentrx_miner_scores', 'Per-miner GenTRX EMA score (validator-smoothed).', ['wallet', 'netuid', 'sim_id', 'uid'], registry=self.registry_gentrx)
        self.prometheus_gentrx_training = Gauge('gentrx_training', 'GenTRX model training statistics (loss, acceptance, timing).', ['wallet', 'netuid', 'sim_id', 'stat'], registry=self.registry_gentrx)
        self._start_metrics_server()
        self.prometheus_initialized = True
        
    def clear_all_metrics(self):
        """
        Clear all Prometheus metrics across all registries.
        
        This is called when a new simulation starts to prevent stale metrics
        from the previous simulation from persisting in graphs.
        """
        bt.logging.info(f"Clearing all metrics for simulation changeover...")
        start = time.time()
        
        try:
            # Clear all gauge metrics
            self.prometheus_simulation_gauges.clear()
            self.prometheus_validator_gauges.clear()
            self.prometheus_miner_gauges.clear()
            self.prometheus_book_gauges.clear()
            self.prometheus_agent_gauges.clear()
            self.prometheus_trades.clear()
            self.prometheus_miner_trades.clear()
            self.prometheus_books.clear()
            self.prometheus_miners.clear()
            self.prometheus_info.clear()            
            bt.logging.success(f"All metrics cleared ({time.time()-start:.4f}s)")
        except Exception as e:
            bt.logging.error(f"Error clearing metrics: {e}")
            bt.logging.error(traceback.format_exc())
    
    async def run(self):
        """
        Main async event loop for the reporting service.

        Drains any stale IPC messages on startup, then waits for 'publish' or
        'shutdown' commands from the validator via the POSIX request queue,
        dispatching to `publish_metrics` and writing results back via shared memory.
        """
        bt.logging.info("Reporting service started")

        while True:
            try:
                self.request_queue.receive(timeout=0.0)
                bt.logging.warning("Drained stale message from reporting request queue")
            except posix_ipc.BusyError:
                break
        
        while self.running:
            try:
                message, _ = self.request_queue.receive(timeout=1.0)
                command = message.decode('utf-8')
                
                if command == 'publish':
                    read_start = time.time()
                    self.request_mem.seek(0)
                    size_bytes = self.request_mem.read(8)
                    data_size = struct.unpack('Q', size_bytes)[0]
                    request_bytes = self.request_mem.read(data_size)
                    
                    deserialize_start = time.time()
                    data = msgpack.unpackb(request_bytes, raw=False, strict_map_key=False)
                    deserialize_time = time.time() - deserialize_start
                    
                    bt.logging.info(f"Read reporting data ({time.time()-read_start:.4f}s, deserialize={deserialize_time:.4f}s)")

                    await self.publish_metrics(data)
                    
                    result = {
                        'initial_balances_published': self.initial_balances_published,                        
                        'miner_stats': self.miner_stats
                    }
                    write_start = time.time()
                    
                    serialize_start = time.time()
                    result_bytes = msgpack.packb(result, use_bin_type=True)
                    serialize_time = time.time() - serialize_start
                    
                    self.response_mem.seek(0)
                    self.response_mem.write(struct.pack('Q', len(result_bytes)))
                    self.response_mem.write(result_bytes)
                    bt.logging.info(f"Wrote reporting response data ({time.time()-write_start:.4f}s, serialize={serialize_time:.4f}s)")

                    drain_start = time.time()
                    drained = 0
                    while True:
                        try:
                            self.response_queue.receive(timeout=0.0)
                            drained += 1
                        except posix_ipc.BusyError:
                            break
                    if drained > 0:
                        bt.logging.warning(f"Drained {drained} stale reporting response signals ({time.time()-drain_start:.4f}s)")
                    send_start = time.time()
                    max_retries = 3
                    sent = False
                    for attempt in range(max_retries):
                        try:
                            self.response_queue.send(b'ready', timeout=1.0)
                            bt.logging.info(f"Reporting Response signal sent ({time.time()-send_start:.4f}s)")
                            sent = True
                            break
                        except posix_ipc.BusyError:
                            bt.logging.warning(f"Reporting Response queue full, retry {attempt+1}/{max_retries}")
                            try:
                                self.response_queue.receive(timeout=0.0)
                                bt.logging.debug("Drained one more stale message from Reporting Response Queue")
                            except posix_ipc.BusyError:
                                pass
                            time.sleep(0.1)
                        except Exception as e:
                            bt.logging.error(f"Error sending Reporting response signal: {e}")
                            break
                    
                elif command == 'shutdown':
                    bt.logging.info("Shutdown command received")
                    self.running = False
                    
            except posix_ipc.BusyError:
                await asyncio.sleep(0.01)
            except Exception as e:
                bt.logging.error(f"Error in reporting loop: {e}")
                bt.logging.error(traceback.format_exc())
        
        self.cleanup()
    
    async def publish_metrics(self, data):
        """
        Deserialise a reporting payload and publish all Prometheus metrics.

        Clears stale metrics whenever the simulation ID changes, then delegates
        to `report()` to push updated gauges.

        Args:
            data (dict): Decoded msgpack payload from the validator containing
                simulation state, account balances, trade data, and scoring results.
        """
        new_sim_id = data['simulation']['simulation_id']
        
        if self.current_sim_id is None:
            # First run after startup - clear any stale metrics from previous validator instance
            bt.logging.info(
                f"First metrics publish after startup (sim_id={new_sim_id}). "
                f"Clearing all metrics to ensure clean slate..."
            )
            self.clear_all_metrics()
        elif new_sim_id != self.current_sim_id:
            # Simulation ID changed during runtime
            bt.logging.warning(
                f"Simulation ID changed: {self.current_sim_id} → {new_sim_id}. "
                f"Clearing all metrics..."
            )
            self.clear_all_metrics()
        
        self.current_sim_id = new_sim_id
        
        def deserialize_to_nested_dict(d):
            """Convert flat string keys back to nested dict."""
            result = defaultdict(lambda: defaultdict(float))
            for key, vol in d.items():
                uid, book_id = map(int, key.split(':'))
                result[uid][book_id] = vol
            return result

        self.recent_trades = {
            int(bookId): [TradeInfo(**t) for t in trades] 
            for bookId, trades in data['recent_trades'].items()
        }
        self.recent_miner_trades = {
            int(uid): {
                int(bookId): [(TradeEvent(**item['trade']), item['role']) for item in trades]
                for bookId, trades in book_trades.items()
            }
            for uid, book_trades in data['recent_miner_trades'].items()
        }

        self.volume_sums = deserialize_to_nested_dict(data['volume_sums'])
        self.maker_volume_sums = deserialize_to_nested_dict(data['maker_volume_sums'])
        self.taker_volume_sums = deserialize_to_nested_dict(data['taker_volume_sums'])
        self.self_volume_sums = deserialize_to_nested_dict(data['self_volume_sums'])
        self.roundtrip_volume_sums = deserialize_to_nested_dict(data['roundtrip_volume_sums'])
        self.inventory_history = data['inventory_history']

        self.total_realized_pnl = {
            int(uid): pnl for uid, pnl in data['total_realized_pnl'].items()
        }
        self.realized_pnl_by_book = {
            int(uid): {
                int(book_id): pnl 
                for book_id, pnl in books.items()
            }
            for uid, books in data['realized_pnl_by_book'].items()
        }

        for key in ['activity_factors', 'pnl_factors', 'kappa_values',
                    'unnormalized_scores', 'scores', 'miner_stats', 'initial_balances',
                    'initial_balances_published', 'simulation_timestamp', 'step',
                    'step_rates', 'fundamental_price', 'shared_state_rewarding',
                    'current_block', 'uid', 'metagraph_data', 'validator_config']:
            setattr(self, key, data[key])
        self.gentrx_scores = data.get('gentrx_scores', {})
        self.gentrx_enabled = data.get('gentrx_enabled', False)
        self.gentrx_training = data.get('gentrx_training', {})
        
        class SimpleState:
            pass
        self.last_state = SimpleState()
        self.last_state.accounts = data['last_state']['accounts']
        self.last_state.books = data['last_state']['books']
        self.last_state.notices = data['last_state']['notices']
        
        class SimpleMetagraph:
            pass
        self.metagraph = SimpleMetagraph()
        for key, value in self.metagraph_data.items():
            setattr(self.metagraph, key, value)
        
        self.simulation = MarketSimulationConfig(**data['simulation'])
        
        if not self.prometheus_initialized:
            self._init_prometheus()
        
        await report(self)
    
    def pagerduty_alert(self, message, details=None):
        """
        Log a critical alert message (stub — the reporting service has no PagerDuty hook).

        Args:
            message (str): Human-readable alert description.
            details (dict, optional): Additional context to log alongside the message.
        """
        bt.logging.error(f"ALERT: {message}")
        if details:
            bt.logging.error(f"Details: {details}")
    
    def cleanup(self):
        """
        Release all IPC and thread-pool resources held by the reporting service.
        """
        self.request_queue.close()
        self.response_queue.close()
        self.request_mem.close()
        self.request_shm.close_fd()
        self.thread_pool.shutdown(wait=True)
        self.report_executor.shutdown(wait=True)

def publish_validator_gauges(self: ReportingService):
    """
    Publishes validator-specific metrics to Prometheus gauges.
    
    Metrics include validator metagraph information (UID, stake, trust, dividends, emission, 
    last update, active status) and system resource usage (CPU, RAM, disk).
    
    Args:
        self (ReportingService): The intelligent markets simulation validator instance
        
    Returns:
        None
    """
    bt.logging.debug(f"Publishing validator metrics...")
    start = time.time()
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="uid").set( self.uid )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="stake").set( self.metagraph.stake[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="validator_trust").set( self.metagraph.validator_trust[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="dividends").set( self.metagraph.dividends[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="emission").set( self.metagraph.emission[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="last_update").set( self.current_block - self.metagraph.last_update[self.uid] )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="active").set( self.metagraph.active[self.uid] )
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    disk_info = psutil.disk_usage('/')
    disk_usage = disk_info.percent
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="cpu_usage_percent").set( cpu_usage )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="ram_usage_percent").set( memory_usage )
    self.prometheus_validator_gauges.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id, validator_gauge_name="disk_usage_percent").set( disk_usage )    
    bt.logging.debug(f"Validator metrics published ({time.time()-start:.4f}s).")

def publish_gentrx_gauges(self: ReportingService) -> None:
    """
    Publish GenTRX distributed-training metrics to the gentrx Prometheus registry.

    Exposes pool-level gauges (enabled flag, simulation share, active miner count),
    per-miner EMA-smoothed scores, and training statistics (loss, acceptance rate,
    checkpoint version, round count, rollback count, and timing).

    Args:
        self (ReportingService): The reporting service instance holding Prometheus
            gauge objects and the current gentrx_scores / gentrx_training state.
    """
    wallet_addr = self.wallet.hotkey.ss58_address
    netuid = self.config.netuid
    simid = self.simulation.simulation_id

    gentrx_scores = getattr(self, 'gentrx_scores', {}) or {}
    gentrx_enabled = getattr(self, 'gentrx_enabled', False)
    gentrx_share = self.validator_config.get('scoring', {}).get('gentrx_simulation_share', 0.0)
    active_miners = sum(1 for s in gentrx_scores.values() if s > 0)

    g = self.prometheus_gentrx_gauges
    g.labels(wallet=wallet_addr, netuid=netuid, sim_id=simid, gentrx_gauge_name="enabled").set(1 if gentrx_enabled else 0)
    g.labels(wallet=wallet_addr, netuid=netuid, sim_id=simid, gentrx_gauge_name="simulation_share").set(gentrx_share)
    g.labels(wallet=wallet_addr, netuid=netuid, sim_id=simid, gentrx_gauge_name="active_miners").set(active_miners)

    ms = self.prometheus_gentrx_miner_scores
    for uid_str, score in gentrx_scores.items():
        ms.labels(wallet=wallet_addr, netuid=netuid, sim_id=simid, uid=str(uid_str)).set(float(score))

    # Training stats from last aggregation round
    tr = getattr(self, 'gentrx_training', {}) or {}
    if tr:
        t = self.prometheus_gentrx_training
        lv = dict(wallet=wallet_addr, netuid=netuid, sim_id=simid)
        for stat, val in (
            ("loss_before",           tr.get("loss_before")),
            ("loss_after",            tr.get("loss_after")),
            ("loss_delta",            (tr["loss_before"] - tr["loss_after"])
                                      if tr.get("loss_before") is not None and tr.get("loss_after") is not None
                                      else None),
            ("n_scored",              tr.get("n_scored")),
            ("n_accepted",            tr.get("n_accepted")),
            ("acceptance_rate",       (tr["n_accepted"] / tr["n_scored"])
                                      if tr.get("n_scored") else None),
            ("version",               tr.get("version")),
            ("agg_round",             tr.get("round")),
            ("rolled_back",           1 if tr.get("rolled_back") else 0),
            ("rounds_aggregated_total", tr.get("rounds_aggregated_total")),
            ("rollbacks_total",       tr.get("rollbacks_total")),
            ("t_score_s",             tr.get("t_score_s")),
            ("t_aggregate_s",         tr.get("t_aggregate_s")),
            ("t_total_s",             tr.get("t_total_s")),
        ):
            if val is not None:
                t.labels(**lv, stat=stat).set(float(val))


def publish_info(self: ReportingService) -> None:
    """
    Publishes static simulation and validator information metrics

    Args:
        self (ReportingService): The intelligent markets simulation validator.
    Returns:
        None
    """
    prometheus_info = {
        'uid': str(self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )) if self.wallet.hotkey.ss58_address in self.metagraph.hotkeys else -1,
        'network': self.config.subtensor.network,
        'coldkey': str(self.wallet.coldkeypub.ss58_address),
        'coldkey_name': self.config.wallet.name,
        'hotkey': str(self.wallet.hotkey.ss58_address),
        'name': self.config.wallet.hotkey
    } | {
        f"config_scoring_{name}": str(value)
        for name, value in self.validator_config['scoring'].items()
    } | {
         f"simulation_{name}" : str(value) for name, value in self.simulation.model_dump().items() if name != 'logDir' and name != 'fee_policy'
    } | self.simulation.fee_policy.to_prom_info()
    self.prometheus_info.labels( wallet=self.wallet.hotkey.ss58_address, netuid=self.config.netuid, sim_id=self.simulation.simulation_id ).info (prometheus_info)
    publish_validator_gauges(self)
    publish_gentrx_gauges(self)

def _set_if_changed(gauge, value, *labels):
    """
    Sets a Prometheus gauge value only if it differs from the current value.
    
    Args:
        gauge: Prometheus gauge object to update
        value: New value to set on the gauge
        *labels: Variable number of positional label values for the gauge
        
    Returns:
        None
    """
    try:
        current = gauge.labels(*labels)._value.get()
        if current != value:
            gauge.labels(*labels).set(value)
    except KeyError:
        gauge.labels(*labels).set(value)

def _set_if_changed_metric(gauge, value, **labels):
    """
    Sets a Prometheus gauge value only if it differs from the current value using keyword labels.
    
    Args:
        gauge: Prometheus gauge object to update
        value: New value to set on the gauge
        **labels: Variable number of keyword label-value pairs for the gauge
        
    Returns:
        None
    """
    try:
        current = gauge.labels(**labels)._value.get()
    except KeyError:
        current = None
    if current != value:
        gauge.labels(**labels).set(value)

def report_worker(validator_data: Dict, state_data: Dict) -> Dict:
    """
    Compute per-miner and per-book metrics from a snapshot of validator state.

    Runs in a thread-pool executor to avoid blocking the reporting event loop.

    Args:
        validator_data (Dict): Snapshot of validator state including volume sums,
            inventory history, realized P&L, activity/pnl factors, kappa values,
            scores, and simulation config.
        state_data (Dict): Current simulator state containing accounts, books,
            and notices.

    Returns:
        Dict: Result dict with keys 'metrics' (computed miner/book data),
            'updated_stats' (unused), and 'error' (str or None on failure).
    """
    result = {
        'metrics': {},
        'updated_stats': {},
        'error': None
    }
    try:
        simulation_timestamp = validator_data['simulation_timestamp']
        step = validator_data['step']
        accounts = state_data['accounts']
        books = state_data['books']
        if not accounts:
            return result
        
        total_realized_pnl = validator_data['total_realized_pnl']
        realized_pnl_by_book = validator_data['realized_pnl_by_book']

        volume_sums = validator_data['volume_sums']
        maker_volume_sums = validator_data['maker_volume_sums']
        taker_volume_sums = validator_data['taker_volume_sums']
        self_volume_sums = validator_data['self_volume_sums']
        roundtrip_volume_sums = validator_data['roundtrip_volume_sums']
        volume_decimals = validator_data['simulation_config']['volumeDecimals']

        daily_volumes = {}
        for agentId in accounts.keys():
            daily_volumes[agentId] = {}
            for bookId in range(validator_data['book_count']):
                total_vol = volume_sums.get(agentId, {}).get(bookId, 0.0)
                total_maker_vol = maker_volume_sums.get(agentId, {}).get(bookId, 0.0)
                total_taker_vol = taker_volume_sums.get(agentId, {}).get(bookId, 0.0)
                total_self_vol = self_volume_sums.get(agentId, {}).get(bookId, 0.0)
                daily_volumes[agentId][bookId] = {
                    'total': total_vol,
                    'maker': total_maker_vol,
                    'taker': total_taker_vol,
                    'self': total_self_vol,
                }

        daily_roundtrip_volumes = {}
        for agentId in accounts.keys():
            daily_roundtrip_volumes[agentId] = {}
            for bookId in range(validator_data['book_count']):
                roundtrip_vol = roundtrip_volume_sums.get(agentId, {}).get(bookId, 0.0)
                daily_roundtrip_volumes[agentId][bookId] = roundtrip_vol
        
        inventory_history = validator_data['inventory_history']
        total_inventory_history = {}
        pnl = {}

        for agentId in accounts.keys():
            if agentId < 0:
                continue
            if agentId not in inventory_history or not inventory_history[agentId]:
                continue
            if len(inventory_history[agentId]) < 2:
                continue
            
            total_inventory_history[agentId] = [
                sum(list(inventory_value.values()))
                for inventory_value in list(inventory_history[agentId].values())
            ]
            pnl[agentId] = total_inventory_history[agentId][-1] - total_inventory_history[agentId][0]

        scores = torch.FloatTensor(list(validator_data['scores'].values()))
        indices = scores.argsort(dim=-1, descending=True)
        placements = torch.empty_like(indices).scatter_(
            -1, indices, torch.arange(scores.size(-1), device=scores.device)
        )

        miner_metrics = {}
        for agentId, accounts_data in accounts.items():
            if agentId < 0:
                continue
            if agentId not in total_inventory_history:
                continue

            base_decimals = validator_data['simulation_config']['baseDecimals']
            quote_decimals = validator_data['simulation_config']['quoteDecimals']

            total_base_balance = round(
                sum([accounts_data[bookId]['bb']['t'] for bookId in books]),
                base_decimals
            )
            total_base_loan = round(
                sum([accounts_data[bookId]['bl'] for bookId in books]),
                base_decimals
            )
            total_base_collateral = round(
                sum([accounts_data[bookId]['bc'] for bookId in books]),
                base_decimals
            )
            total_quote_balance = round(
                sum([accounts_data[bookId]['qb']['t'] for bookId in books]),
                quote_decimals
            )
            total_quote_loan = round(
                sum([accounts_data[bookId]['ql'] for bookId in books]),
                quote_decimals
            )
            total_quote_collateral = round(
                sum([accounts_data[bookId]['qc'] for bookId in books]),
                quote_decimals
            )

            total_daily_volume = {
                role: round(
                    sum([book_volume[role] for book_volume in daily_volumes[agentId].values()]),
                    volume_decimals
                )
                for role in ['total', 'maker', 'taker', 'self']
            }

            average_daily_volume = {
                role: round(
                    total_daily_volume[role] / len(daily_volumes[agentId]),
                    volume_decimals
                )
                for role in ['total', 'maker', 'taker', 'self']
            }

            min_daily_volume = {
                role: min([book_volume[role] for book_volume in daily_volumes[agentId].values()])
                for role in ['total', 'maker', 'taker', 'self']
            }

            total_roundtrip_volume = round(
                sum(daily_roundtrip_volumes[agentId].values()),
                volume_decimals
            )
            average_roundtrip_volume = round(
                total_roundtrip_volume / len(daily_roundtrip_volumes[agentId]),
                volume_decimals
            )
            min_roundtrip_volume = min(daily_roundtrip_volumes[agentId].values()) if daily_roundtrip_volumes[agentId] else 0.0

            activity_factor = (
                sum(validator_data['activity_factors'][agentId].values()) /
                len(validator_data['activity_factors'][agentId])
            )
            pnl_factor = (
                sum(validator_data['pnl_factors'][agentId].values()) /
                len(validator_data['pnl_factors'][agentId])
            )
            kappa_values = validator_data['kappa_values'][agentId] if agentId in validator_data['kappa_values'] else None

            miner_metrics[agentId] = {
                'total_base_balance': total_base_balance,
                'total_base_loan': total_base_loan,
                'total_base_collateral': total_base_collateral,
                'total_quote_balance': total_quote_balance,
                'total_quote_loan': total_quote_loan,
                'total_quote_collateral': total_quote_collateral,
                'total_inventory_value': total_inventory_history[agentId][-1],
                'inventory_value_change': (
                    total_inventory_history[agentId][-1] - total_inventory_history[agentId][-2]
                    if len(total_inventory_history[agentId]) > 1 else 0.0
                ),
                'pnl': pnl[agentId],
                'pnl_change': (
                    pnl[agentId] - (total_inventory_history[agentId][-2] - total_inventory_history[agentId][0])
                    if len(total_inventory_history[agentId]) > 1 else 0.0
                ),
                'total_realized_pnl': total_realized_pnl.get(agentId, 0.0),
                'total_daily_volume': total_daily_volume,
                'average_daily_volume': average_daily_volume,
                'min_daily_volume': min_daily_volume,
                'total_roundtrip_volume': total_roundtrip_volume,
                'average_roundtrip_volume': average_roundtrip_volume,
                'min_roundtrip_volume': min_roundtrip_volume,
                'activity_factor': activity_factor,
                'pnl_factor': pnl_factor,
                'kappa': kappa_values['median'] if kappa_values else None,
                'kappa_penalty': kappa_values.get('penalty') if kappa_values else None,
                'activity_weighted_normalized_median': kappa_values.get('activity_weighted_normalized_median') if kappa_values else None,
                'kappa_score': kappa_values.get('score') if kappa_values else None,
                'pnl_score': kappa_values.get('pnl_score') if kappa_values else None,
                'combined_score': kappa_values.get('final_score') if kappa_values else None,
                'unnormalized_score': validator_data['unnormalized_scores'].get(agentId, 0.0),
                'score': scores[agentId].item() if agentId < len(scores) else 0.0,
                'gentrx_score': float(validator_data.get('gentrx_scores', {}).get(agentId, 0.0)),
                'placement': placements[agentId].item() if agentId < len(placements) else len(scores),
            }

        result['metrics'] = {
            'miner_metrics': miner_metrics,
            'daily_volumes': daily_volumes,
            'daily_roundtrip_volumes': daily_roundtrip_volumes,
            'total_inventory_history': total_inventory_history,
            'total_realized_pnl': total_realized_pnl,
            'realized_pnl_by_book': realized_pnl_by_book,
            'pnl': pnl,
            'scores': scores.tolist(),
            'placements': placements.tolist(),
        }
    except Exception as ex:
        result['error'] = str(ex)
        result['traceback'] = traceback.format_exc()
    return result


async def report(self: ReportingService) -> None:
    """
    Calculates and publishes metrics related to simulation state, validator and agent performance.

    Args:
        self (ReportingService): The intelligent markets simulation validator.
    Returns:
        None
    """
    try:
        self.shared_state_reporting = True
        report_step = self.step
        simulation_duration = duration_from_timestamp(self.simulation_timestamp)
        bt.logging.info(f"Publishing Metrics at Step {self.step} ({simulation_duration})...")
        report_start = time.time()
        updates = deque()    
        bt.logging.debug(f"Collecting simulation metrics...")
        start = time.time()
        
        agent_gauges = self.prometheus_agent_gauges
        book_gauges = self.prometheus_book_gauges
        miner_gauges = self.prometheus_miner_gauges
        wallet_addr = self.wallet.hotkey.ss58_address
        netuid = self.config.netuid
        simid = self.simulation.simulation_id

        updates.append((
            self.prometheus_simulation_gauges,
            self.simulation_timestamp,
            wallet_addr,
            netuid,
            simid,
            "timestamp"
        ))

        updates.append((
            self.prometheus_simulation_gauges,
            sum(self.step_rates) / len(self.step_rates) if len(self.step_rates) > 0 else 0,
            wallet_addr,
            netuid,
            simid,
            "step_rate"
        ))
        bt.logging.debug(f"Simulation metrics collected ({time.time()-start:.4f}s).")

        has_new_trades = False
        has_new_miner_trades = False

        publish_info(self)

        bt.logging.debug(f"Collecting book metrics...")
        book_start = time.time()
        for bookId, book in self.last_state.books.items():
            if book['b']:
                bid_cumsum = 0
                for i, level in enumerate(book['b']):
                    updates.append((book_gauges, level['p'],
                        wallet_addr, netuid, simid, bookId, i, "bid"))
                    updates.append((book_gauges, level['q'],
                        wallet_addr, netuid, simid, bookId, i, "bid_vol"))
                    bid_cumsum += level['q']
                    updates.append((book_gauges, bid_cumsum,
                        wallet_addr, netuid, simid, bookId, i, "bid_vol_sum"))
                    if i == 20: break
            if book['a']:
                ask_cumsum = 0
                for i, level in enumerate(book['a']):
                    updates.append((book_gauges, level['p'],
                        wallet_addr, netuid, simid, bookId, i, "ask"))
                    updates.append((book_gauges, level['q'],
                        wallet_addr, netuid, simid, bookId, i, "ask_vol"))
                    ask_cumsum += level['q']
                    updates.append((book_gauges, ask_cumsum,
                        wallet_addr, netuid, simid, bookId, i, "ask_vol_sum"))
                    if i == 20: break
            if book['b'] and book['a']:
                mid = (book['b'][0]['p'] + book['a'][0]['p']) / 2
                updates.append((book_gauges, mid,
                    wallet_addr, netuid, simid, bookId, 0, "mid"))

                def get_price(side, idx):
                    if side == 'bid':
                        return book['b'][idx]['p'] if len(book['b']) > idx else 0
                    if side == 'ask':
                        return book['a'][idx]['p'] if len(book['a']) > idx else 0

                def get_vol(side, idx):
                    if side == 'bid':
                        return book['b'][idx]['q'] if len(book['b']) > idx else 0
                    if side == 'ask':
                        return book['a'][idx]['q'] if len(book['a']) > idx else 0

                updates.append((self.prometheus_books, 1.0,
                    wallet_addr, netuid, simid, self.simulation_timestamp, simulation_duration, bookId,
                    get_price('bid',4), get_vol('bid',4), get_price('bid',3), get_vol('bid',3), get_price('bid',2), get_vol('bid',2),
                    get_price('bid',1), get_vol('bid',1), get_price('bid',0), get_vol('bid',0),
                    get_price('ask',4), get_vol('ask',4), get_price('ask',3), get_vol('ask',3), get_price('ask',2), get_vol('ask',2),
                    get_price('ask',1), get_vol('ask',1), get_price('ask',0), get_vol('ask',0),
                    "books"
                ))
            if book['e']:
                trades = [event for event in book['e'] if event['y'] == 't']
                if trades:
                    last_trade = trades[-1]
                    if isinstance(self.fundamental_price[0], pd.Series):
                        updates.append((book_gauges,
                            self.fundamental_price[bookId].iloc[-1],
                            wallet_addr, netuid, simid, bookId, 0, "fundamental_price"))
                    else:
                        if self.fundamental_price[bookId]:
                            updates.append((book_gauges,
                                self.fundamental_price[bookId],
                                wallet_addr, netuid, simid, bookId, 0, "fundamental_price"))
                        else:
                            try:
                                book_gauges.remove(wallet_addr, netuid, simid, bookId, 0, "fundamental_price")
                            except KeyError:
                                pass

                    updates.append((book_gauges, last_trade['p'],
                        wallet_addr, netuid, simid, bookId, 0, "trade_price"))
                    updates.append((book_gauges, sum([trade['q'] for trade in trades]),
                        wallet_addr, netuid, simid, bookId, 0, "trade_volume"))
                    updates.append((book_gauges, sum([trade['q'] for trade in trades if trade['s'] == 0]),
                        wallet_addr, netuid, simid, bookId, 0, "trade_buy_volume"))
                    updates.append((book_gauges, sum([trade['q'] for trade in trades if trade['s'] == 1]),
                        wallet_addr, netuid, simid, bookId, 0, "trade_sell_volume"))

                    has_new_trades = True
            if self.simulation.fee_policy.fee_type == 'dynamic':
                DISMTR = self.last_state.books[bookId]['mtr']
                DISmakerRate = self.last_state.accounts[0][bookId]['f']['m']
                DIStakerRate = self.last_state.accounts[0][bookId]['f']['t']
                updates.append((book_gauges, DISmakerRate,
                        wallet_addr, netuid, simid, bookId, 0, "dynamic_maker_rate"))
                updates.append((book_gauges, DIStakerRate,
                        wallet_addr, netuid, simid, bookId, 0, "dynamic_taker_rate"))
                updates.append((book_gauges, DISMTR,
                        wallet_addr, netuid, simid, bookId, 0, "maker_taker_ratio"))
        bt.logging.debug(f"Book metrics collected ({time.time()-book_start:.4f}s).")

        if has_new_trades:
            bt.logging.debug(f"Collecting trade metrics...")
            start = time.time()
            for bookId, trades in self.recent_trades.items():
                for trade in trades:
                    updates.append((self.prometheus_trades, 1.0,
                        wallet_addr, netuid, simid, trade.timestamp, duration_from_timestamp(trade.timestamp),
                        bookId, trade.taker_agent_id, trade.id, trade.taker_id, trade.taker_agent_id, trade.maker_id, trade.maker_agent_id,
                        trade.maker_fee, trade.taker_fee, trade.price, trade.quantity, trade.side, "trades"))

            bt.logging.debug(f"Trade metrics collected ({time.time()-start:.4f}s).")

        if not self.last_state.accounts:
            bt.logging.info(f"Applying {len(updates)} metric updates...")
            apply_start = time.time()
            for update in updates:
                _set_if_changed(*update)
            bt.logging.info(f"Applied {len(updates)} updates in {time.time()-apply_start:.4f}s")
            bt.logging.info(f"Metrics Published for Step {report_step} ({time.time()-report_start}s).")
            return
            
        bt.logging.debug(f"Computing miner metrics in worker process...")
        computation_start = time.time()
        volume_sums_snapshot = {uid: dict(books) for uid, books in self.volume_sums.items()}
        maker_volume_sums_snapshot = {uid: dict(books) for uid, books in self.maker_volume_sums.items()}
        taker_volume_sums_snapshot = {uid: dict(books) for uid, books in self.taker_volume_sums.items()}
        self_volume_sums_snapshot = {uid: dict(books) for uid, books in self.self_volume_sums.items()}
    
        validator_data = {
            'simulation_timestamp': self.simulation_timestamp,
            'step': self.step,
            'volume_sums': volume_sums_snapshot,
            'maker_volume_sums': maker_volume_sums_snapshot,
            'taker_volume_sums': taker_volume_sums_snapshot,
            'self_volume_sums': self_volume_sums_snapshot,
            'roundtrip_volume_sums': {uid: dict(books) for uid, books in self.roundtrip_volume_sums.items()},
            'inventory_history': self.inventory_history,
            'total_realized_pnl': self.total_realized_pnl,
            'realized_pnl_by_book': self.realized_pnl_by_book,
            'activity_factors': self.activity_factors,
            'pnl_factors': self.pnl_factors,
            'kappa_values': self.kappa_values,
            'unnormalized_scores': self.unnormalized_scores,
            'scores': self.scores,
            'book_count': self.simulation.book_count,
            'simulation_config': {
                'volumeDecimals': self.simulation.volumeDecimals,
                'baseDecimals': self.simulation.baseDecimals,
                'quoteDecimals': self.simulation.quoteDecimals,
            }
        }

        state_data = {
            'accounts': self.last_state.accounts,
            'books': self.last_state.books,
            'notices': self.last_state.notices,
        }
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self.report_executor, report_worker, validator_data, state_data)
        while not future.done():
            await asyncio.sleep(0.001)
        result = future.result()

        if result['error']:
            bt.logging.error(f"Error in report worker: {result['error']}\n{result.get('traceback', 'N/A')}")
            return

        bt.logging.debug(f"Miner metrics computed ({time.time()-computation_start:.4f}s).")

        metrics = result['metrics']
        miner_metrics = metrics['miner_metrics']
        daily_volumes = metrics['daily_volumes']
        daily_roundtrip_volumes = metrics['daily_roundtrip_volumes']
        self.realized_pnl_by_book = metrics['realized_pnl_by_book']

        bt.logging.debug(f"Collecting agent book metrics...")
        start = time.time()

        bt.logging.debug(f"Pre-extracting inventory/kappa data...")
        extract_start = time.time()

        start_inventories = {}
        last_inventories = {}
        kappa_data = {}
        for agentId in self.last_state.accounts.keys():
            if agentId < 0:
                continue
            if agentId not in self.inventory_history or not self.inventory_history[agentId]:
                continue
            if len(self.inventory_history[agentId]) < 2:
                continue
            inv_values = list(self.inventory_history[agentId].values())
            start_inventories[agentId] = [i for i in inv_values if len(i) > 0][0]
            last_inventories[agentId] = inv_values[-1]
            kappa_data[agentId] = self.kappa_values.get(agentId)
        bt.logging.debug(f"Pre-extraction complete ({time.time()-extract_start:.4f}s)")

        for agentId, accounts in self.last_state.accounts.items():
            initial_balance_publish_status = {bookId: False for bookId in range(self.simulation.book_count)}
            for bookId, account in accounts.items():
                if agentId in self.initial_balances and self.initial_balances[agentId][bookId]['BASE'] is not None and not self.initial_balances_published.get(agentId, False):
                    updates.append((agent_gauges, self.initial_balances[agentId][bookId]['BASE'],
                        wallet_addr, netuid, simid, bookId, agentId, "base_balance_initial"))
                    updates.append((agent_gauges, self.initial_balances[agentId][bookId]['QUOTE'],
                        wallet_addr, netuid, simid, bookId, agentId, "quote_balance_initial"))
                    updates.append((agent_gauges, self.initial_balances[agentId][bookId]['WEALTH'],
                        wallet_addr, netuid, simid, bookId, agentId, "wealth_initial"))
                    initial_balance_publish_status[bookId] = True
            if all(initial_balance_publish_status.values()):
                self.initial_balances_published[agentId] = True

            if agentId not in start_inventories:
                continue

            start_inv = start_inventories[agentId]
            last_inv = last_inventories[agentId]
            kappas = kappa_data[agentId]

            for bookId, account in accounts.items():
                updates.append((agent_gauges, account['bb']['t'], wallet_addr, netuid, simid, bookId, agentId, "base_balance_total"))
                updates.append((agent_gauges, account['bb']['f'], wallet_addr, netuid, simid, bookId, agentId, "base_balance_free"))
                updates.append((agent_gauges, account['bb']['r'], wallet_addr, netuid, simid, bookId, agentId, "base_balance_reserved"))
                updates.append((agent_gauges, account['qb']['t'], wallet_addr, netuid, simid, bookId, agentId, "quote_balance_total"))
                updates.append((agent_gauges, account['qb']['f'], wallet_addr, netuid, simid, bookId, agentId, "quote_balance_free"))
                updates.append((agent_gauges, account['qb']['r'], wallet_addr, netuid, simid, bookId, agentId, "quote_balance_reserved"))
                updates.append((agent_gauges, account['bl'], wallet_addr, netuid, simid, bookId, agentId, "base_loan"))
                updates.append((agent_gauges, account['bc'], wallet_addr, netuid, simid, bookId, agentId, "base_collateral"))
                updates.append((agent_gauges, account['ql'], wallet_addr, netuid, simid, bookId, agentId, "quote_loan"))
                updates.append((agent_gauges, account['qc'], wallet_addr, netuid, simid, bookId, agentId, "quote_collateral"))
                if account['f']['v']:
                    updates.append((agent_gauges, account['f']['v'], wallet_addr, netuid, simid, bookId, agentId, "fees_traded_volume"))
                updates.append((agent_gauges, account['f']['m'], wallet_addr, netuid, simid, bookId, agentId, "fees_maker_rate"))
                updates.append((agent_gauges, account['f']['t'], wallet_addr, netuid, simid, bookId, agentId, "fees_taker_rate"))
                updates.append((agent_gauges, last_inv[bookId], wallet_addr, netuid, simid, bookId, agentId, "inventory_value"))
                updates.append((agent_gauges, last_inv[bookId] - start_inv[bookId], wallet_addr, netuid, simid, bookId, agentId, "pnl"))
                if agentId in self.realized_pnl_by_book:
                    book_realized_pnl = self.realized_pnl_by_book[agentId].get(bookId, 0.0)
                    updates.append((agent_gauges, book_realized_pnl, wallet_addr, netuid, simid, bookId, agentId, "realized_pnl"))
                else:
                    updates.append((agent_gauges, 0.0, wallet_addr, netuid, simid, bookId, agentId, "realized_pnl"))
                updates.append((agent_gauges, daily_volumes[agentId][bookId]['total'], wallet_addr, netuid, simid, bookId, agentId, "daily_volume"))
                updates.append((agent_gauges, daily_volumes[agentId][bookId]['maker'], wallet_addr, netuid, simid, bookId, agentId, "daily_maker_volume"))
                updates.append((agent_gauges, daily_volumes[agentId][bookId]['taker'], wallet_addr, netuid, simid, bookId, agentId, "daily_taker_volume"))
                updates.append((agent_gauges, daily_volumes[agentId][bookId]['self'], wallet_addr, netuid, simid, bookId, agentId, "daily_self_volume"))
                updates.append((agent_gauges, daily_roundtrip_volumes[agentId][bookId], wallet_addr, netuid, simid, bookId, agentId, "daily_roundtrip_volume"))
                updates.append((agent_gauges, self.activity_factors.get(agentId, {}).get(bookId, 0.0), wallet_addr, netuid, simid, bookId, agentId, "activity_factor"))
                updates.append((agent_gauges, self.pnl_factors.get(agentId, {}).get(bookId, 1.0), wallet_addr, netuid, simid, bookId, agentId, "pnl_factor"))
                if kappas:
                    if kappas['books'][bookId] is not None:
                        updates.append((agent_gauges, kappas['books'][bookId], wallet_addr, netuid, simid, bookId, agentId, "kappa"))
                    else:
                        try:
                            agent_gauges.remove(wallet_addr, netuid, simid, bookId, agentId, "kappa")
                        except KeyError:
                            pass
                    if 'books_weighted' in kappas and kappas['books_weighted'][bookId] is not None:
                        updates.append((agent_gauges, kappas['books_weighted'][bookId], wallet_addr, netuid, simid, bookId, agentId, "weighted_kappa"))
                    else:
                        try:
                            agent_gauges.remove(wallet_addr, netuid, simid, bookId, agentId, "weighted_kappa")
                        except KeyError:
                            pass
                else:
                    try:
                        agent_gauges.remove(wallet_addr, netuid, simid, bookId, agentId, "kappa")
                    except KeyError:
                        pass       
        bt.logging.debug(f"Agent book metrics collected ({time.time()-start:.4f}s).")

        bt.logging.debug(f"Collecting miner trade metrics...")
        start = time.time()
        for agentId, notices in self.last_state.notices.items():
            if agentId < 0:
                continue
            for notice in notices:
                if notice['y'] in ["EVENT_TRADE", "ET"]:
                    has_new_miner_trades = True
                    break
            if has_new_miner_trades:
                break
        if has_new_miner_trades:
            for uid, book_miner_trades in self.recent_miner_trades.items():
                for bookId, miner_trades in book_miner_trades.items():
                    if len(miner_trades) > 0:
                        last_maker_trade = None
                        last_taker_trade = None
                        for miner_trade, role in self.recent_miner_trades[uid][bookId]:
                            updates.append((self.prometheus_miner_trades, 1.0,
                                wallet_addr, netuid, simid,
                                miner_trade.timestamp, duration_from_timestamp(miner_trade.timestamp),
                                miner_trade.bookId, uid, role,
                                miner_trade.price, miner_trade.quantity,
                                miner_trade.side if role == 'taker' else int(not miner_trade.side),
                                miner_trade.makerFee if role == 'maker' else miner_trade.takerFee,
                                "miner_trades"
                            ))
                            if role == 'maker':
                                last_maker_trade = miner_trade
                            if role == 'taker':
                                last_taker_trade = miner_trade
                        if last_maker_trade:
                            updates.append((agent_gauges, last_maker_trade.makerFeeRate, wallet_addr, netuid, simid, bookId, uid, "fees_last_maker_rate"))
                        if last_taker_trade:
                            updates.append((agent_gauges, last_taker_trade.takerFeeRate, wallet_addr, netuid, simid, bookId, uid, "fees_last_taker_rate"))
        bt.logging.debug(f"Miner trade metrics collected ({time.time()-start:.4f}s).")

        bt.logging.debug(f"Collecting miner metrics...")
        self.prometheus_miners.clear()
        start = time.time()
        for agentId in miner_metrics:
            m = miner_metrics[agentId]

            updates.append((miner_gauges, m['total_base_balance'], wallet_addr, netuid, simid, agentId, "total_base_balance"))
            updates.append((miner_gauges, m['total_base_loan'], wallet_addr, netuid, simid, agentId, "total_base_loan"))
            updates.append((miner_gauges, m['total_base_collateral'], wallet_addr, netuid, simid, agentId, "total_base_collateral"))
            updates.append((miner_gauges, m['total_quote_balance'], wallet_addr, netuid, simid, agentId, "total_quote_balance"))
            updates.append((miner_gauges, m['total_quote_loan'], wallet_addr, netuid, simid, agentId, "total_quote_loan"))
            updates.append((miner_gauges, m['total_quote_collateral'], wallet_addr, netuid, simid, agentId, "total_quote_collateral"))
            updates.append((miner_gauges, m['total_inventory_value'], wallet_addr, netuid, simid, agentId, "total_inventory_value"))
            updates.append((miner_gauges, m['pnl'], wallet_addr, netuid, simid, agentId, "pnl"))
            updates.append((miner_gauges, m['total_realized_pnl'], wallet_addr, netuid, simid, agentId, "total_realized_pnl"))

            updates.append((miner_gauges, m['total_daily_volume']['total'], wallet_addr, netuid, simid, agentId, "total_daily_volume"))
            updates.append((miner_gauges, m['total_daily_volume']['maker'], wallet_addr, netuid, simid, agentId, "total_daily_maker_volume"))
            updates.append((miner_gauges, m['total_daily_volume']['taker'], wallet_addr, netuid, simid, agentId, "total_daily_taker_volume"))
            updates.append((miner_gauges, m['total_daily_volume']['self'], wallet_addr, netuid, simid, agentId, "total_daily_self_volume"))

            updates.append((miner_gauges, m['average_daily_volume']['total'], wallet_addr, netuid, simid, agentId, "average_daily_volume"))
            updates.append((miner_gauges, m['average_daily_volume']['maker'], wallet_addr, netuid, simid, agentId, "average_daily_maker_volume"))
            updates.append((miner_gauges, m['average_daily_volume']['taker'], wallet_addr, netuid, simid, agentId, "average_daily_taker_volume"))
            updates.append((miner_gauges, m['average_daily_volume']['self'], wallet_addr, netuid, simid, agentId, "average_daily_self_volume"))

            updates.append((miner_gauges, m['min_daily_volume']['total'], wallet_addr, netuid, simid, agentId, "min_daily_volume"))
            updates.append((miner_gauges, m['min_daily_volume']['maker'], wallet_addr, netuid, simid, agentId, "min_daily_maker_volume"))
            updates.append((miner_gauges, m['min_daily_volume']['taker'], wallet_addr, netuid, simid, agentId, "min_daily_taker_volume"))
            updates.append((miner_gauges, m['min_daily_volume']['self'], wallet_addr, netuid, simid, agentId, "min_daily_self_volume"))
            
            updates.append((miner_gauges, m['total_roundtrip_volume'], wallet_addr, netuid, simid, agentId, "total_roundtrip_volume"))
            updates.append((miner_gauges, m['average_roundtrip_volume'], wallet_addr, netuid, simid, agentId, "average_roundtrip_volume"))
            updates.append((miner_gauges, m['min_roundtrip_volume'], wallet_addr, netuid, simid, agentId, "min_roundtrip_volume"))

            updates.append((miner_gauges, m['activity_factor'], wallet_addr, netuid, simid, agentId, "activity_factor"))
            updates.append((miner_gauges, m['pnl_factor'], wallet_addr, netuid, simid, agentId, "pnl_factor"))

            if m['kappa'] is not None:
                updates.append((miner_gauges, m['kappa'], wallet_addr, netuid, simid, agentId, "kappa"))
                if m['activity_weighted_normalized_median'] is not None:
                    updates.append((miner_gauges, m['activity_weighted_normalized_median'], wallet_addr, netuid, simid, agentId, "activity_weighted_normalized_median_kappa"))
                if m['kappa_penalty'] is not None:
                    updates.append((miner_gauges, m['kappa_penalty'], wallet_addr, netuid, simid, agentId, "kappa_penalty"))
                if m['kappa_score'] is not None:
                    updates.append((miner_gauges, m['kappa_score'], wallet_addr, netuid, simid, agentId, "kappa_score"))
                if m['pnl_score'] is not None:
                    updates.append((miner_gauges, m['pnl_score'], wallet_addr, netuid, simid, agentId, "pnl_score"))
                else:
                    try:
                        miner_gauges.remove(wallet_addr, netuid, simid, agentId, "pnl_score")
                    except KeyError:
                        pass
                if m['combined_score'] is not None:
                    updates.append((miner_gauges, m['combined_score'], wallet_addr, netuid, simid, agentId, "combined_score"))
                else:
                    try:
                        miner_gauges.remove(wallet_addr, netuid, simid, agentId, "combined_score")
                    except KeyError:
                        pass
            else:
                try:
                    miner_gauges.remove(wallet_addr, netuid, simid, agentId, "kappa")
                except KeyError:
                    pass

            updates.append((miner_gauges, m['unnormalized_score'], wallet_addr, netuid, simid, agentId, "unnormalized_score"))
            updates.append((miner_gauges, m['score'], wallet_addr, netuid, simid, agentId, "score"))
            updates.append((miner_gauges, m['gentrx_score'], wallet_addr, netuid, simid, agentId, "gentrx_score"))
            updates.append((miner_gauges, m['placement'], wallet_addr, netuid, simid, agentId, "placement"))

            updates.append((miner_gauges, (self.metagraph.trust[agentId] if len(self.metagraph.trust) > agentId else 0.0), wallet_addr, netuid, simid, agentId, "trust"))
            updates.append((miner_gauges, (self.metagraph.consensus[agentId] if len(self.metagraph.consensus) > agentId else 0.0), wallet_addr, netuid, simid, agentId, "consensus"))
            updates.append((miner_gauges, (self.metagraph.incentive[agentId] if len(self.metagraph.incentive) > agentId else 0.0), wallet_addr, netuid, simid, agentId, "incentive"))
            updates.append((miner_gauges, (self.metagraph.emission[agentId] if len(self.metagraph.emission) > agentId else 0.0), wallet_addr, netuid, simid, agentId, "emission"))

            _ms = self.miner_stats.get(agentId)
            if _ms and _ms['requests'] >= 100:
                updates.append((miner_gauges, _ms['requests'], wallet_addr, netuid, simid, agentId, "requests"))
                updates.append((miner_gauges, _ms['requests'] - _ms['failures'] - _ms['timeouts'] - _ms['rejections'], wallet_addr, netuid, simid, agentId, "success"))
                updates.append((miner_gauges, _ms['failures'], wallet_addr, netuid, simid, agentId, "failures"))
                updates.append((miner_gauges, _ms['timeouts'], wallet_addr, netuid, simid, agentId, "timeouts"))
                updates.append((miner_gauges, _ms['rejections'], wallet_addr, netuid, simid, agentId, "rejections"))
                updates.append((miner_gauges, (sum(_ms['call_time']) / len(_ms['call_time']) if _ms['call_time'] else 0), wallet_addr, netuid, simid, agentId, "call_time"))
                self.miner_stats[agentId] = {'requests': 0, 'timeouts': 0, 'failures': 0, 'rejections': 0, 'call_time': []}

            _set_if_changed_metric(
                self.prometheus_miners,
                1.0,
                wallet=wallet_addr,
                netuid=netuid,
                sim_id=simid,
                agent_id=agentId,
                timestamp=self.simulation_timestamp,
                timestamp_str=duration_from_timestamp(self.simulation_timestamp),
                placement=m['placement'],
                base_balance=m['total_base_balance'],
                base_loan=m['total_base_loan'],
                base_collateral=m['total_base_collateral'],
                quote_balance=m['total_quote_balance'],
                quote_loan=m['total_quote_loan'],
                quote_collateral=m['total_quote_collateral'],
                inventory_value=m['total_inventory_value'],
                inventory_value_change=m['inventory_value_change'],
                pnl=m['pnl'],
                pnl_change=m['pnl_change'],
                total_realized_pnl=m['total_realized_pnl'],
                total_daily_volume=m['total_daily_volume']['total'],
                min_daily_volume=m['min_daily_volume']['total'],
                average_daily_volume=m['average_daily_volume']['total'],
                total_roundtrip_volume=m['total_roundtrip_volume'],
                min_roundtrip_volume=m['min_roundtrip_volume'], 
                average_roundtrip_volume=m['average_roundtrip_volume'],
                activity_factor=m['activity_factor'],
                pnl_factor=m['pnl_factor'],
                kappa=m['kappa'],
                kappa_penalty=m['kappa_penalty'],
                kappa_score=m['kappa_score'],
                pnl_score=m['pnl_score'],
                combined_score=m['combined_score'],
                unnormalized_score=m['unnormalized_score'],
                score=m['score'],
                miner_gauge_name='miners'
            )
        bt.logging.debug(f"Miner metrics collected ({time.time()-start:.4f}s).")
        
        bt.logging.info(f"Applying {len(updates)} metric updates...")
        apply_start = time.time()
        GAUGES_TO_CLEAR = {self.prometheus_trades, self.prometheus_books, self.prometheus_miner_trades}
        cleared_metrics = set()
        for update in updates:
            gauge = update[0]            
            if gauge in GAUGES_TO_CLEAR and gauge not in cleared_metrics:
                gauge.clear()
                cleared_metrics.add(gauge)            
            _set_if_changed(*update)
        bt.logging.info(f"Applied {len(updates)} updates in {time.time()-apply_start:.4f}s")
        
        bt.logging.info(f"Metrics Published for Step {report_step} ({time.time()-report_start}s).")
    except Exception as ex:
        self.pagerduty_alert(f"Unable to publish metrics : {ex}", details={"traceback": traceback.format_exc()})
    finally:
        self.shared_state_reporting = False
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.logging.set_info()
    
    parser.add_argument('--netuid', type=int, default=1)
    parser.add_argument('--logging.level', type=str, default="info")
    parser.add_argument('--prometheus.port', type=int, default=9001)
    parser.add_argument('--prometheus.level', type=str, default='INFO')
    parser.add_argument('--cpu-cores', type=str, default=None)
    
    config = bt.Config(parser)
    bt.logging(config=config)
    
    if config.cpu_cores:
        cores = [int(c) for c in config.cpu_cores.split(',')]
        os.sched_setaffinity(0, set(cores))
        bt.logging.info(f"Reporting service assigned to cores: {cores}")
    
    service = ReportingService(config)
    
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        bt.logging.info("Reporting service stopped by user")
    except Exception as e:
        bt.logging.error(f"Reporting service crashed: {e}")
        bt.logging.error(traceback.format_exc())
        sys.exit(1)