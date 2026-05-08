# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

if __name__ != "__mp_main__":
    import os
    import json
    import signal
    import sys
    import platform
    import time
    import argparse
    import torch
    import traceback
    import xml.etree.ElementTree as ET
    import msgspec
    import math
    import shutil
    import zipfile
    import asyncio
    import posix_ipc
    import mmap
    import msgpack
    import atexit
    import subprocess
    import struct
    import heapq
    import aiofiles
    import aiofiles.os
    import select
    import gc
    import copy
    from datetime import datetime, timedelta
    from ypyjson import YpyObject

    import bittensor as bt

    from GenTRX.src.bt_log import gtx_log

    import uvicorn
    from typing import Tuple, Dict, List
    from fastapi import FastAPI, APIRouter
    from fastapi import Request
    import threading
    from threading import Thread, Lock, Event
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    from collections import deque, defaultdict
    import numpy as np

    import psutil
    from git import Repo
    from pathlib import Path

    from taos import __spec_version__
    from taos.common.neurons.validator import BaseValidatorNeuron
    from taos.im.utils import duration_from_timestamp
    from taos.im.utils.save import save_state_worker
    from taos.im.utils.reward import get_inventory_value
    from taos.im.utils.affinity import get_core_allocation

    from taos.im.config import add_im_validator_args
    from taos.im.protocol.simulator import SimulatorResponseBatch
    from taos.im.protocol import MarketSimulationStateUpdate, FinanceEventNotification, FinanceAgentResponse
    from taos.im.protocol.models import MarketSimulationConfig, TradeInfo
    from taos.im.protocol.events import SimulationStartEvent, TradeEvent

    class Validator(BaseValidatorNeuron):
        """
        Intelligent market simulation validator implementation.

        The validator is run as a FastAPI client in order to receive messages from the simulator engine for processing and forwarding to miners.
        Metagraph maintenance, weight setting, state persistence and other general bittensor routines are executed in a separate thread.
        The validator also handles publishing of metrics via Prometheus for visualization and analysis, as well as retrieval and recording of seed data for simulation price process generation.
        """

        @classmethod
        def add_args(cls, parser: argparse.ArgumentParser) -> None:
            """
            Registers Intelligent-Markets-specific CLI configuration parameters.

            Args:
                parser (argparse.ArgumentParser): The main argument parser to extend.

            Returns:
                None
            """
            add_im_validator_args(cls, parser)

        def _setup_signal_handlers(self):
            """
            Registers OS signal handlers for graceful shutdown.

            Behavior:
                - Captures SIGINT, SIGTERM, and SIGHUP (if available).
                - Logs the received signal.
                - Triggers full validator cleanup.
                - Exits the process cleanly.

            Returns:
                None
            """
            def signal_handler(signum, frame):
                signal_name = signal.Signals(signum).name
                bt.logging.info(f"Received {signal_name}, initiating graceful shutdown...")
                self.cleanup()
                sys.exit(0)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, signal_handler)

        def _start_query_service(self):
            """
            Launches the validator's query service and initializes POSIX IPC resources.

            Responsibilities:
                - Spawns the query subprocess with correct wallet and network parameters.
                - Waits for creation of shared memory blocks and message queues.
                - Initializes memory maps for request/response communication.
                - Verifies that the query service is alive during startup.
                - Raises a RuntimeError if IPC initialization fails or service dies.

            Returns:
                None

            Raises:
                RuntimeError: If IPC endpoints are not ready within timeout
                            or if the subprocess exits unexpectedly.
            """

            bt.logging.info(f"Starting query service from: ../validator/query.py")

            core_allocation = get_core_allocation(grad_server_cores=int(os.environ.get("GRAD_CORES_COUNT", "0")))
            cmd = [
                sys.executable,
                '-u',
                '../validator/query.py',
                '--logging.trace' if self.config.logging.trace else '--logging.debug' if self.config.logging.debug else '--logging.info',
                '--wallet.path', self.config.wallet.path,
                '--wallet.name', self.config.wallet.name,
                '--wallet.hotkey', self.config.wallet.hotkey,
                '--subtensor.network', self.config.subtensor.network,
                '--netuid', str(self.config.netuid),
                '--neuron.timeout', str(self.config.neuron.timeout),
                '--neuron.global_query_timeout', str(self.config.neuron.global_query_timeout),
                '--compression.level', str(self.config.compression.level),
                '--compression.engine', self.config.compression.engine,
                '--compression.parallel_workers', str(self.config.compression.parallel_workers),
                '--cpu-cores', ','.join(map(str, core_allocation['query'])),
                '--notify-fd', str(self.query_notify_write)
            ]
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            self.query_process = subprocess.Popen(
                cmd,
                stderr=sys.stderr,
                env=env,
                pass_fds=(self.query_notify_write,)
            )
            bt.logging.info(f"Query service PID: {self.query_process.pid}")
            os.close(self.query_notify_write)

            # Wait for IPC resources to be created with retry
            queue_name = f"/validator_query_{self.config.wallet.hotkey}"

            bt.logging.info("Waiting for query service IPC resources...")
            max_retries = 30  # 30 seconds max
            for attempt in range(max_retries):
                try:
                    self.request_queue = posix_ipc.MessageQueue(f"{queue_name}_req")
                    self.response_queue = posix_ipc.MessageQueue(f"{queue_name}_res")
                    self.request_shm = posix_ipc.SharedMemory(f"{queue_name}_req_shm")
                    self.response_shm = posix_ipc.SharedMemory(f"{queue_name}_res_shm")

                    self.request_mem = mmap.mmap(self.request_shm.fd, self.request_shm.size)
                    self.response_mem = mmap.mmap(self.response_shm.fd, self.response_shm.size)

                    bt.logging.info(f"Query service ready (request_shm: {self.request_shm.size / 1024 / 1024:.0f}MB, response_shm: {self.response_shm.size / 1024 / 1024:.0f}MB)")
                    break
                except posix_ipc.ExistentialError:
                    if attempt == 0:
                        bt.logging.debug("IPC resources not ready yet, waiting...")
                    time.sleep(1)
                    # Check if process died
                    if self.query_process.poll() is not None:
                        raise RuntimeError(f"Query service died with exit code {self.query_process.returncode}")
            else:
                raise RuntimeError("Timeout waiting for query service IPC resources")

            # Wait for pipe ready signal
            bt.logging.info("Waiting for query service ready signal...")
            for attempt in range(100):
                if self.query_process.poll() is not None:
                    raise RuntimeError(f"Query service died with exit code {self.query_process.returncode}")
                readable, _, _ = select.select([self.query_notify_read], [], [], 0.1)
                if readable:
                    ready_signal = os.read(self.query_notify_read, 1)
                    if ready_signal == b'R':
                        bt.logging.success("Query service ready!")
                        return
                    elif ready_signal != b'':
                        bt.logging.warning(f"Unexpected ready signal: {ready_signal}")
                        return

            raise RuntimeError("Query service did not send ready signal within 10s")

        def _start_reporting_service(self):
            bt.logging.info(f"Starting reporting service from: ../validator/report.py")

            self._reporting = False
            core_allocation = get_core_allocation(grad_server_cores=int(os.environ.get("GRAD_CORES_COUNT", "0")))
            cmd = [
                sys.executable,
                '-u',
                '../validator/report.py',
                '--logging.trace' if self.config.logging.trace else '--logging.debug' if self.config.logging.debug else '--logging.info',
                '--wallet.path', self.config.wallet.path,
                '--wallet.name', self.config.wallet.name,
                '--wallet.hotkey', self.config.wallet.hotkey,
                '--subtensor.network', self.config.subtensor.network,
                '--netuid', str(self.config.netuid),
                '--prometheus.port', str(self.config.prometheus.port),
                '--prometheus.level', str(self.config.prometheus.level),
                '--cpu-cores', ','.join(map(str, core_allocation['reporting'])),
            ]
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            self.reporting_process = subprocess.Popen(cmd, stderr=sys.stderr, env=env)
            bt.logging.info(f"Reporting service PID: {self.reporting_process.pid}")

            bt.logging.info("Waiting for reporting service IPC resources...")
            max_retries = 30
            for attempt in range(max_retries):
                try:
                    self.reporting_request_queue = posix_ipc.MessageQueue("/validator-report-req")
                    self.reporting_response_queue = posix_ipc.MessageQueue("/validator-report-res")
                    self.reporting_request_shm = posix_ipc.SharedMemory("/validator-report-data")
                    self.reporting_response_shm = posix_ipc.SharedMemory("/validator-report-response-data")

                    self.reporting_request_mem = mmap.mmap(self.reporting_request_shm.fd, self.reporting_request_shm.size)
                    self.reporting_response_mem = mmap.mmap(self.reporting_response_shm.fd, self.reporting_response_shm.size)

                    bt.logging.info(f"Reporting service ready (shm: {self.reporting_request_shm.size / 1024 / 1024:.0f}MB)")
                    return

                except posix_ipc.ExistentialError:
                    if attempt == 0:
                        bt.logging.debug("IPC resources not ready yet, waiting...")
                    time.sleep(1)

                    if self.reporting_process.poll() is not None:
                        raise RuntimeError(f"Reporting service died with exit code {self.reporting_process.returncode}")

            raise RuntimeError("Timeout waiting for reporting service IPC resources")

        def _start_seed_service(self):
            """
            Launches the seed service subprocess.
            """
            bt.logging.info("Starting seed service from: ../validator/seed.py")

            cmd = [
                sys.executable,
                '-u',
                '../validator/seed.py',
                '--seed.fundamental.symbol.coinbase', self.config.simulation.seeding.fundamental.symbol.coinbase,
                '--seed.fundamental.symbol.binance', self.config.simulation.seeding.fundamental.symbol.binance,
                '--seed.external.symbol.coinbase', self.config.simulation.seeding.external.symbol.coinbase,
                '--seed.external.symbol.binance', self.config.simulation.seeding.external.symbol.binance,
                '--seed.external.sampling_seconds', str(self.config.simulation.seeding.external.sampling_seconds),
                '--logging.level', 'INFO' if not (self.config.logging.debug or self.config.logging.trace) else 'DEBUG',
            ]

            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            self.seed_process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env
            )
            bt.logging.info(f"Seed service PID: {self.seed_process.pid}")
            time.sleep(2.0)
            if self.seed_process.poll() is not None:
                raise RuntimeError(f"Seed service died with exit code {self.seed_process.returncode}")
            if hasattr(self, 'simulation') and self.simulation.logDir:
                self._notify_seed_log_dir_change(self.simulation.logDir)

        def _notify_seed_log_dir_change(self, new_log_dir: str) -> None:
            """
            Notify seed service of log directory change via file + signal.

            Args:
                new_log_dir: New log directory path
            """
            try:
                log_dir_file = '/tmp/validator_log_dir.txt'
                with open(log_dir_file, 'w') as f:
                    f.write(new_log_dir)
                if not hasattr(self, 'seed_process') or not self.seed_process:
                    return
                if self.seed_process.poll() is not None:
                    bt.logging.warning(
                        f"Seed service died (exit code {self.seed_process.returncode}), "
                        "cannot notify of log dir change"
                    )
                    return
                self.seed_process.send_signal(signal.SIGUSR1)
                bt.logging.info(f"Notified seed service of log directory change: {new_log_dir}")

            except ProcessLookupError:
                bt.logging.warning("Seed process not found when trying to notify log dir change")
            except Exception as ex:
                bt.logging.warning(f"Failed to notify seed service of log dir change: {ex}")

        def monitor(self) -> None:
            """
            Periodically checks simulator health and restarts if needed with timeout protection.

            Runs in a blocking loop:
                - Sleeps 5 minutes between checks.
                - Logs simulator availability.
                - Handles and logs unexpected exceptions with timeout protection.

            Returns:
                None
            """
            last_restart_time = 0
            restart_cooldown = 60.0
            check_timeout = 30.0
            restart_timeout = 120.0

            while True:
                try:
                    time.sleep(300)
                    bt.logging.info(f"Checking simulator state...")
                    check_start = time.time()
                    try:
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(check_simulator, self)
                            try:
                                is_healthy = future.result(timeout=check_timeout)
                            except concurrent.futures.TimeoutError:
                                bt.logging.error(f"Simulator health check timed out after {check_timeout}s")
                                is_healthy = False
                        check_elapsed = time.time() - check_start
                        if not is_healthy:
                            time_since_restart = time.time() - last_restart_time
                            if time_since_restart < restart_cooldown:
                                bt.logging.warning(
                                    f"Simulator unhealthy but restart on cooldown "
                                    f"({time_since_restart:.1f}s < {restart_cooldown}s)"
                                )
                                continue
                            bt.logging.warning(f"Simulator unhealthy (check took {check_elapsed:.1f}s), restarting...")
                            restart_start = time.time()
                            try:
                                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                    restart_future = executor.submit(restart_simulator, self)
                                    try:
                                        restart_future.result(timeout=restart_timeout)
                                        last_restart_time = time.time()
                                        restart_elapsed = time.time() - restart_start
                                        bt.logging.info(f"Simulator restarted successfully ({restart_elapsed:.1f}s)")
                                    except concurrent.futures.TimeoutError:
                                        bt.logging.error(f"Simulator restart timed out after {restart_timeout}s")
                                        self.pagerduty_alert(
                                            f"Simulator restart timeout after {restart_timeout}s"
                                        )
                                        last_restart_time = time.time()
                            except Exception as restart_ex:
                                bt.logging.error(f"Simulator restart failed: {restart_ex}")
                                bt.logging.error(traceback.format_exc())
                                self.pagerduty_alert(
                                    f"Simulator restart failed: {restart_ex}",
                                    details={"trace": traceback.format_exc()}
                                )
                                last_restart_time = time.time()
                        else:
                            bt.logging.info(f"Simulator online! (check took {check_elapsed:.1f}s)")

                    except Exception as check_ex:
                        bt.logging.error(f"Simulator health check failed: {check_ex}")
                        bt.logging.error(traceback.format_exc())

                except Exception as ex:
                    bt.logging.error(f"Failure in simulator monitor : {traceback.format_exc()}")
                    time.sleep(10)

        def update_repo(self, end=False) -> bool:
            """
            Checks for source or config changes in the repository and reloads components.

            Behavior:
                - Pulls latest remote changes.
                - Rebuilds simulator when C++ or Python sources change.
                - Restarts simulator on configuration changes.
                - Updates validator process when its own Python source changes.
                - Handles special behavior on simulation end.

            Args:
                end (bool): Whether the update is performed during simulation shutdown.

            Returns:
                bool: True if update steps completed successfully, False on error.
            """
            try:
                validator_py_files_changed, simulator_config_changed, simulator_py_files_changed, simulator_cpp_files_changed = check_repo(self)
                remote = self.repo.remotes[self.config.repo.remote]

                if not end:
                    if validator_py_files_changed and not (simulator_cpp_files_changed or simulator_py_files_changed):
                        bt.logging.warning("VALIDATOR LOGIC UPDATED - PULLING AND DEPLOYING.")
                        remote.pull()
                        update_validator(self)
                else:
                    try:
                        remote.pull()
                    except Exception as ex:
                        self.pagerduty_alert(f"Failed to pull changes from repo on simulation end : {ex}")
                    if simulator_cpp_files_changed or simulator_py_files_changed:
                        bt.logging.warning("SIMULATOR SOURCE CHANGED")
                        rebuild_simulator(self)
                    if simulator_config_changed:
                        bt.logging.warning("SIMULATOR CONFIG CHANGED")
                    restart_simulator(self, end)
                    if validator_py_files_changed:
                        update_validator(self)
                return True
            except Exception as ex:
                self.pagerduty_alert(f"Failed to update repo : {ex}", details={"traceback" : traceback.format_exc()})
                return False

        def _compress_outputs(self,  start=False):
            """
            Compresses old simulator log outputs and performs disk cleanup.

            Responsibilities:
                - Groups historical .log files into ZIP archives.
                - Removes original log files once archived.
                - Enforces storage retention when disk usage exceeds 85%.
                - Deletes dated archives and directories as needed.
                - Handles exceptions gracefully with PagerDuty reporting.

            Args:
                start (bool): If True, performs cleanup of prior simulation logs
                    even if timestamps overlap with the new run.

            Returns:
                None
            """
            self.compressing = True
            try:
                if self.simulation.logDir:
                    log_root = Path(self.simulation.logDir).parent
                    for output_dir in log_root.iterdir():
                        if output_dir.is_dir():
                            log_archives = {}
                            log_path = Path(output_dir)
                            for log_file in log_path.iterdir():
                                if log_file.is_file() and log_file.suffix == '.log':
                                    _parts = log_file.name.split('.')
                                    if len(_parts) < 2:
                                        continue
                                    log_period = _parts[1]
                                    if '-' not in log_period:
                                        continue
                                    if len(log_period) == 13:
                                        log_end = (int(log_period.split('-')[1][:2]) * 3600 + int(log_period.split('-')[1][2:4]) * 60 + int(log_period.split('-')[1][4:])) * 1_000_000_000
                                    else:
                                        log_end = (int(log_period.split('-')[1][:2]) * 86400 + int(log_period.split('-')[1][2:4]) * 3600 + int(log_period.split('-')[1][4:6]) * 60 + int(log_period.split('-')[1][6:])) * 1_000_000_000
                                    if log_end < self.simulation_timestamp or (start and str(output_dir.resolve()) != self.simulation.logDir):
                                        log_type = log_file.name.split('-')[0]
                                        label = f"{log_type}_{log_period}"
                                        if not label in log_archives:
                                            log_archives[label] = []
                                        log_archives[label].append(log_file)
                            for label, log_files in log_archives.items():
                                archive = log_path / f"{label}.zip"
                                bt.logging.info(f"Compressing {label} files to {archive.name}...")
                                with zipfile.ZipFile(archive, "w" if not archive.exists() else "a", compression=zipfile.ZIP_DEFLATED) as zipf:
                                    for log_file in log_files:
                                        try:
                                            zipf.write(log_file, arcname=Path(log_file).name)
                                            os.remove(log_file)
                                            bt.logging.debug(f"Added {log_file.name} to {archive.name}")
                                        except Exception as ex:
                                            bt.logging.error(f"Failed to add {log_file.name} to {archive.name} : {ex}")
                    if psutil.disk_usage('/').percent > 85:
                        min_retention_date = int((datetime.today() - timedelta(days=7)).strftime("%Y%m%d"))
                        bt.logging.warning(f"Disk usage > 85% - cleaning up old outputs...")
                        for output in sorted(log_root.iterdir(), key=lambda f: f.name[:13]):
                            try:
                                archive_date = int(output.name[:8])
                            except:
                                continue
                            if archive_date < min_retention_date:
                                try:
                                    if output.is_file() and output.name.endswith('.zip'):
                                        output.unlink()
                                    elif output.is_dir():
                                        shutil.rmtree(output)
                                    disk_usage = psutil.disk_usage('/').percent
                                    bt.logging.success(f"Deleted {output.name} ({disk_usage}% disk available).")
                                    if disk_usage <= 85:
                                        break
                                except Exception as ex:
                                    self.pagerduty_alert(f"Failed to remove output {output.name} : {ex}", details={"trace" : traceback.format_exc()})


            except Exception as ex:
                self.pagerduty_alert(f"Failure during output compression : {ex}", details={"trace" : traceback.format_exc()})
            finally:
                self.compressing = False

        def compress_outputs(self, start=False):
            """
            Launches asynchronous log compression in a background thread.

            Behavior:
                - Ensures only one compression job runs at a time.
                - Spawns a daemon thread to execute `_compress_outputs()`.

            Args:
                start (bool): If True, forces compression of pre-run logs.

            Returns:
                None
            """
            if not self.compressing:
                Thread(target=self._compress_outputs, args=(start,), daemon=True, name=f'compress_{self.step}').start()

        def load_simulation_config(self) -> None:
            """
            Loads the market-simulation configuration from its XML definition.

            Responsibilities:
                - Parses the XML file into a MarketSimulationConfig object.
                - Initializes paths for validator and simulation state files.
                - Loads the previous saved state (if any).

            Returns:
                None
            """
            self.xml_config = ET.parse(self.config.simulation.xml_config).getroot()
            self.simulation = MarketSimulationConfig.from_xml(self.xml_config)
            self.validator_state_file = self.config.neuron.full_path + f"/validator.mp"
            self.simulation_state_file = self.config.neuron.full_path + f"/{self.simulation.label()}.mp"
            self._initialize_all_structures()
            self.load_state()

        def __init__(self, config=None) -> None:
            """
            Initializes the Intelligent Markets validator node.

            Responsibilities:
                - Loads simulation configuration from XML.
                - Initializes metagraph and subnet info.
                - Sets up executors, event loops, state locks, and signal handlers.
                - Loads prior simulation/validator state if available.
                - Initializes metrics, reporting, and query service.
                - Starts IPC-backed query subprocess.

            Args:
                config: Validator configuration object.

            Raises:
                Exception: If the simulation config XML file is missing.
            """
            super(Validator, self).__init__(config=config)

            kappa_w = self.config.scoring.kappa.weight
            pnl_w = self.config.scoring.pnl.weight
            gentrx_sim_share = getattr(getattr(self.config.scoring, 'gentrx', None), 'simulation_share', 0.0) or 0.0
            burn_ratio = getattr(self.config.neuron, 'burn_ratio', 0.0) or 0.0
            tolerance = 1e-6

            trading_sum = kappa_w + pnl_w
            if abs(trading_sum - 1.0) > tolerance:
                error_msg = (
                    f"Trading-pool weights must sum to 1.0, got "
                    f"kappa={kappa_w:.6f} + pnl={pnl_w:.6f} = {trading_sum:.6f}."
                )
                bt.logging.error(error_msg)
                raise ValueError(error_msg)

            if not (0.0 <= gentrx_sim_share <= 1.0):
                error_msg = (
                    f"--scoring.gentrx.simulation_share must be in [0, 1], "
                    f"got {gentrx_sim_share:.6f}."
                )
                bt.logging.error(error_msg)
                raise ValueError(error_msg)

            bt.logging.info(
                f"Scoring weights validated: trading=(kappa={kappa_w:.4f}, pnl={pnl_w:.4f}), "
                f"gentrx.simulation_share={gentrx_sim_share:.4f}, burn_ratio={burn_ratio:.4f}"
            )

            # Load the simulator config XML file data in order to make context and parameters accessible for reporting and output location.
            if not os.path.exists(self.config.simulation.xml_config):
                raise Exception(f"Simulator config does not exist at {self.config.simulation.xml_config}!")
            self.simulator_config_file = os.path.realpath(Path(self.config.simulation.xml_config))
            # Initialize subnet info and other basic validator/simulation properties
            self.subnet_info = self.subtensor.get_metagraph_info(self.config.netuid)
            
            self.benchmark_agents = []
            self.benchmark_start_uid = self.subnet_info.max_uids
            if self.config.benchmark.agents:
                self._load_benchmark_agents()
            self.effective_max_uids = self.subnet_info.max_uids + len(self.benchmark_agents)
            self.scores = torch.zeros(
                self.effective_max_uids, dtype=torch.float32, device=self.device
            )
            self.gentrx_scores = torch.zeros(
                self.effective_max_uids, dtype=torch.float32, device=self.device
            )
            
            self.last_state = None
            self.last_response = None
            self.msgpack_error_counter = 0
            self.simulation_timestamp = 0
            self.start_time = None
            self.start_timestamp = None
            self.last_state_time = None
            self.step_rates = []
            self._last_defrag_hour = -1
            self._last_prune_timestamp = None

            self.main_loop = asyncio.new_event_loop()
            self._main_loop_ready = Event()
            core_allocation = get_core_allocation(grad_server_cores=int(os.environ.get("GRAD_CORES_COUNT", "0")))
            ipc_cores = core_allocation['ipc']
            if len(ipc_cores) >= 2:
                mid = len(ipc_cores) // 2
                query_cores = ipc_cores[:mid]
                reporting_cores = ipc_cores[mid:]
            else:
                query_cores = ipc_cores
                reporting_cores = ipc_cores
            self.query_ipc_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix='query_ipc',
                initializer=lambda: os.sched_setaffinity(0, set(query_cores))
            )
            self.reporting_ipc_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix='reporting_ipc',
                initializer=lambda: os.sched_setaffinity(0, set(reporting_cores))
            )
            bt.logging.info(f"Query IPC executor assigned to cores: {query_cores}")
            bt.logging.info(f"Reporting IPC executor assigned to cores: {reporting_cores}")
            validator_cores = core_allocation['validator']
            os.sched_setaffinity(0, set(validator_cores))
            bt.logging.info(f"Validator assigned to cores: {validator_cores}")
            self.reward_cores = core_allocation['reward']
            self.reward_executor = ThreadPoolExecutor(max_workers=len(self.reward_cores),initializer=lambda: os.sched_setaffinity(0, set(self.reward_cores)))
            bt.logging.info(f"Reward executor assigned to cores: {self.reward_cores}")
            self.save_state_executor = ThreadPoolExecutor(max_workers=1)
            self.maintenance_executor = ThreadPoolExecutor(max_workers=1)

            self.maintaining = False
            self.compressing = False
            self.querying = False
            self._receiving_state = False
            self._rewarding = False
            self._pending_reward_tasks = 0
            self._saving = False
            self._reporting = False
            self._rewarding_lock = Lock()
            self._saving_lock = Lock()
            self._reporting_lock = Lock()
            self._setup_signal_handlers()
            self._cleanup_done = False
            atexit.register(self.cleanup)

            self.initial_balances_published = {uid : False for uid in range(self.effective_max_uids)}
            self.volume_sums = defaultdict(lambda: defaultdict(float))
            self.maker_volume_sums = defaultdict(lambda: defaultdict(float))
            self.taker_volume_sums = defaultdict(lambda: defaultdict(float))
            self.self_volume_sums = defaultdict(lambda: defaultdict(float))
            self.open_positions = defaultdict(lambda: defaultdict(lambda: {
                'longs': deque(),
                'shorts': deque()
            }))
            self.inventory_history = {uid : {} for uid in range(self.effective_max_uids)}
            self.realized_pnl_history = defaultdict(lambda: defaultdict(dict))
            self.roundtrip_volumes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            self.roundtrip_volume_sums = defaultdict(lambda: defaultdict(float))
            self.kappa_cache = {}
            self.pending_notices = {uid: [] for uid in range(self.effective_max_uids)}

            self.load_simulation_config()

            self.router = APIRouter()
            self.router.add_api_route("/orderbook", self.orderbook, methods=["GET"])
            self.router.add_api_route("/account", self.account, methods=["GET"])

            # ---- GenTRX distributed training (optional) ----
            # Log via the `gtx_log` shim (imported at module top) — forwards
            # to bt.logging with the `[GTX]` prefix baked in, so every record
            # surfaces through the same stream the rest of the validator uses.
            self._gentrx = None
            _gentrx_enabled = getattr(getattr(self.config, "gentrx", None), "enabled", False)
            gtx_log.info(f"init: enabled={_gentrx_enabled}")
            if _gentrx_enabled:
                try:
                    from GenTRX.src.service import GenTRXService
                    # Miner UIDs are computed dynamically — metagraph changes
                    # over time as miners register/dereg, and miners may not be
                    # registered yet at validator init.
                    my_hotkey = self.wallet.hotkey.ss58_address

                    def _current_miner_uids():
                        uids = [
                            uid for uid in range(self.metagraph.n)
                            if self.metagraph.axons[uid].hotkey != my_hotkey
                            and self.metagraph.axons[uid].ip != "0.0.0.0"
                            and self.metagraph.axons[uid].port != 0
                        ]
                        # Include benchmark agents that have active axons (scored,
                        # not rewarded — same pattern as trading benchmarks).
                        for i, agent in enumerate(self.benchmark_agents):
                            axon = agent['axon']
                            if axon.ip != "0.0.0.0" and axon.port != 0:
                                uids.append(self.benchmark_start_uid + i)
                        return uids

                    gtx_log.info(f"init: server_url={self.config.gentrx.gradient_server_url}, current miner_uids={_current_miner_uids()}")

                    def _get_current_block():
                        try:
                            return self.subtensor.get_current_block()
                        except Exception:
                            return 0

                    self._gentrx = GenTRXService.from_config(
                        self.config,
                        deliver_fn=self._deliver_gentrx_assignments,
                        miner_uids_fn=_current_miner_uids,
                        get_block_fn=_get_current_block,
                        validator_uid=self.uid,
                    )
                    if self._gentrx:
                        if self._gentrx.is_healthy():
                            gtx_log.info(f"init: gradient server healthy at {self.config.gentrx.gradient_server_url}")
                        else:
                            gtx_log.warning(f"init: gradient server unreachable at {self.config.gentrx.gradient_server_url} — will retry on each tick")
                        # Commit validator bucket read credentials to chain.
                        # Miners and the aggregator use this to discover the bucket
                        # (data, scores, checkpoints) without pre-configured env vars.
                        try:
                            from GenTRX.src.chain import BucketInfo, GenTRXChain
                            val_bucket = BucketInfo.from_validator_env()
                            if val_bucket:
                                gtx_chain = GenTRXChain(self.subtensor, self.config.netuid, self.metagraph)
                                gtx_chain.commit_bucket(self.wallet, val_bucket)
                                gtx_log.info("validator bucket committed to chain")
                            else:
                                gtx_log.debug("GENTRX_VALIDATOR_S3_* not set — skipping chain commitment")
                        except Exception as exc:
                            gtx_log.warning(f"validator chain commit failed: {exc}")
                        # Register benchmark miner buckets with the gradient server.
                        # These miners can't commit to chain (not registered), so the
                        # validator injects their bucket info via the REST API.
                        for bm in self.benchmark_agents:
                            bkt = bm.get('gentrx_bucket')
                            if bkt:
                                ok = self._gentrx.register_benchmark_bucket(bm['uid'], bkt)
                                if ok:
                                    gtx_log.info(f"registered benchmark bucket: uid={bm['uid']} bucket={bkt.get('bucket_name')}")
                                else:
                                    gtx_log.warning(f"failed to register benchmark bucket: uid={bm['uid']}")
                    else:
                        gtx_log.warning("init: from_config returned None — check that --gentrx.gradient_server_url is set")
                except ImportError as exc:
                    gtx_log.warning(f"init: import failed — {exc}")
                except Exception as exc:
                    gtx_log.error(f"init: failed — {exc}")
                    import traceback
                    gtx_log.error(traceback.format_exc())
            gtx_log.info(f"init: {'ACTIVE' if self._gentrx else 'DISABLED'}")

            self.repo_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
            self.repo = Repo(self.repo_path)
            self.update_repo()

            self.miner_stats = {uid : {'requests' : 0, 'timeouts' : 0, 'failures' : 0, 'rejections' : 0, 'call_time' : []} for uid in range(self.effective_max_uids)}
            self.query_process = None
            self.query_notify_read, self.query_notify_write = os.pipe()
            import fcntl
            flags = fcntl.fcntl(self.query_notify_read, fcntl.F_GETFL)
            fcntl.fcntl(self.query_notify_read, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            bt.logging.info(f"Created notification pipe: read_fd={self.query_notify_read}, write_fd={self.query_notify_write}")
            self._start_query_service()
            self.report_process = None
            self._start_reporting_service()
            self.seed_process = None
            self._start_seed_service()
            
        def _load_benchmark_agents(self):
            """Load benchmark agent configurations"""
            try:
                with open(self.config.benchmark.agents, 'r') as f:
                    benchmark_config = json.load(f)                
                for idx, agent in enumerate(benchmark_config['agents']):
                    uid = self.benchmark_start_uid + idx
                    entry = {
                        'uid': uid,
                        'name': agent['name'],
                        'ip': agent['ip'],
                        'port': agent['port'],
                        'hotkey': agent['hotkey'],
                        'axon': bt.AxonInfo(
                            ip=agent['ip'],
                            port=agent['port'],
                            hotkey=agent['hotkey'],
                            coldkey='benchmark',
                            version=self.metagraph.axons[0].version,
                            ip_type=self.metagraph.axons[0].ip_type,
                            protocol=self.metagraph.axons[0].protocol,
                        )
                    }
                    if 'gentrx_bucket' in agent:
                        entry['gentrx_bucket'] = agent['gentrx_bucket']
                    self.benchmark_agents.append(entry)
                    bt.logging.info(f"Loaded benchmark agent: {agent['name']} (UID {uid}) at {agent['ip']}:{agent['port']}")            
            except Exception as ex:
                bt.logging.error(f"Failed to load benchmark agents: {ex}")
                bt.logging.error(traceback.format_exc())
                self.benchmark_agents = []
                
        def _initialize_all_structures(self):
            """
            Initialize all UID-indexed structures to effective_max_uids.
            """
            bt.logging.info(
                f"Initializing structures for {self.effective_max_uids} UIDs "
                f"(network: {self.subnet_info.max_uids}, benchmarks: {len(self.benchmark_agents)}) "
                f"with {self.simulation.book_count} books"
            )
            book_count = self.simulation.book_count
            if not hasattr(self, 'activity_factors') or len(self.activity_factors) < self.effective_max_uids:
                self.activity_factors = {
                    uid: {bookId: 0.0 for bookId in range(book_count)}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'pnl_factors') or len(self.pnl_factors) < self.effective_max_uids:
                self.pnl_factors = {
                    uid: {bookId: 1.0 for bookId in range(book_count)}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'kappa_values') or len(self.kappa_values) < self.effective_max_uids:
                self.kappa_values = {
                    uid: {
                        'books': {bookId: None for bookId in range(book_count)},
                        'books_weighted': {bookId: 0.0 for bookId in range(book_count)},
                        'total': None,
                        'average': None,
                        'median': None,
                        'normalized_average': 0.0,
                        'normalized_median': 0.0,
                        'normalized_total': 0.0,
                        'activity_weighted_normalized_median': 0.0,
                        'penalty': 0.0,
                        'score': 0.0,
                    }
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'unnormalized_scores') or len(self.unnormalized_scores) < self.effective_max_uids:
                self.unnormalized_scores = {uid: 0.0 for uid in range(self.effective_max_uids)}
            if not hasattr(self, 'trade_volumes') or len(self.trade_volumes) < self.effective_max_uids:
                self.trade_volumes = {
                    uid: {
                        bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                        for bookId in range(book_count)
                    }
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'initial_balances') or len(self.initial_balances) < self.effective_max_uids:
                self.initial_balances = {
                    uid: {
                        bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                        for bookId in range(book_count)
                    }
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'recent_miner_trades') or len(self.recent_miner_trades) < self.effective_max_uids:
                self.recent_miner_trades = {
                    uid: {bookId: [] for bookId in range(book_count)}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'miner_stats') or len(self.miner_stats) < self.effective_max_uids:
                self.miner_stats = {
                    uid: {'requests': 0, 'timeouts': 0, 'failures': 0, 'rejections': 0, 'call_time': []}
                    for uid in range(self.effective_max_uids)
                }
            if not hasattr(self, 'recent_trades'):
                self.recent_trades = {bookId: [] for bookId in range(book_count)}
            if not hasattr(self, 'fundamental_price'):
                self.fundamental_price = {bookId: None for bookId in range(book_count)}
            
            bt.logging.success(f"All structures initialized for {self.effective_max_uids} UIDs")
                
        def get_extended_metagraph(self):
            """Get metagraph extended with benchmark agents"""
            if not self.benchmark_agents:
                return self.metagraph

            extended_axons = [None] * self.effective_max_uids
            extended_hotkeys = [None] * self.effective_max_uids

            for uid, axon in enumerate(self.metagraph.axons):
                extended_axons[uid] = axon
                extended_hotkeys[uid] = self.metagraph.hotkeys[uid]

            for i, agent in enumerate(self.benchmark_agents):
                uid = self.benchmark_start_uid + i
                extended_axons[uid] = agent['axon']
                extended_hotkeys[uid] = agent['hotkey']

            placeholder_axon = bt.AxonInfo(
                version=self.metagraph.axons[0].version if self.metagraph.axons else 0,
                hotkey="",
                coldkey="",
                ip="0.0.0.0",
                port=0,
                ip_type=4,
                protocol=4,
                placeholder1=0,
                placeholder2=0,
            )
            
            for uid in range(self.effective_max_uids):
                if extended_axons[uid] is None:
                    extended_axons[uid] = placeholder_axon
                    extended_hotkeys[uid] = placeholder_axon.hotkey
            
            class ExtendedMetagraph:
                def __init__(self, uids, hotkeys, axons):
                    self.uids = torch.tensor(uids)
                    self.hotkeys = hotkeys
                    self.axons = axons
            
            return ExtendedMetagraph(
                list(range(self.effective_max_uids)),
                extended_hotkeys,
                extended_axons
            )

        def __enter__(self):
            """
            Enables use of Validator as a context manager.

            Returns:
                Validator: The active validator instance.
            """
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Ensures cleanup is triggered when exiting a context manager block.

            Args:
                exc_type: Exception type, if any.
                exc_val: Exception instance, if any.
                exc_tb: Traceback, if any.

            Returns:
                bool: False to propagate exceptions.
            """
            self.cleanup()
            return False

        @property
        def shared_state_rewarding(self):
            with self._rewarding_lock:
                return self._rewarding

        @shared_state_rewarding.setter
        def shared_state_rewarding(self, value):
            with self._rewarding_lock:
                self._rewarding = value

        @property
        def shared_state_saving(self):
            with self._saving_lock:
                return self._saving

        @shared_state_saving.setter
        def shared_state_saving(self, value):
            with self._saving_lock:
                self._saving = value

        @property
        def shared_state_reporting(self):
            with self._reporting_lock:
                return self._reporting

        @shared_state_reporting.setter
        def shared_state_reporting(self, value):
            with self._reporting_lock:
                self._reporting = value

        async def wait_for(self, check_fn: callable, message: str, interval: float = 0.01):
            """
            Asynchronously waits for a condition to become False.

            Behavior:
                - Logs a message once per second while waiting.
                - Returns immediately if condition is already False.
                - Provides debug timing on completion.

            Args:
                check_fn (callable): Function returning a boolean condition.
                message (str): Log message describing the wait condition.
                interval (float): Interval between checks in seconds.

            Returns:
                None
            """
            if not check_fn():
                return

            start_time = time.time()
            last_log_time = start_time

            bt.logging.info(message)

            while check_fn():
                await asyncio.sleep(interval)

                current_time = time.time()
                elapsed = current_time - start_time

                if current_time - last_log_time >= 1.0:
                    bt.logging.info(f"{message} (waited {elapsed:.1f}s)")
                    last_log_time = current_time

            total_wait = time.time() - start_time
            bt.logging.debug(f"Wait completed after {total_wait:.1f}s")

        async def wait_for_event(self, event: asyncio.Event, wait_process: str, run_process: str):
            """
            Waits for an asyncio.Event to be set before continuing execution.

            Provides periodic logging while waiting, and measures the total wait
            duration for operational visibility.

            Args:
                event (asyncio.Event): The event that must be completed.
                wait_process (str): Name of the process being waited on.
                run_process (str): Name of the process to run afterward.

            Returns:
                None
            """
            if not event.is_set():
                bt.logging.debug(f"Waiting for {wait_process} to complete before {run_process}...")
                start_wait = time.time()
                while not event.is_set():
                    try:
                        await asyncio.wait_for(event.wait(), timeout=1.0)
                        break
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0)
                        elapsed = time.time() - start_wait
                        if int(elapsed) % 5 == 0 and elapsed > 0:
                            bt.logging.debug(f"Still waiting for {wait_process}... ({elapsed:.1f}s)")
                total_wait = time.time() - start_wait
                bt.logging.debug(f"Waited {total_wait:.1f}s for {wait_process}")

        async def _write_ipc_nonblocking(self, mem, queue, data_bytes, operation="IPC", executor=None):
            """
            Write to shared memory in executor thread.
            """
            if executor is None:
                executor = self.query_ipc_executor
            write_timeout = 10.0

            def write_worker():
                try:
                    mem.seek(0)
                    mem.write(struct.pack('Q', len(data_bytes)))
                    mem.write(data_bytes)
                    return True
                except Exception as e:
                    self.pagerduty_alert(f"{operation} write failed: {e}")
                    return False

            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        executor,
                        write_worker
                    ),
                    timeout=write_timeout
                )
                return result
            except asyncio.TimeoutError:
                self.pagerduty_alert(f"{operation} write timeout after {write_timeout}s")
                return False

        def load_fundamental(self):
            """
            Loads fundamental price data from simulation output files.

            Behavior:
                - Reads per-block fundamental CSV files from the simulation log directory.
                - Extracts the latest fundamental price for each book ID.
                - Falls back to None for each book if no directory exists.
                - Stores results in `self.fundamental_price`.

            Returns:
                None
            """
            if self.simulation.logDir:
                prices = {}
                for block in range(self.simulation.block_count):
                    block_file = os.path.join(self.simulation.logDir, f'fundamental.{block * self.simulation.books_per_block}-{self.simulation.books_per_block * (block + 1) - 1}.csv')
                    fp_line = None
                    book_ids = None
                    for line in open(block_file, 'r').readlines():
                        if not book_ids:
                            book_ids = [int(col) for col in line.split(',') if col != "Timestamp\n"]
                        if line.strip() != '':
                            fp_line = line
                    prices = prices | {book_ids[i] : float(price) for i, price in enumerate(fp_line.strip().split(',')[:-1])}
            else:
                prices = {bookId : None for bookId in range(self.simulation.book_count)}
            self.fundamental_price = prices

        def onStart(self, timestamp, event : SimulationStartEvent) -> None:
            """
            Handles the simulator start event.

            Responsibilities:
                - Reloads simulation configuration.
                - Shifts timestamps for trade volumes, inventory, realized P&L, and round-trip volumes
                - Recalculates all volume sums to ensure consistency
                - Records simulation start time and timestamp.
                - Initializes output directory and launches log compression.
                - Loads fundamental prices for all books.
                - Resets initial balances and recent trade structures.
                - Clears open positions (can't carry over between simulations)
                - Saves initial state.

            Args:
                timestamp (int): Simulation start timestamp (always 0).
                event (SimulationStartEvent): Contains simulation log directory and metadata.

            Returns:
                None
            """
            self.load_simulation_config()
            volume_decimals = self.simulation.volumeDecimals

            bt.logging.info("Shifting timestamps for simulation restart...")

            old_simulation_timestamp = self.simulation_timestamp  # End time of old simulation
            new_simulation_timestamp = timestamp  # Start time of new simulation (0)

            lookback_period = self.config.scoring.kappa.lookback
            volume_assessment_period = self.config.scoring.activity.trade_volume_assessment_period

            new_threshold = new_simulation_timestamp - lookback_period
            new_volume_threshold = new_simulation_timestamp - volume_assessment_period

            pruned_total = defaultdict(lambda: defaultdict(float))
            pruned_maker = defaultdict(lambda: defaultdict(float))
            pruned_taker = defaultdict(lambda: defaultdict(float))
            pruned_self = defaultdict(lambda: defaultdict(float))
            pruned_roundtrip = defaultdict(lambda: defaultdict(float))

            # Shift trade volume timestamps
            bt.logging.info("Shifting trade volume timestamps...")
            shifted_trade_volumes = {}
            for uid in range(self.effective_max_uids):
                if uid in self.trade_volumes:
                    shifted_trade_volumes[uid] = {}
                    for bookId in range(self.simulation.book_count):
                        if bookId in self.trade_volumes[uid]:
                            shifted_trade_volumes[uid][bookId] = {}
                            for role in ['total', 'maker', 'taker', 'self']:
                                if role in self.trade_volumes[uid][bookId]:
                                    shifted_times = {}
                                    for prev_time, volume in self.trade_volumes[uid][bookId][role].items():
                                        time_from_old_end = old_simulation_timestamp - prev_time
                                        new_time = new_simulation_timestamp - time_from_old_end
                                        if new_time >= new_volume_threshold:
                                            shifted_times[new_time] = volume
                                        else:
                                            if role == 'total':
                                                pruned_total[uid][bookId] += volume
                                            elif role == 'maker':
                                                pruned_maker[uid][bookId] += volume
                                            elif role == 'taker':
                                                pruned_taker[uid][bookId] += volume
                                            elif role == 'self':
                                                pruned_self[uid][bookId] += volume

                                    if shifted_times:
                                        shifted_trade_volumes[uid][bookId][role] = shifted_times

            self.trade_volumes = {
                uid: {
                    bookId: {
                        role: shifted_trade_volumes.get(uid, {}).get(bookId, {}).get(role, {})
                        for role in ['total', 'maker', 'taker', 'self']
                    }
                    for bookId in range(self.simulation.book_count)
                }
                for uid in range(self.effective_max_uids)
            }

            bt.logging.info("Adjusting volume sums for pruned data...")
            for uid in pruned_total:
                for bookId in pruned_total[uid]:
                    self.volume_sums[uid][bookId] = max(
                        0.0,
                        self.volume_sums[uid][bookId] - pruned_total[uid][bookId]
                    )
                    self.volume_sums[uid][bookId] = round(self.volume_sums[uid][bookId], volume_decimals)

            for uid in pruned_maker:
                for bookId in pruned_maker[uid]:
                    self.maker_volume_sums[uid][bookId] = max(
                        0.0,
                        self.maker_volume_sums[uid][bookId] - pruned_maker[uid][bookId]
                    )
                    self.maker_volume_sums[uid][bookId] = round(self.maker_volume_sums[uid][bookId], volume_decimals)

            for uid in pruned_taker:
                for bookId in pruned_taker[uid]:
                    self.taker_volume_sums[uid][bookId] = max(
                        0.0,
                        self.taker_volume_sums[uid][bookId] - pruned_taker[uid][bookId]
                    )
                    self.taker_volume_sums[uid][bookId] = round(self.taker_volume_sums[uid][bookId], volume_decimals)

            for uid in pruned_self:
                for bookId in pruned_self[uid]:
                    self.self_volume_sums[uid][bookId] = max(
                        0.0,
                        self.self_volume_sums[uid][bookId] - pruned_self[uid][bookId]
                    )
                    self.self_volume_sums[uid][bookId] = round(self.self_volume_sums[uid][bookId], volume_decimals)

            bt.logging.info(f"Adjusted volume sums after pruning old data")

            bt.logging.info("Shifting inventory history timestamps...")
            shifted_inventory = {}
            for uid in range(self.effective_max_uids):
                if uid in self.inventory_history and self.inventory_history[uid]:
                    hist = self.inventory_history[uid]
                    if len(hist) > 3:
                        timestamps_to_keep = sorted(hist.keys())[-3:]
                        hist = {ts: hist[ts] for ts in timestamps_to_keep}

                    shifted_inventory[uid] = {}
                    for prev_time, values in hist.items():
                        time_from_old_end = old_simulation_timestamp - prev_time
                        new_time = new_simulation_timestamp - time_from_old_end
                        shifted_inventory[uid][new_time] = values

            self.inventory_history = {
                uid: shifted_inventory.get(uid, {})
                for uid in range(self.effective_max_uids)
            }

            bt.logging.info("Shifting realized P&L history timestamps...")
            shifted_pnl_history = {}
            self._last_prune_timestamp = None
            for uid in range(self.effective_max_uids):
                if uid in self.realized_pnl_history and self.realized_pnl_history[uid]:
                    hist = self.realized_pnl_history[uid]
                    shifted_pnl_history[uid] = {}
                    for prev_time, books in hist.items():
                        time_from_old_end = old_simulation_timestamp - prev_time
                        new_time = new_simulation_timestamp - time_from_old_end
                        if new_time >= new_threshold:
                            shifted_pnl_history[uid][new_time] = books

            self.realized_pnl_history = defaultdict(lambda: defaultdict(dict))
            for uid, timestamps_data in shifted_pnl_history.items():
                for ts, books in timestamps_data.items():
                    for book_id, pnl in books.items():
                        self.realized_pnl_history[uid][ts][book_id] = pnl

            bt.logging.info(f"Shifted realized P&L history: {len(shifted_pnl_history)} UIDs with data")

            bt.logging.info("Shifting round-trip volume timestamps...")
            shifted_rt_volumes = {}
            for uid in range(self.effective_max_uids):
                if uid in self.roundtrip_volumes:
                    shifted_rt_volumes[uid] = {}
                    for bookId in range(self.simulation.book_count):
                        if bookId in self.roundtrip_volumes[uid]:
                            shifted_times = {}
                            for prev_time, volume in self.roundtrip_volumes[uid][bookId].items():
                                time_from_old_end = old_simulation_timestamp - prev_time
                                new_time = new_simulation_timestamp - time_from_old_end

                                if new_time >= new_volume_threshold:
                                    shifted_times[new_time] = volume
                                else:
                                    pruned_roundtrip[uid][bookId] += volume

                            if shifted_times:
                                shifted_rt_volumes[uid][bookId] = shifted_times

            self.roundtrip_volumes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            for uid, books in shifted_rt_volumes.items():
                for book_id, volumes in books.items():
                    for ts, volume in volumes.items():
                        self.roundtrip_volumes[uid][book_id][ts] = volume

            bt.logging.info(f"Shifted round-trip volumes: {len(shifted_rt_volumes)} UIDs with data")

            bt.logging.info("Adjusting round-trip volume sums for pruned data...")
            for uid in pruned_roundtrip:
                for bookId in pruned_roundtrip[uid]:
                    self.roundtrip_volume_sums[uid][bookId] = max(
                        0.0,
                        self.roundtrip_volume_sums[uid][bookId] - pruned_roundtrip[uid][bookId]
                    )
                    self.roundtrip_volume_sums[uid][bookId] = round(
                        self.roundtrip_volume_sums[uid][bookId],
                        volume_decimals
                    )

            bt.logging.info(f"Adjusted round-trip volume sums: {len(self.roundtrip_volume_sums)} total entries")

            self.start_time = time.time()
            self.simulation_timestamp = timestamp
            self.start_timestamp = self.simulation_timestamp
            self.last_state_time = None
            self.step_rates = []
            if event.logDir != self.simulation.logDir:
                bt.logging.info(f"Simulation log directory changed: {self.simulation.logDir} -> {event.logDir}")
                self._notify_seed_log_dir_change(event.logDir)
                self.simulation.logDir = event.logDir
            bt.logging.info("Clearing open positions (simulation-specific state)...")
            self.open_positions = defaultdict(lambda: defaultdict(lambda: {
                'longs': deque(),
                'shorts': deque()
            }))

            self.compress_outputs(start=True)

            bt.logging.info("-"*40)
            bt.logging.info("SIMULATION STARTED")
            bt.logging.info("-"*40)
            bt.logging.info(f"START TIME: {self.start_time}")
            bt.logging.info(f"TIMESTAMP : {self.start_timestamp}")
            bt.logging.info(f"OUT DIR   : {self.simulation.logDir}")
            bt.logging.info("-"*40)

            self.load_fundamental()
            self.initial_balances = {
                uid : {
                    bookId : {'BASE' : None, 'QUOTE' : None, 'WEALTH' : self.simulation.miner_wealth}
                    for bookId in range(self.simulation.book_count)
                } for uid in range(self.effective_max_uids)
            }
            self.recent_trades = {bookId : [] for bookId in range(self.simulation.book_count)}
            self.recent_miner_trades = {
                uid : {bookId : [] for bookId in range(self.simulation.book_count)}
                for uid in range(self.effective_max_uids)
            }

            asyncio.run_coroutine_threadsafe(self._save_state_sync(), self.main_loop).result()
            bt.logging.info("Simulation restart complete")

        def onEnd(self) -> None:
            """
            Triggered when end of simulation event is published by simulator.
            Resets quantities as necessary, updates, rebuilds and launches simulator with the latest configuration.
            """
            bt.logging.info("SIMULATION ENDED")
            self.simulation.logDir = None
            self._notify_seed_log_dir_change(None)
            self.fundamental_price = {bookId : None for bookId in range(self.simulation.book_count)}
            self.pending_notices = {uid : [] for uid in range(self.effective_max_uids)}
            asyncio.run_coroutine_threadsafe(self._save_state_sync(), self.main_loop).result()
            self.update_repo(end=True)

        def _build_simulation_state(self):
            """Build simulation state dictionary.
            
            Constructs a dictionary containing all simulation-level state that needs
            to be persisted, including timing information, balances, recent trades,
            and pending notices.
            
            Returns:
                dict: Simulation state containing start_time, start_timestamp, step_rates,
                    initial_balances, recent_trades, recent_miner_trades, pending_notices,
                    and simulation.logDir.
            """
            return {
                "start_time": self.start_time,
                "start_timestamp": self.start_timestamp,
                "step_rates": list(self.step_rates),
                "initial_balances": self.initial_balances,
                "recent_trades": {
                    book_id: [t.model_dump(mode="json") for t in trades]
                    for book_id, trades in self.recent_trades.items()
                },
                "recent_miner_trades": {
                    uid: {
                        book_id: [[t.model_dump(mode="json"), r] for t, r in trades]
                        for book_id, trades in uid_trades.items()
                    }
                    for uid, uid_trades in self.recent_miner_trades.items()
                },
                "pending_notices": self.pending_notices,
                "simulation.logDir": self.simulation.logDir,
            }

        def _snapshot_inventory_history(self):
            """Snapshot inventory_history to eliminate nested defaultdict references.
            
            Creates a clean snapshot of the inventory_history structure by converting
            nested defaultdicts to regular dicts for all UIDs up to max_uids.
            
            Returns:
                dict: Nested dictionary mapping uid -> timestamp -> book_id -> inventory.
            """
            result = {}
            for uid in range(self.effective_max_uids):
                if uid in self.inventory_history and self.inventory_history[uid]:
                    result[uid] = {}
                    for ts, books in self.inventory_history[uid].items():
                        result[uid][ts] = dict(books)
            return result

        def _snapshot_realized_pnl_history(self):
            """Snapshot realized_pnl_history to eliminate nested defaultdict references.
            
            Creates a clean snapshot of the realized PnL history structure by converting
            nested defaultdicts to regular dicts for all UIDs up to max_uids.
            
            Returns:
                dict: Nested dictionary mapping uid -> timestamp -> book_id -> pnl.
            """
            result = {}
            for uid in range(self.effective_max_uids):
                if uid in self.realized_pnl_history:
                    result[uid] = {}
                    for ts, books in self.realized_pnl_history[uid].items():
                        result[uid][ts] = dict(books)
            return result

        def _snapshot_2_level_dict(self, source_dict):
            """Snapshot a 2-level dict structure to eliminate defaultdict references.
            
            Utility method for converting 2-level nested defaultdict structures to
            regular dicts for all UIDs up to max_uids.
            
            Args:
                source_dict (dict): Source dictionary with structure uid -> book_id -> value.
                
            Returns:
                dict: Snapshot with regular dict objects replacing defaultdicts.
            """
            result = {}
            for uid in range(self.effective_max_uids):
                if uid in source_dict:
                    result[uid] = dict(source_dict[uid])
            return result

        def _snapshot_volume_sums(self):
            """Snapshot all volume sum dictionaries.
            
            Creates snapshots of all volume-related tracking dictionaries including
            total volumes, maker/taker volumes, self-trade volumes, and roundtrip volumes.
            
            Returns:
                dict: Dictionary containing snapshots of volume_sums, maker_volume_sums,
                    taker_volume_sums, self_volume_sums, and roundtrip_volume_sums.
            """
            return {
                'volume_sums': self._snapshot_2_level_dict(self.volume_sums),
                'maker_volume_sums': self._snapshot_2_level_dict(self.maker_volume_sums),
                'taker_volume_sums': self._snapshot_2_level_dict(self.taker_volume_sums),
                'self_volume_sums': self._snapshot_2_level_dict(self.self_volume_sums),
                'roundtrip_volume_sums': self._snapshot_2_level_dict(self.roundtrip_volume_sums),
            }

        def _snapshot_trade_volumes(self):
            """Snapshot trade_volumes with full 3-level structure preservation.
            
            Creates a clean snapshot of the trade_volumes structure which tracks volumes
            by UID, book ID, role (maker/taker), and timestamp.
            
            Returns:
                dict: Nested dictionary mapping uid -> book_id -> role -> timestamp -> volume.
            """
            result = {}
            for uid in range(self.effective_max_uids):
                if uid not in self.trade_volumes:
                    continue
                result[uid] = {}
                for book_id, roles in self.trade_volumes[uid].items():
                    result[uid][book_id] = {}
                    for role, volumes in roles.items():
                        result[uid][book_id][role] = dict(volumes)
            return result

        def _snapshot_roundtrip_volumes(self):
            """Snapshot roundtrip_volumes with full 3-level structure preservation.
            
            Creates a clean snapshot of the roundtrip_volumes structure which tracks
            roundtrip trading volumes by UID, book ID, and timestamp.
            
            Returns:
                dict: Nested dictionary mapping uid -> book_id -> timestamp -> volume.
            """
            result = {}
            for uid in range(self.effective_max_uids):
                if uid not in self.roundtrip_volumes:
                    continue
                result[uid] = {}
                for book_id, volumes in self.roundtrip_volumes[uid].items():
                    result[uid][book_id] = dict(volumes)
            return result

        def _snapshot_open_positions(self):
            """Snapshot open_positions with explicit long/short list copies.
            
            Creates a clean snapshot of the open_positions structure which tracks
            currently open long and short positions by UID and book ID.
            
            Returns:
                dict: Nested dictionary mapping uid -> book_id -> {'longs': list, 'shorts': list}.
            """
            result = {}
            for uid in range(self.effective_max_uids):
                if uid not in self.open_positions:
                    continue
                result[uid] = {}
                for book_id, pos in self.open_positions[uid].items():
                    result[uid][book_id] = {
                        'longs': list(pos.get('longs', [])),
                        'shorts': list(pos.get('shorts', []))
                    }
            return result

        def _build_validator_state(
            self,
            inventory_snapshot,
            realized_pnl_snapshot,
            volume_sums_snapshots,
            trade_volumes_snapshot,
            roundtrip_volumes_snapshot,
            open_positions_snapshot
        ):
            """Assemble validator state dict from all snapshots.
            
            This method is shared by both sync and async construction paths
            to ensure consistency and maintainability.
            
            Args:
                inventory_snapshot (dict): Snapshot of inventory_history.
                realized_pnl_snapshot (dict): Snapshot of realized_pnl_history.
                volume_sums_snapshots (dict): Dictionary containing all volume sum snapshots.
                trade_volumes_snapshot (dict): Snapshot of trade_volumes.
                roundtrip_volumes_snapshot (dict): Snapshot of roundtrip_volumes.
                open_positions_snapshot (dict): Snapshot of open_positions.
                
            Returns:
                dict: Complete validator state containing step, simulation_timestamp, hotkeys,
                    scores, activity_factors, pnl_factors, and all snapshot data.
            """
            return {
                "step": self.step,
                "simulation_timestamp": self.simulation_timestamp,
                "hotkeys": self.hotkeys,
                "scores": [score.item() for score in self.scores],
                "gentrx_scores": [score.item() for score in self.gentrx_scores],
                "activity_factors": self.activity_factors,
                "pnl_factors": self.pnl_factors,
                "inventory_history": inventory_snapshot,
                "kappa_values": self.kappa_values,
                "realized_pnl_history": realized_pnl_snapshot,
                "open_positions": open_positions_snapshot,
                "unnormalized_scores": self.unnormalized_scores,
                "deregistered_uids": self.deregistered_uids,
                "trade_volumes": trade_volumes_snapshot,
                "roundtrip_volumes": roundtrip_volumes_snapshot,
                "volume_sums": volume_sums_snapshots['volume_sums'],
                "maker_volume_sums": volume_sums_snapshots['maker_volume_sums'],
                "taker_volume_sums": volume_sums_snapshots['taker_volume_sums'],
                "self_volume_sums": volume_sums_snapshots['self_volume_sums'],
                "roundtrip_volume_sums": volume_sums_snapshots['roundtrip_volume_sums'],
            }

        def _construct_save_data_sync(self):
            """Synchronously build all state data for saving.
            
            Constructs both simulation and validator state by calling each step method
            directly in sequence. This is the synchronous alternative to the async
            construction path.
            
            Returns:
                tuple: Three-element tuple containing:
                    - simulation_state_data (dict): Simulation state dictionary.
                    - validator_state_data (dict): Validator state dictionary.
                    - prep_time (float): Total time spent preparing the data in seconds.
            """
            start = time.time()
            bt.logging.debug("Preparing state for saving (sync)...")

            simulation_state_data = self._build_simulation_state()

            bt.logging.debug("Creating snapshots...")
            snapshot_start = time.time()

            inventory_snapshot = self._snapshot_inventory_history()
            realized_pnl_snapshot = self._snapshot_realized_pnl_history()
            volume_sums_snapshots = self._snapshot_volume_sums()
            trade_volumes_snapshot = self._snapshot_trade_volumes()
            roundtrip_volumes_snapshot = self._snapshot_roundtrip_volumes()
            open_positions_snapshot = self._snapshot_open_positions()

            bt.logging.debug(f"Created snapshots ({time.time()-snapshot_start:.4f}s)")

            validator_state_data = self._build_validator_state(
                inventory_snapshot,
                realized_pnl_snapshot,
                volume_sums_snapshots,
                trade_volumes_snapshot,
                roundtrip_volumes_snapshot,
                open_positions_snapshot
            )

            prep_time = time.time() - start
            bt.logging.info(f"Prepared save data sync ({prep_time:.4f}s)")
            return simulation_state_data, validator_state_data, prep_time

        async def _construct_save_data_async(self):
            """Asynchronously build all state data for saving.
            
            Constructs both simulation and validator state by running each step method
            in an executor thread pool and yielding control between steps. This allows
            other async operations to continue while state preparation is in progress.
            
            Returns:
                tuple: Three-element tuple containing:
                    - simulation_state_data (dict): Simulation state dictionary.
                    - validator_state_data (dict): Validator state dictionary.
                    - prep_time (float): Total time spent preparing the data in seconds.
            """
            start = time.time()
            bt.logging.debug("Preparing state for saving (async)...")
            loop = asyncio.get_event_loop()

            simulation_state_data = await loop.run_in_executor(
                self.save_state_executor,
                self._build_simulation_state
            )
            await asyncio.sleep(0)

            bt.logging.debug("Creating snapshots...")
            snapshot_start = time.time()

            inventory_snapshot = await loop.run_in_executor(
                self.save_state_executor,
                self._snapshot_inventory_history
            )
            await asyncio.sleep(0)

            realized_pnl_snapshot = await loop.run_in_executor(
                self.save_state_executor,
                self._snapshot_realized_pnl_history
            )
            await asyncio.sleep(0)

            volume_sums_snapshots = await loop.run_in_executor(
                self.save_state_executor,
                self._snapshot_volume_sums
            )
            await asyncio.sleep(0)

            trade_volumes_snapshot = await loop.run_in_executor(
                self.save_state_executor,
                self._snapshot_trade_volumes
            )
            await asyncio.sleep(0)

            roundtrip_volumes_snapshot = await loop.run_in_executor(
                self.save_state_executor,
                self._snapshot_roundtrip_volumes
            )
            await asyncio.sleep(0)

            open_positions_snapshot = await loop.run_in_executor(
                self.save_state_executor,
                self._snapshot_open_positions
            )
            await asyncio.sleep(0)

            bt.logging.debug(f"Created snapshots ({time.time()-snapshot_start:.4f}s)")

            validator_state_data = self._build_validator_state(
                inventory_snapshot,
                realized_pnl_snapshot,
                volume_sums_snapshots,
                trade_volumes_snapshot,
                roundtrip_volumes_snapshot,
                open_positions_snapshot
            )

            prep_time = time.time() - start
            bt.logging.info(f"Prepared save data async ({prep_time:.4f}s)")
            return simulation_state_data, validator_state_data, prep_time

        def _defragment_histories(self):
            """Rebuild history dicts to eliminate memory fragmentation.
            
            Reconstructs all major history and volume tracking dictionaries to reduce
            memory fragmentation that accumulates over time from repeated insertions
            and deletions. This helps maintain consistent memory usage patterns.
            
            The following structures are defragmented:
                - inventory_history
                - realized_pnl_history
                - trade_volumes
                - roundtrip_volumes
                - volume_sums (all variants)
            
            Side Effects:
                Triggers garbage collection after rebuilding all structures.
            """
            start = time.time()
            bt.logging.info("Defragmenting history structures...")

            # Rebuild inventory_history
            new_inv = {}
            for uid, hist in self.inventory_history.items():
                if hist:
                    new_inv[uid] = {ts: dict(books) for ts, books in hist.items()}
            self.inventory_history = new_inv

            # Rebuild realized_pnl_history
            new_pnl = defaultdict(lambda: defaultdict(dict))
            for uid, timestamps in self.realized_pnl_history.items():
                for ts, books in timestamps.items():
                    new_pnl[uid][ts] = dict(books)
            self.realized_pnl_history = new_pnl

            # Rebuild trade_volumes
            new_volumes = {}
            for uid, books in self.trade_volumes.items():
                new_volumes[uid] = {}
                for book_id, roles in books.items():
                    new_volumes[uid][book_id] = {
                        role: dict(volumes) for role, volumes in roles.items()
                    }
            self.trade_volumes = new_volumes

            # Rebuild roundtrip_volumes
            new_rt = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            for uid, books in self.roundtrip_volumes.items():
                for book_id, volumes in books.items():
                    for ts, vol in volumes.items():
                        new_rt[uid][book_id][ts] = vol
            self.roundtrip_volumes = new_rt

            # Rebuild volume sums (convert defaultdict to regular dict then back)
            self.volume_sums = defaultdict(lambda: defaultdict(float), {
                uid: dict(books) for uid, books in self.volume_sums.items()
            })
            self.maker_volume_sums = defaultdict(lambda: defaultdict(float), {
                uid: dict(books) for uid, books in self.maker_volume_sums.items()
            })
            self.taker_volume_sums = defaultdict(lambda: defaultdict(float), {
                uid: dict(books) for uid, books in self.taker_volume_sums.items()
            })
            self.self_volume_sums = defaultdict(lambda: defaultdict(float), {
                uid: dict(books) for uid, books in self.self_volume_sums.items()
            })
            self.roundtrip_volume_sums = defaultdict(lambda: defaultdict(float), {
                uid: dict(books) for uid, books in self.roundtrip_volume_sums.items()
            })
            
            gc.collect()
            bt.logging.info(f"Defragmentation complete ({time.time()-start:.4f}s)")

        async def _save_state(self) -> bool:
            """
            Saves simulation and validator state asynchronously via executor workers.
            Uses atomic file writes (temp + rename) to prevent corruption.
            Backups are organized by simulation ID and timestamp.

            Returns:
                bool: True if state saved successfully, False otherwise.
            """
            if self.shared_state_saving:
                bt.logging.warning(f"Skipping save at step {self.step} — previous save still running.")
                return False

            self.shared_state_saving = True

            try:
                bt.logging.info(f"Starting state saving for step {self.step}...")
                total_start = time.time()

                save_step = self.step
                save_timestamp = self.simulation_timestamp
                sim_id = self.simulation.simulation_id

                # Defragment at the start of each simulation hour
                simulation_hour = save_timestamp // 3600_000_000_000
                last_defrag_hour = getattr(self, '_last_defrag_hour', -1)

                if simulation_hour != last_defrag_hour:
                    bt.logging.info(f"Simulation hour {simulation_hour} - triggering defragmentation")
                    async with self._reward_lock:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            self.save_state_executor,
                            self._defragment_histories
                        )
                    self._last_defrag_hour = simulation_hour

                async with self._reward_lock:
                    total_inv_entries = sum(len(hist) for hist in self.inventory_history.values())
                    total_pnl_entries = sum(len(hist) for hist in self.realized_pnl_history.values())
                    total_vol_entries = sum(
                        sum(len(roles.get('total', {})) for roles in books.values())
                        for books in self.trade_volumes.values()
                    )

                    bt.logging.info(
                        f"Save prep: {total_inv_entries} inventory entries, "
                        f"{total_pnl_entries} P&L entries, {total_vol_entries} volume entries"
                    )

                    simulation_state_data, validator_state_data, prep_time = await self._construct_save_data_async()
                    
                async def wait_for_query_and_receive(process):
                    """
                    Wait for both query completion AND state reception before continuing.
                    This prevents IO contention between mmap reads and msgpack serialization.
                    """
                    wait_start = time.time()
                    waited = False
                    while self.querying:
                        if not waited:
                            bt.logging.debug(f"Waiting for query to complete before {process}...")
                            waited = True
                        await asyncio.sleep(0.05)
                    while self._receiving_state:
                        if not waited:
                            bt.logging.debug(f"Waiting for state reception to complete before {process}...")
                            waited = True
                        await asyncio.sleep(0.05)
                    wait_time = time.time() - wait_start
                    if wait_time > 0.01:
                        bt.logging.info(f"Waited {wait_time:.4f}s for query/reception to complete before {process}")
                    await asyncio.sleep(0)

                async def async_save_worker():
                    """Non-blocking async file I/O with atomic writes."""
                    try:
                        await wait_for_query_and_receive('serializing simulation state')
                        sim_serialize_start = time.time()
                        loop = asyncio.get_event_loop()
                        sim_bytes = await loop.run_in_executor(
                            self.save_state_executor,
                            lambda: msgpack.packb(simulation_state_data, use_bin_type=True)
                        )
                        sim_serialize_time = time.time() - sim_serialize_start
                        await wait_for_query_and_receive('serializing validator state')
                        val_serialize_start = time.time()
                        val_bytes = await loop.run_in_executor(
                            self.save_state_executor,
                            lambda: msgpack.packb(validator_state_data, use_bin_type=True)
                        )
                        val_serialize_time = time.time() - val_serialize_start

                        sim_mb = len(sim_bytes) / 1024 / 1024
                        val_mb = len(val_bytes) / 1024 / 1024
                        bt.logging.info(
                            f"Serialized: sim={sim_mb:.2f}MB, val={val_mb:.2f}MB "
                            f"({sim_serialize_time + val_serialize_time:.4f}s)"
                        )
                        await wait_for_query_and_receive('writing simulation state file')
                        sim_start = time.time()
                        sim_temp = f"{self.simulation_state_file}.tmp.{save_timestamp}"
                        try:
                            async with aiofiles.open(sim_temp, 'wb') as f:
                                await f.write(sim_bytes)
                            await aiofiles.os.replace(sim_temp, self.simulation_state_file)
                        except Exception as ex:
                            if os.path.exists(sim_temp):
                                try:
                                    await aiofiles.os.remove(sim_temp)
                                except:
                                    pass
                            raise ex
                        sim_time = time.time() - sim_start

                        await wait_for_query_and_receive('writing validator state file')
                        val_start = time.time()
                        val_temp = f"{self.validator_state_file}.tmp.{save_timestamp}"
                        try:
                            async with aiofiles.open(val_temp, 'wb') as f:
                                await f.write(val_bytes)
                            await aiofiles.os.replace(val_temp, self.validator_state_file)
                        except Exception as ex:
                            if os.path.exists(val_temp):
                                try:
                                    await aiofiles.os.remove(val_temp)
                                except:
                                    pass
                            raise ex
                        val_time = time.time() - val_start

                        return {
                            'success': True,
                            'simulation_save_time': sim_time,
                            'validator_save_time': val_time,
                            'io_time': sim_serialize_time + val_serialize_time + sim_time + val_time,
                            'sim_size_mb': sim_mb,
                            'val_size_mb': val_mb
                        }
                    except Exception as ex:
                        return {
                            'success': False,
                            'error': str(ex),
                            'traceback': traceback.format_exc()
                        }

                result = await async_save_worker()
                total_time = time.time() - total_start
                if result['success']:
                    bt.logging.success(
                        f"State saved: simulation ({result['simulation_save_time']:.4f}s, {result['sim_size_mb']:.2f}MB), "
                        f"validator ({result['validator_save_time']:.4f}s, {result['val_size_mb']:.2f}MB), "
                        f"total ({total_time:.4f}s, prep={prep_time:.4f}s, io={result['io_time']:.4f}s)"
                    )

                    await wait_for_query_and_receive('creating state backups')
                    max_backups = 5
                    for state_file in [self.simulation_state_file, self.validator_state_file]:
                        if os.path.exists(state_file):
                            backup = f"{state_file}.{sim_id}.{save_timestamp}"
                            try:
                                async with aiofiles.open(state_file, 'rb') as src:
                                    content = await src.read()
                                async with aiofiles.open(backup, 'wb') as dst:
                                    await dst.write(content)
                                bt.logging.debug(f"Created backup: {backup}")
                            except Exception as ex:
                                bt.logging.warning(f"Failed to create backup {backup}: {ex}")
                            state_path = Path(state_file)
                            try:
                                all_backups = []
                                for backup_file in state_path.parent.glob(f"{state_path.name}.*.*"):
                                    try:
                                        parts = backup_file.name.split('.')
                                        if len(parts) >= 3:
                                            sim_id = parts[-2]
                                            timestamp = int(parts[-1])
                                            all_backups.append((backup_file, sim_id, timestamp))
                                    except:
                                        continue
                                all_backups.sort(key=lambda x: (x[1], x[2]), reverse=True)
                                for old_backup, _, _ in all_backups[max_backups:]:
                                    try:
                                        await aiofiles.os.remove(old_backup)
                                        bt.logging.debug(f"Removed old backup: {old_backup.name}")
                                    except Exception as ex:
                                        bt.logging.warning(f"Failed to remove {old_backup.name}: {ex}")
                            except Exception as ex:
                                bt.logging.warning(f"Failed to clean old backups: {ex}")
                else:
                    self.pagerduty_alert(
                        f"Failed to save state for step {save_step}: {result['error']}",
                        details={"trace": result.get('traceback')}
                    )
                return result['success']

            except Exception as ex:
                self.pagerduty_alert(
                    f"Failed to prepare state for save: {ex}",
                    details={"trace": traceback.format_exc()}
                )
                return False
            finally:
                self.shared_state_saving = False
                bt.logging.debug(f"Released save lock for step {save_step}")

        def save_state(self) -> None:
            """
            Schedules the asynchronous state-saving coroutine on the main event loop.

            Behavior:
                - Executes only at specific scoring intervals.
                - Ensures no previous save task is still running.
                - Dispatches `_save_state()` thread-safely from the maintenance thread.

            Returns:
                None
            """
            if not self.last_state or self.last_state.timestamp % self.config.scoring.interval != 4_000_000_000:
                return
            if self.shared_state_saving:
                bt.logging.warning(f"Skipping save at step {self.step} — previous save still running.")
                return
            if self.querying:
                bt.logging.warning(f"Skipping save at step {self.step} — query in progress")
                return
            bt.logging.debug(f"[SAVE] Scheduling from thread: {threading.current_thread().name}")
            bt.logging.debug(f"[SAVE] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
            self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._save_state()))

        async def _save_state_sync(self):
            """Save validator state synchronously from an async context.
            
            The method performs the following steps:
                1. Constructs simulation and validator state data synchronously
                2. Writes both state files to disk via save_state_worker
                3. Sends PagerDuty alert if save operation fails
            
            Raises:
                Exception: If state construction or save operation fails. Exceptions
                    are caught, logged, and trigger PagerDuty alerts.
                    
            Side Effects:
                - Writes simulation_state_file and validator_state_file to disk
                - Sends PagerDuty alerts on failure
                - Logs save operation status and timing
            """
            try:
                bt.logging.info("Saving state (sync)...")
                simulation_state_data, validator_state_data, prep_time = self._construct_save_data_sync()
                result = save_state_worker(
                    simulation_state_data,
                    validator_state_data,
                    self.simulation_state_file,
                    self.validator_state_file
                )
                if result['success']:
                    bt.logging.success(f"State saved directly!")
                else:
                    self.pagerduty_alert(f"Direct save failed: {result['error']}")
            except Exception as ex:
                self.pagerduty_alert(f"Error in direct save: {ex}", details={"trace": traceback.format_exc()})
                
        def migrate_sampling_interval(self, old_interval: int, new_interval: int):
            """
            Re-align trade volumes from old sampling interval to new interval.
            
            Uses floor bucketing: trades are assigned to the start of their interval bucket.
            
            Args:
                old_interval: Previous sampling interval (e.g., 600_000_000_000 for 10 min)
                new_interval: New sampling interval (e.g., 3600_000_000_000 for 1 hour)
            """
            bt.logging.info(
                f"Migrating sampling intervals: "
                f"{old_interval}ns ({old_interval / 1e9:.0f}s) → "
                f"{new_interval}ns ({new_interval / 1e9:.0f}s)"
            )
            
            start_time = time.time()
            volume_decimals = self.simulation.volumeDecimals
            
            migrated_trade_entries = 0
            migrated_rt_entries = 0

            bt.logging.info("Migrating trade_volumes...")
            for uid in range(self.effective_max_uids):
                if uid not in self.trade_volumes:
                    continue
                for book_id in range(self.simulation.book_count):
                    if book_id not in self.trade_volumes[uid]:
                        continue
                    for role in ['total', 'maker', 'taker', 'self']:
                        old_volumes = self.trade_volumes[uid][book_id][role]
                        if not old_volumes:
                            continue
                        new_volumes = {}
                        for old_ts, volume in old_volumes.items():
                            new_ts = (old_ts // new_interval) * new_interval
                            if new_ts not in new_volumes:
                                new_volumes[new_ts] = 0.0
                            new_volumes[new_ts] += volume
                            migrated_trade_entries += 1
                        for ts in new_volumes:
                            new_volumes[ts] = round(new_volumes[ts], volume_decimals)
                        self.trade_volumes[uid][book_id][role] = new_volumes
            bt.logging.info(f"Migrated {migrated_trade_entries} trade_volume entries")

            bt.logging.info("Migrating roundtrip_volumes...")
            new_roundtrip_volumes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            for uid in range(self.effective_max_uids):
                if uid not in self.roundtrip_volumes:
                    continue
                for book_id in range(self.simulation.book_count):
                    if book_id not in self.roundtrip_volumes[uid]:
                        continue
                    old_rt_volumes = self.roundtrip_volumes[uid][book_id]
                    if not old_rt_volumes:
                        continue
                    new_rt_volumes = {}
                    for old_ts, volume in old_rt_volumes.items():
                        new_ts = (old_ts // new_interval) * new_interval
                        if new_ts not in new_rt_volumes:
                            new_rt_volumes[new_ts] = 0.0
                        new_rt_volumes[new_ts] += volume
                        migrated_rt_entries += 1
                    for ts in new_rt_volumes:
                        new_rt_volumes[ts] = round(new_rt_volumes[ts], volume_decimals)
                    for ts, vol in new_rt_volumes.items():
                        new_roundtrip_volumes[uid][book_id][ts] = vol

            self.roundtrip_volumes = new_roundtrip_volumes
            bt.logging.info(f"Migrated {migrated_rt_entries} roundtrip_volume entries")
            
            elapsed = time.time() - start_time
            bt.logging.success(
                f"Sampling interval migration complete ({elapsed:.4f}s): "
                f"{migrated_trade_entries + migrated_rt_entries} total entries migrated"
            )

        def load_state(self) -> None:
            """
            Loads validator and simulation state from msgpack or legacy PyTorch files.

            Behavior:
                - Converts `.pt` state files to msgpack if detected.
                - Loads simulation variables (balances, trades, notices, timestamps).
                - Loads validator data (scores, activity, Kappa values, volumes).
                - Reconstructs missing fields or reshapes data when schema versions differ.
                - Handles changes in effective_max_uids (benchmark agent count changes).
                - Reinitializes state when `neuron.reset=True` or files are absent.

            Returns:
                None
            """
            if os.path.exists(self.simulation_state_file.replace('.mp', '.pt')):
                bt.logging.info("Pytorch simulation state file exists - converting to msgpack...")
                pt_simulation_state = torch.load(self.simulation_state_file.replace('.mp', '.pt'), weights_only=False)
                with open(self.simulation_state_file, 'wb') as file:
                    packed_data = msgpack.packb(
                        {
                            "start_time": pt_simulation_state['start_time'],
                            "start_timestamp": pt_simulation_state['start_timestamp'],
                            "step_rates": pt_simulation_state['step_rates'],
                            "initial_balances": pt_simulation_state['initial_balances'],
                            "recent_trades": {book_id : [t.model_dump(mode='json') for t in book_trades] for book_id, book_trades in pt_simulation_state['recent_trades'].items()},
                            "recent_miner_trades": {uid : {book_id : [[t.model_dump(mode='json'), r] for t, r in trades] for book_id, trades in uid_miner_trades.items()} for uid, uid_miner_trades in pt_simulation_state['recent_miner_trades'].items()},
                            "pending_notices": pt_simulation_state['pending_notices'],
                            "simulation.logDir": pt_simulation_state['simulation.logDir']
                        }, use_bin_type=True
                    )
                    file.write(packed_data)
                os.rename(self.simulation_state_file.replace('.mp', '.pt'), self.simulation_state_file.replace('.mp', '.pt') + ".bak")
                bt.logging.info(f"Pytorch simulation state file converted to msgpack at {self.simulation_state_file}")

            if not self.config.neuron.reset and os.path.exists(self.simulation_state_file):
                bt.logging.info(f"Loading simulation state variables from {self.simulation_state_file}...")

            def try_load_state_file(filepath, file_type="simulation"):
                """
                Attempt to load a state file, trying backups if the primary fails.
                Backups are organized by simulation ID and timestamp.
                Format: <filename>.<sim_id>.<timestamp>

                Args:
                    filepath: Path to the primary state file
                    file_type: "simulation" or "validator" for logging

                Returns:
                    Unpacked state dict or None if all attempts fail
                """
                if os.path.exists(filepath):
                    try:
                        bt.logging.info(f"Loading {file_type} state from {filepath}...")
                        with open(filepath, 'rb') as file:
                            byte_data = file.read()
                        use_list = (file_type == "simulation")
                        state = msgpack.unpackb(byte_data, use_list=use_list, strict_map_key=False)
                        bt.logging.success(f"Loaded {file_type} state from primary file")
                        return state
                    except Exception as ex:
                        bt.logging.error(f"Failed to load {file_type} state from {filepath}: {ex}")
                        bt.logging.warning(f"Attempting to load from backups...")
                state_path = Path(filepath)
                all_backups = list(state_path.parent.glob(f"{state_path.name}.*.*"))
                
                if not all_backups:
                    bt.logging.warning(f"No backup files found for {filepath}")
                    return None

                parsed_backups = []
                for backup_file in all_backups:
                    try:
                        parts = backup_file.name.split('.')
                        if len(parts) >= 3:
                            sim_id = parts[-2]
                            timestamp = parts[-1]
                            if '_' in sim_id and timestamp.isdigit():
                                parsed_backups.append({
                                    'path': backup_file,
                                    'sim_id': sim_id,
                                    'timestamp': int(timestamp)
                                })
                    except Exception as ex:
                        bt.logging.debug(f"Skipping malformed backup filename: {backup_file.name}")
                        continue

                if not parsed_backups:
                    bt.logging.warning(f"No valid backup files found for {filepath}")
                    return None

                sorted_backups = sorted(
                    parsed_backups,
                    key=lambda x: (x['sim_id'], x['timestamp']),
                    reverse=True
                )
                bt.logging.info(
                    f"Found {len(sorted_backups)} valid backups across "
                    f"{len(set(b['sim_id'] for b in sorted_backups))} simulation runs"
                )
                bt.logging.info("Trying backups (most recent simulation first)...")

                for backup_info in sorted_backups:
                    backup_file = backup_info['path']
                    try:
                        bt.logging.info(
                            f"Attempting backup: {backup_file.name} "
                            f"(sim_id={backup_info['sim_id']}, ts={backup_info['timestamp']})"
                        )
                        with open(backup_file, 'rb') as file:
                            byte_data = file.read()
                        use_list = (file_type == "simulation")
                        state = msgpack.unpackb(byte_data, use_list=use_list, strict_map_key=False)
                        bt.logging.success(f"Successfully loaded from backup: {backup_file.name}")

                        if os.path.exists(filepath):
                            from datetime import datetime
                            corrupted_backup = f"{filepath}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.corrupted"
                            try:
                                shutil.copy2(filepath, corrupted_backup)
                                bt.logging.info(f"Saved corrupted primary file as: {corrupted_backup}")
                            except Exception as ex:
                                bt.logging.warning(f"Failed to backup corrupted file: {ex}")
                        bt.logging.info(f"Restoring {backup_file.name} as primary state file...")
                        shutil.copy2(backup_file, filepath)
                        bt.logging.success(f"Restored backup to {filepath}")
                        return state
                    except Exception as ex:
                        bt.logging.warning(f"Backup {backup_file.name} failed: {ex}")
                        continue
                bt.logging.error(f"All {len(sorted_backups)} backups failed for {file_type} state")
                return None

            if not self.config.neuron.reset:
                simulation_state = try_load_state_file(self.simulation_state_file, "simulation")

                if simulation_state:
                    self.start_time = simulation_state["start_time"]
                    self.start_timestamp = simulation_state["start_timestamp"]
                    self.step_rates = simulation_state.get("step_rates", [])
                    loaded_pending_notices = simulation_state.get("pending_notices", {})
                    self.pending_notices = {uid: [] for uid in range(self.effective_max_uids)}
                    for uid, notices in loaded_pending_notices.items():
                        if uid < self.effective_max_uids:
                            self.pending_notices[uid] = notices
                        else:
                            bt.logging.debug(f"Skipping pending_notices for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
                    loaded_initial_balances = simulation_state.get("initial_balances", {})
                    self.initial_balances = {
                        uid: {bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                            for bookId in range(self.simulation.book_count)}
                        for uid in range(self.effective_max_uids)
                    }
                    for uid, initial_balances in loaded_initial_balances.items():
                        if uid < self.effective_max_uids:
                            if 'WEALTH' not in initial_balances.get(0, {}):
                                self.initial_balances[uid] = {
                                    bookId: initial_balance | {'WEALTH': self.simulation.miner_wealth}
                                    for bookId, initial_balance in initial_balances.items()
                                }
                            else:
                                self.initial_balances[uid] = initial_balances
                        else:
                            bt.logging.debug(f"Skipping initial_balances for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

                    self.recent_trades = {
                        book_id: [TradeInfo.model_construct(**t) for t in book_trades]
                        for book_id, book_trades in simulation_state["recent_trades"].items()
                    }
                    loaded_recent_miner_trades = simulation_state.get("recent_miner_trades", {})
                    self.recent_miner_trades = {
                        uid: {bookId: [] for bookId in range(self.simulation.book_count)}
                        for uid in range(self.effective_max_uids)
                    }
                    if loaded_recent_miner_trades:
                        for uid, uid_miner_trades in loaded_recent_miner_trades.items():
                            if uid < self.effective_max_uids:
                                self.recent_miner_trades[uid] = {
                                    book_id: [[TradeEvent.model_construct(**t), r] for t, r in trades]
                                    for book_id, trades in uid_miner_trades.items()
                                }
                            else:
                                bt.logging.debug(f"Skipping recent_miner_trades for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
                    
                    self.simulation.logDir = simulation_state["simulation.logDir"]
                    bt.logging.success(f"Loaded simulation state")
                else:
                    bt.logging.warning("All simulation state files corrupted, initializing fresh state")
                    simulation_state = None
            else:
                simulation_state = None

            if simulation_state is None:
                if self.config.neuron.reset:
                    bt.logging.warning(f"`neuron.reset is True, ignoring previous state")
                else:
                    bt.logging.info(f"No valid simulation state found, initializing new state")

                self.pending_notices = {uid: [] for uid in range(self.effective_max_uids)}
                self.initial_balances = {
                    uid: {bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                        for bookId in range(self.simulation.book_count)}
                    for uid in range(self.effective_max_uids)
                }
                self.recent_trades = {bookId: [] for bookId in range(self.simulation.book_count)}
                self.recent_miner_trades = {
                    uid: {bookId: [] for bookId in range(self.simulation.book_count)}
                    for uid in range(self.effective_max_uids)
                }
                self.fundamental_price = {bookId: None for bookId in range(self.simulation.book_count)}

            if os.path.exists(self.validator_state_file.replace('.mp', '.pt')):
                bt.logging.info("Pytorch validator state file exists - converting to msgpack...")
                pt_validator_state = torch.load(self.validator_state_file.replace('.mp', '.pt'), weights_only=False)
                pt_validator_state["scores"] = [score.item() for score in pt_validator_state['scores']]
                with open(self.validator_state_file, 'wb') as file:
                    packed_data = msgpack.packb(
                        pt_validator_state, use_bin_type=True
                    )
                    file.write(packed_data)
                os.rename(self.validator_state_file.replace('.mp', '.pt'), self.validator_state_file.replace('.mp', '.pt') + ".bak")
                bt.logging.info(f"Pytorch validator state file converted to msgpack at {self.validator_state_file}")

            if not self.config.neuron.reset:
                validator_state = try_load_state_file(self.validator_state_file, "validator")
                if validator_state:
                    bt.logging.info(f"Populating validator data...")
                    self.step = validator_state["step"]
                    self.simulation_timestamp = validator_state.get("simulation_timestamp", 0)
                    self.hotkeys = validator_state["hotkeys"]
                    self.deregistered_uids = list(validator_state.get("deregistered_uids", []))
                    
                    loaded_scores = validator_state["scores"]
                    self.scores = torch.zeros(self.effective_max_uids, dtype=torch.float32, device=self.device)
                    num_scores_to_copy = min(len(loaded_scores), self.effective_max_uids)
                    self.scores[:num_scores_to_copy] = torch.tensor(loaded_scores[:num_scores_to_copy])

                    if len(loaded_scores) > self.effective_max_uids:
                        bt.logging.warning(
                            f"Loaded state has {len(loaded_scores)} scores but current effective_max_uids is {self.effective_max_uids}. "
                            f"Truncating scores for UIDs {self.effective_max_uids} onwards."
                        )
                    elif len(loaded_scores) < self.effective_max_uids:
                        bt.logging.info(
                            f"Loaded state has {len(loaded_scores)} scores but current effective_max_uids is {self.effective_max_uids}. "
                            f"Initializing scores for UIDs {len(loaded_scores)} onwards as 0.0."
                        )

                    loaded_gentrx = validator_state.get("gentrx_scores")
                    self.gentrx_scores = torch.zeros(self.effective_max_uids, dtype=torch.float32, device=self.device)
                    if loaded_gentrx is not None:
                        num_g_to_copy = min(len(loaded_gentrx), self.effective_max_uids)
                        self.gentrx_scores[:num_g_to_copy] = torch.tensor(loaded_gentrx[:num_g_to_copy])

                    loaded_activity = validator_state.get("activity_factors", {})
                    if loaded_activity and isinstance(list(loaded_activity.values())[0], float):
                        loaded_activity = {
                            uid: {bookId: loaded_activity[uid] for bookId in range(self.simulation.book_count)}
                            for uid in loaded_activity
                        }
                    
                    self.activity_factors = {
                        uid: {bookId: 0.0 for bookId in range(self.simulation.book_count)}
                        for uid in range(self.effective_max_uids)
                    }
                    for uid, books in loaded_activity.items():
                        if uid < self.effective_max_uids:
                            self.activity_factors[uid] = books
                        else:
                            bt.logging.debug(f"Skipping activity_factors for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
                    
                    loaded_pnl = validator_state.get("pnl_factors", {})
                    if loaded_pnl and isinstance(list(loaded_pnl.values())[0], float):
                        loaded_pnl = {
                            uid: {bookId: loaded_pnl[uid] for bookId in range(self.simulation.book_count)}
                            for uid in loaded_pnl
                        }

                    self.pnl_factors = {
                        uid: {bookId: 1.0 for bookId in range(self.simulation.book_count)}
                        for uid in range(self.effective_max_uids)
                    }
                    for uid, books in loaded_pnl.items():
                        if uid < self.effective_max_uids:
                            self.pnl_factors[uid] = books
                        else:
                            bt.logging.debug(f"Skipping pnl_factors for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

                    loaded_inventory = validator_state.get("inventory_history", {})
                    self.inventory_history = {uid: {} for uid in range(self.effective_max_uids)}
                    for uid, hist in loaded_inventory.items():
                        if uid < self.effective_max_uids:
                            self.inventory_history[uid] = hist
                        else:
                            bt.logging.debug(f"Skipping inventory_history for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

                    bt.logging.info(f"Processing inventory history...")
                    for uid in self.inventory_history:
                        for timestamp in self.inventory_history[uid]:
                            if len(self.inventory_history[uid][timestamp]) < self.simulation.book_count:
                                for bookId in range(len(self.inventory_history[uid][timestamp]), self.simulation.book_count):
                                    self.inventory_history[uid][timestamp][bookId] = 0.0
                            if len(self.inventory_history[uid][timestamp]) > self.simulation.book_count:
                                self.inventory_history[uid][timestamp] = {
                                    k: v for k, v in self.inventory_history[uid][timestamp].items()
                                    if k < self.simulation.book_count
                                }

                    bt.logging.info(f"Processing kappa values...")
                    if 'kappa_values' in validator_state:
                        loaded_kappa = validator_state['kappa_values']
                        self.kappa_values = {
                            uid: {
                                'books': {bookId: None for bookId in range(self.simulation.book_count)},
                                'books_weighted': {bookId: 0.0 for bookId in range(self.simulation.book_count)},
                                'total': None,
                                'average': None,
                                'median': None,
                                'normalized_average': 0.0,
                                'normalized_median': 0.0,
                                'normalized_total': 0.0,
                                'activity_weighted_normalized_median': 0.0,
                                'penalty': 0.0,
                                'score': 0.0,
                            }
                            for uid in range(self.effective_max_uids)
                        }
                        for uid, kappa_data in loaded_kappa.items():
                            if uid < self.effective_max_uids and kappa_data is not None:
                                kappa = kappa_data.copy()
                                if 'books' in kappa:
                                    kappa['books'] = {
                                        book_id: val for book_id, val in kappa['books'].items()
                                        if book_id < self.simulation.book_count
                                    }
                                    for book_id in range(len(kappa['books']), self.simulation.book_count):
                                        kappa['books'][book_id] = None
                                if 'books_weighted' in kappa:
                                    kappa['books_weighted'] = {
                                        book_id: val for book_id, val in kappa['books_weighted'].items()
                                        if book_id < self.simulation.book_count
                                    }
                                    for book_id in range(len(kappa['books_weighted']), self.simulation.book_count):
                                        kappa['books_weighted'][book_id] = 0.0
                                self.kappa_values[uid] = kappa
                            elif uid >= self.effective_max_uids:
                                bt.logging.debug(f"Skipping kappa_values for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
                    elif 'sharpe_values' in validator_state:
                        bt.logging.info("Converting sharpe_values to kappa_values format...")
                        self.kappa_values = {
                            uid: {
                                'books': {bookId: None for bookId in range(self.simulation.book_count)},
                                'books_weighted': {bookId: 0.0 for bookId in range(self.simulation.book_count)},
                                'total': None,
                                'average': None,
                                'median': None,
                                'normalized_average': 0.0,
                                'normalized_median': 0.0,
                                'normalized_total': 0.0,
                                'activity_weighted_normalized_median': 0.0,
                                'penalty': 0.0,
                                'score': 0.0,
                            }
                            for uid in range(self.effective_max_uids)
                        }
                        for uid, sharpe_data in validator_state['sharpe_values'].items():
                            if uid < self.effective_max_uids and sharpe_data:
                                self.kappa_values[uid] = {
                                    'books': sharpe_data.get('books_realized', {bookId: None for bookId in range(self.simulation.book_count)}),
                                    'books_weighted': sharpe_data.get('books_weighted_realized', {bookId: 0.0 for bookId in range(self.simulation.book_count)}),
                                    'total': sharpe_data.get('total_realized'),
                                    'average': sharpe_data.get('average_realized'),
                                    'median': sharpe_data.get('median_realized'),
                                    'normalized_average': sharpe_data.get('normalized_average_realized', 0.0),
                                    'normalized_median': sharpe_data.get('normalized_median_realized', 0.0),
                                    'normalized_total': sharpe_data.get('normalized_total_realized', 0.0),
                                    'activity_weighted_normalized_median': sharpe_data.get('activity_weighted_normalized_median_realized', 0.0),
                                    'penalty': sharpe_data.get('penalty_realized', 0.0),
                                    'score': sharpe_data.get('score_realized', 0.0),
                                }
                            elif uid >= self.effective_max_uids:
                                bt.logging.debug(f"Skipping sharpe_values conversion for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
                    else:
                        self.kappa_values = {
                            uid: {
                                'books': {bookId: None for bookId in range(self.simulation.book_count)},
                                'books_weighted': {bookId: 0.0 for bookId in range(self.simulation.book_count)},
                                'total': None,
                                'average': None,
                                'median': None,
                                'normalized_average': 0.0,
                                'normalized_median': 0.0,
                                'normalized_total': 0.0,
                                'activity_weighted_normalized_median': 0.0,
                                'penalty': 0.0,
                                'score': 0.0,
                            }
                            for uid in range(self.effective_max_uids)
                        }

                    loaded_unnorm = validator_state.get("unnormalized_scores", {})
                    self.unnormalized_scores = {uid: 0.0 for uid in range(self.effective_max_uids)}
                    for uid, score in loaded_unnorm.items():
                        if uid < self.effective_max_uids:
                            self.unnormalized_scores[uid] = score
                        else:
                            bt.logging.debug(f"Skipping unnormalized_scores for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

                    bt.logging.info(f"Processing trade volumes...")
                    loaded_trade_vols = validator_state.get("trade_volumes", {})
                    self.trade_volumes = {
                        uid: {bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                            for bookId in range(self.simulation.book_count)}
                        for uid in range(self.effective_max_uids)
                    }
                    for uid, books in loaded_trade_vols.items():
                        if uid < self.effective_max_uids:
                            self.trade_volumes[uid] = books
                        else:
                            bt.logging.debug(f"Skipping trade_volumes for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

                    bt.logging.info(f"Processing trade volume histories...")
                    reorg = False
                    for uid in range(self.effective_max_uids):
                        if uid not in self.trade_volumes:
                            self.trade_volumes[uid] = {
                                bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                                for bookId in range(self.simulation.book_count)
                            }
                            
                        for bookId in self.trade_volumes[uid]:
                            if 'total' not in self.trade_volumes[uid][bookId]:
                                if not reorg:
                                    bt.logging.info(f"Optimizing miner volume history structures...")
                                    reorg = True
                                volumes = {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                                for time, role_volume in self.trade_volumes[uid][bookId].items():
                                    sampled_time = math.ceil(time / self.config.scoring.activity.trade_volume_sampling_interval) * self.config.scoring.activity.trade_volume_sampling_interval
                                    for role, volume in role_volume.items():
                                        if sampled_time not in volumes[role]:
                                            volumes[role][sampled_time] = 0.0
                                        volumes[role][sampled_time] += volume
                                self.trade_volumes[uid][bookId] = {
                                    role: {time: round(volumes[role][time], self.simulation.volumeDecimals) for time in volumes[role]}
                                    for role in volumes
                                }
                        
                        if len(self.trade_volumes[uid]) < self.simulation.book_count:
                            for bookId in range(len(self.trade_volumes[uid]), self.simulation.book_count):
                                self.trade_volumes[uid][bookId] = {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                        if len(self.trade_volumes[uid]) > self.simulation.book_count:
                            self.trade_volumes[uid] = {
                                k: v for k, v in self.trade_volumes[uid].items()
                                if k < self.simulation.book_count
                            }
                        
                        if uid not in self.activity_factors:
                            self.activity_factors[uid] = {bookId: 0.0 for bookId in range(self.simulation.book_count)}
                            
                        if len(self.activity_factors[uid]) < self.simulation.book_count:
                            for bookId in range(len(self.activity_factors[uid]), self.simulation.book_count):
                                self.activity_factors[uid][bookId] = 0.0
                        if len(self.activity_factors[uid]) > self.simulation.book_count:
                            self.activity_factors[uid] = {
                                k: v for k, v in self.activity_factors[uid].items()
                                if k < self.simulation.book_count
                            }

                        if uid not in self.pnl_factors:
                            self.pnl_factors[uid] = {bookId: 1.0 for bookId in range(self.simulation.book_count)}
                        
                        if len(self.pnl_factors[uid]) < self.simulation.book_count:
                            for bookId in range(len(self.pnl_factors[uid]), self.simulation.book_count):
                                self.pnl_factors[uid][bookId] = 1.0
                        if len(self.pnl_factors[uid]) > self.simulation.book_count:
                            self.pnl_factors[uid] = {
                                k: v for k, v in self.pnl_factors[uid].items()
                                if k < self.simulation.book_count
                            }

                    def load_volume_sums(data, name, book_count):
                        """Load volume sums and prune books that exceed book_count."""
                        result = defaultdict(lambda: defaultdict(float))

                        if name not in data:
                            bt.logging.info(f"No {name} in saved state, initializing empty")
                            return result

                        volume_data = data[name]

                        if volume_data:
                            first_key = next(iter(volume_data.keys()))

                            if isinstance(first_key, (tuple, list)) and len(first_key) == 2:
                                bt.logging.info(f"Converting {name} from old tuple-key format to nested dict...")
                                for key, vol in volume_data.items():
                                    uid, book_id = key
                                    if book_id < book_count and uid < self.effective_max_uids:
                                        result[uid][book_id] = vol
                                bt.logging.debug(f"Converted {len(volume_data)} entries in {name}")

                            elif isinstance(first_key, int):
                                first_value = volume_data[first_key]

                                if isinstance(first_value, dict):
                                    bt.logging.debug(f"Loading {name} in nested dict format...")
                                    for uid, books in volume_data.items():
                                        if uid < self.effective_max_uids:
                                            for book_id, vol in books.items():
                                                if book_id < book_count:
                                                    result[uid][book_id] = vol
                                        else:
                                            bt.logging.debug(f"Skipping {name} for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
                                else:
                                    bt.logging.warning(f"Unexpected format for {name}: single-level dict")
                                    if 0 < book_count and first_key < self.effective_max_uids:
                                        result[first_key][0] = first_value
                            else:
                                bt.logging.warning(f"Unknown format for {name}, initializing empty")

                        return result

                    bt.logging.info(f"Processing volume sums...")
                    self.volume_sums = load_volume_sums(validator_state, "volume_sums", self.simulation.book_count)
                    self.maker_volume_sums = load_volume_sums(validator_state, "maker_volume_sums", self.simulation.book_count)
                    self.taker_volume_sums = load_volume_sums(validator_state, "taker_volume_sums", self.simulation.book_count)
                    self.self_volume_sums = load_volume_sums(validator_state, "self_volume_sums", self.simulation.book_count)
                    self.roundtrip_volume_sums = load_volume_sums(validator_state, "roundtrip_volume_sums", self.simulation.book_count)

                    bt.logging.info(f"Processing roundtrip volumes...")
                    self.roundtrip_volumes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
                    if "roundtrip_volumes" in validator_state:
                        for uid, books in validator_state["roundtrip_volumes"].items():
                            if uid < self.effective_max_uids:
                                for book_id, volumes in books.items():
                                    if book_id >= self.simulation.book_count:
                                        continue
                                    for timestamp, volume in volumes.items():
                                        self.roundtrip_volumes[uid][book_id][timestamp] = volume
                            else:
                                bt.logging.debug(f"Skipping roundtrip_volumes for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

                    bt.logging.info(f"Processing realized PnL history...")
                    self.realized_pnl_history = defaultdict(lambda: defaultdict(dict))
                    if "realized_pnl_history" in validator_state:
                        for uid, hist in validator_state["realized_pnl_history"].items():
                            if uid < self.effective_max_uids:
                                for timestamp, books in hist.items():
                                    ts_pnl = {
                                        book_id: round(pnl, self.simulation.volumeDecimals)
                                        for book_id, pnl in books.items()
                                        if book_id < self.simulation.book_count and round(pnl, self.simulation.volumeDecimals) != 0.0
                                    }
                                    self.realized_pnl_history[uid][timestamp] = ts_pnl
                            else:
                                bt.logging.debug(f"Skipping realized_pnl_history for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")

                    bt.logging.info(f"Processing open positions...")
                    self.open_positions = defaultdict(lambda: defaultdict(lambda: {
                        'longs': deque(),
                        'shorts': deque()
                    }))
                    if "open_positions" in validator_state:
                        legacy_count = 0
                        for uid, books in validator_state["open_positions"].items():
                            if uid < self.effective_max_uids:
                                for book_id, pos in books.items():
                                    if book_id >= self.simulation.book_count:
                                        continue
                                    longs = []
                                    for p in pos['longs']:
                                        if len(p) == 4:
                                            longs.append(tuple(p))
                                        elif len(p) == 3:
                                            longs.append((*p, 0.0))
                                            legacy_count += 1
                                    shorts = []
                                    for p in pos['shorts']:
                                        if len(p) == 4:
                                            shorts.append(tuple(p))
                                        elif len(p) == 3:
                                            shorts.append((*p, 0.0))
                                            legacy_count += 1
                                    self.open_positions[uid][book_id]['longs'] = deque(longs)
                                    self.open_positions[uid][book_id]['shorts'] = deque(shorts)
                            else:
                                bt.logging.debug(f"Skipping open_positions for UID {uid} (exceeds effective_max_uids={self.effective_max_uids})")
                        if legacy_count > 0:
                            bt.logging.info(f"Converted {legacy_count} legacy positions to new format")

                    current_sampling_interval = self.config.scoring.activity.trade_volume_sampling_interval
                    detected_old_interval = None                    
                    bt.logging.info(f"Checking for sampling interval changes (current config: {current_sampling_interval}ns)...")
                    for uid in self.trade_volumes:
                        if detected_old_interval is not None:
                            break
                        for book_id in self.trade_volumes[uid]:
                            if detected_old_interval is not None:
                                break
                            for role in ['total', 'maker', 'taker', 'self']:
                                timestamps = sorted(self.trade_volumes[uid][book_id][role].keys())
                                if len(timestamps) >= 3:
                                    detected_old_interval = timestamps[2] - timestamps[1]
                                    bt.logging.info(
                                        f"Detected old sampling interval from trade_volumes[{uid}][{book_id}][{role}]: "
                                        f"{detected_old_interval}ns"
                                    )
                                    break
                    if detected_old_interval is None:
                        for uid in self.roundtrip_volumes:
                            if detected_old_interval is not None:
                                break
                            for book_id in self.roundtrip_volumes[uid]:
                                timestamps = sorted(self.roundtrip_volumes[uid][book_id].keys())
                                if len(timestamps) >= 2:
                                    detected_old_interval = timestamps[1] - timestamps[0]
                                    bt.logging.info(
                                        f"Detected old sampling interval from roundtrip_volumes[{uid}][{book_id}]: "
                                        f"{detected_old_interval}ns"
                                    )
                                    break
                    if detected_old_interval is not None and detected_old_interval != current_sampling_interval:
                        bt.logging.warning(
                            f"Sampling interval mismatch detected! "
                            f"Old: {detected_old_interval}ns ({detected_old_interval / 1e9:.0f}s), "
                            f"New: {current_sampling_interval}ns ({current_sampling_interval / 1e9:.0f}s)"
                        )
                        self.migrate_sampling_interval(detected_old_interval, current_sampling_interval)
                    elif detected_old_interval is None:
                        bt.logging.info("No existing volume data found, no migration needed")
                    else:
                        bt.logging.info(f"Sampling interval unchanged ({current_sampling_interval}ns), no migration needed")

                    bt.logging.info("Downsampling inventory_history to minimal storage...")
                    downsampled_count = 0
                    for uid in self.inventory_history:
                        hist = self.inventory_history[uid]
                        if not hist:
                            continue
                        if len(hist) <= 3:
                            continue
                        timestamps = sorted(hist.keys())
                        minimal_hist = {
                            timestamps[0]: hist[timestamps[0]],
                            timestamps[-2]: hist[timestamps[-2]],
                            timestamps[-1]: hist[timestamps[-1]]
                        }
                        old_size = len(hist)
                        self.inventory_history[uid] = minimal_hist
                        downsampled_count += (old_size - 3)
                    bt.logging.info(
                        f"Downsampled inventory_history: removed {downsampled_count} entries"
                    )

                    bt.logging.success(f"Loaded validator state for {self.effective_max_uids} UIDs")
                else:
                    bt.logging.warning("All validator state files corrupted, initializing fresh state")
                    validator_state = None
            else:
                validator_state = None

            if validator_state is None:
                if self.config.neuron.reset:
                    bt.logging.warning(f"`neuron.reset is True, ignoring previous validator state")
                else:
                    bt.logging.info(f"No valid validator state found, initializing new state")

                self.activity_factors = {
                    uid: {bookId: 0.0 for bookId in range(self.simulation.book_count)}
                    for uid in range(self.effective_max_uids)
                }
                self.pnl_factors = {
                    uid: {bookId: 1.0 for bookId in range(self.simulation.book_count)}
                    for uid in range(self.effective_max_uids)
                }
                self.inventory_history = {uid: {} for uid in range(self.effective_max_uids)}
                self.kappa_values = {
                    uid: {
                        'books': {bookId: None for bookId in range(self.simulation.book_count)},
                        'books_weighted': {bookId: 0.0 for bookId in range(self.simulation.book_count)},
                        'total': None,
                        'average': None,
                        'median': None,
                        'normalized_average': 0.0,
                        'normalized_median': 0.0,
                        'normalized_total': 0.0,
                        'activity_weighted_normalized_median': 0.0,
                        'penalty': 0.0,
                        'score': 0.0,
                    }
                    for uid in range(self.effective_max_uids)
                }
                self.unnormalized_scores = {uid: 0.0 for uid in range(self.effective_max_uids)}
                self.trade_volumes = {
                    uid: {bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                        for bookId in range(self.simulation.book_count)}
                    for uid in range(self.effective_max_uids)
                }
                self.volume_sums = defaultdict(lambda: defaultdict(float))
                self.maker_volume_sums = defaultdict(lambda: defaultdict(float))
                self.taker_volume_sums = defaultdict(lambda: defaultdict(float))
                self.self_volume_sums = defaultdict(lambda: defaultdict(float))
                self.roundtrip_volumes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
                self.roundtrip_volume_sums = defaultdict(lambda: defaultdict(float))
                self.realized_pnl_history = defaultdict(lambda: defaultdict(dict))
                self.open_positions = defaultdict(lambda: defaultdict(lambda: {
                    'longs': deque(),
                    'shorts': deque()
                }))

        def get_n_target_miners(self) -> int:
            """
            Count registered miner UIDs in the metagraph (validators excluded).

            Used by `prepare_weights` to size the GenTRX pool's participation
            denominator. A miner is identified by having a reachable axon
            endpoint and not being the running validator itself.
            """
            try:
                my_hotkey = self.wallet.hotkey.ss58_address
            except Exception:
                my_hotkey = None
            count = 0
            for uid in range(self.metagraph.n):
                axon = self.metagraph.axons[uid]
                if my_hotkey is not None and axon.hotkey == my_hotkey:
                    continue
                if axon.ip == "0.0.0.0" or axon.port == 0:
                    continue
                count += 1
            return count

        def handle_deregistration(self, uid) -> None:
            """
            Handles deregistration of a validator or miner UID.

            Behavior:
                - Flags the UID for balance/state reset.
                - Zeros current score.
                - Logs deregistration action.

            Args:
                uid (int): UID being deregistered.

            Returns:
                None
            """
            self.deregistered_uids.append(uid)
            self.scores[uid] = 0.0
            if hasattr(self, 'gentrx_scores'):
                self.gentrx_scores[uid] = 0.0
            # Reset per-round GenTRX EMA so new miner at the same UID slot
            # doesn't inherit the old miner's score history.
            if hasattr(self, '_gentrx_ema') and self._gentrx_ema:
                self._gentrx_ema.pop(uid, None)
            bt.logging.debug(f"UID {uid} Deregistered - Scheduled for reset.")

        def process_resets(self, state : MarketSimulationStateUpdate) -> None:
            """
            Processes reset notices delivered by the simulator.

            Behavior:
                - Detects successful agent reset events (RDRA / ERDRA).
                - Resets Kappa values, activity factors, volume histories, inventory,
                and all accumulated metrics for each affected UID.
                - Removes UID from deregistration list after reset.
                - Restores the UID to a clean initial state.
                - Issues a PagerDuty alert if reset fails.

            Args:
                state (MarketSimulationStateUpdate): Contains notices and reset messages.

            Returns:
                None
            """
            for notice in state.notices[self.uid]:
                if notice['y'] in ["RESPONSE_DISTRIBUTED_RESET_AGENT", "RDRA"] or notice['y'] in ["ERROR_RESPONSE_DISTRIBUTED_RESET_AGENT", "ERDRA"]:
                    for reset in notice['r']:
                        if reset['u']:
                            bt.logging.info(f"Agent {reset['a']} Balances Reset! {reset}")
                            if reset['a'] in self.deregistered_uids:
                                self.kappa_values[reset['a']] = {
                                    'books': {
                                        bookId: None for bookId in range(self.simulation.book_count)
                                    },
                                    'books_weighted': {
                                        bookId: 0.0 for bookId in range(self.simulation.book_count)
                                    },
                                    'total': None,
                                    'average': None,
                                    'median': None,
                                    'normalized_average': 0.0,
                                    'normalized_median': 0.0,
                                    'normalized_total': 0.0,
                                    'activity_weighted_normalized_median': 0.0,
                                    'penalty': 0.0,
                                    'score': 0.0,
                                }
                                self.unnormalized_scores[reset['a']] = 0.0
                                self.activity_factors[reset['a']] = {bookId: 0.0 for bookId in range(self.simulation.book_count)}
                                self.pnl_factors[reset['a']] = {bookId: 1.0 for bookId in range(self.simulation.book_count)}
                                self.inventory_history[reset['a']] = {}
                                self.trade_volumes[reset['a']] = {bookId: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}} for bookId in range(self.simulation.book_count)}

                                for book_id in range(self.simulation.book_count):
                                    self.volume_sums[reset['a']][book_id] = 0.0
                                    self.maker_volume_sums[reset['a']][book_id] = 0.0
                                    self.taker_volume_sums[reset['a']][book_id] = 0.0
                                    self.self_volume_sums[reset['a']][book_id] = 0.0
                                self.roundtrip_volumes[reset['a']] = defaultdict(lambda: defaultdict(float))
                                for book_id in range(self.simulation.book_count):
                                    self.roundtrip_volume_sums[reset['a']][book_id] = 0.0

                                self.realized_pnl_history[reset['a']] = {}
                                self.open_positions[reset['a']] = defaultdict(lambda: {
                                    'longs': deque(),
                                    'shorts': deque()
                                })

                                self.initial_balances[reset['a']] = {bookId: {'BASE': None, 'QUOTE': None, 'WEALTH': None} for bookId in range(self.simulation.book_count)}
                                self.initial_balances_published[reset['a']] = False
                                self.deregistered_uids.remove(reset['a'])
                                self.miner_stats[reset['a']] = {'requests': 0, 'timeouts': 0, 'failures': 0, 'rejections': 0, 'call_time': []}
                                self.recent_miner_trades[reset['a']] = {bookId: [] for bookId in range(self.simulation.book_count)}
                        else:
                            self.pagerduty_alert(f"Failed to Reset Agent {reset['a']} : {reset['m']}")\
                                
        def resync_metagraph(self):
            """
            Resyncs the metagraph and updates the hotkeys and scores based on the new metagraph.
            Extended to handle benchmark agents (UIDs >= subnet_info.max_uids).
            """
            bt.logging.trace("resync_metagraph()")
            previous_metagraph = copy.deepcopy(self.metagraph)
            bt.logging.debug("Syncing metagraph...")
            self.metagraph.sync(subtensor=self.subtensor)
            if previous_metagraph.axons == self.metagraph.axons and len(self.hotkeys) == len(self.metagraph.hotkeys):            
                bt.logging.debug("No axon changes!")
                return

            bt.logging.info(
                "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
            )
            for uid, hotkey in enumerate(self.hotkeys):
                if uid < len(self.metagraph.hotkeys) and hotkey != self.metagraph.hotkeys[uid]:
                    self.handle_deregistration(uid)

            old_metagraph_size = len(self.hotkeys)
            new_metagraph_size = len(self.metagraph.hotkeys)
            
            if old_metagraph_size != new_metagraph_size:
                bt.logging.info(f"Metagraph size changed: {old_metagraph_size} -> {new_metagraph_size}")
                old_effective_max_uids = self.effective_max_uids
                self.benchmark_start_uid = new_metagraph_size
                self.effective_max_uids = new_metagraph_size + len(self.benchmark_agents)
                if old_effective_max_uids != self.effective_max_uids:
                    bt.logging.info(
                        f"Resizing scores tensor: {old_effective_max_uids} -> {self.effective_max_uids} "
                        f"(network: {new_metagraph_size}, benchmarks: {len(self.benchmark_agents)})"
                    )
                    new_scores = torch.zeros(self.effective_max_uids, dtype=torch.float32, device=self.device)
                    min_len = min(old_effective_max_uids, self.effective_max_uids)
                    new_scores[:min_len] = self.scores[:min_len]
                    self.scores = new_scores

                    new_gentrx = torch.zeros(self.effective_max_uids, dtype=torch.float32, device=self.device)
                    min_len_g = min(old_effective_max_uids, self.effective_max_uids)
                    new_gentrx[:min_len_g] = self.gentrx_scores[:min_len_g]
                    self.gentrx_scores = new_gentrx
                    if self.benchmark_agents:
                        bt.logging.info("Updating benchmark agent UIDs...")
                        for idx, agent in enumerate(self.benchmark_agents):
                            old_uid = agent['uid']
                            new_uid = self.benchmark_start_uid + idx
                            if old_uid != new_uid:
                                bt.logging.info(f"Benchmark agent {agent['name']}: UID {old_uid} -> {new_uid}")
                                agent['uid'] = new_uid
                                for data_dict in [
                                    self.miner_stats,
                                    self.initial_balances,
                                    self.activity_factors,
                                    self.pnl_factors,
                                    self.kappa_values,
                                    self.unnormalized_scores,
                                    self.inventory_history,
                                    self.trade_volumes,
                                    self.realized_pnl_history,
                                    self.open_positions,
                                ]:
                                    if old_uid in data_dict and new_uid != old_uid:
                                        data_dict[new_uid] = data_dict.pop(old_uid)
                                for volume_dict in [
                                    self.volume_sums,
                                    self.maker_volume_sums,
                                    self.taker_volume_sums,
                                    self.self_volume_sums,
                                    self.roundtrip_volume_sums,
                                ]:
                                    if old_uid in volume_dict:
                                        volume_dict[new_uid] = volume_dict.pop(old_uid)
                                if old_uid in self.roundtrip_volumes:
                                    self.roundtrip_volumes[new_uid] = self.roundtrip_volumes.pop(old_uid)
                                if old_uid < len(self.scores):
                                    self.scores[new_uid] = self.scores[old_uid]
                                    self.scores[old_uid] = 0.0
                                if old_uid < len(self.gentrx_scores):
                                    self.gentrx_scores[new_uid] = self.gentrx_scores[old_uid]
                                    self.gentrx_scores[old_uid] = 0.0

                # Expand per-UID dicts to cover any UIDs not yet present (new network
                # registrations, or holes left after benchmark shifting above).
                _bc = self.simulation.book_count
                _kappa_default = {
                    'books': {b: None for b in range(_bc)},
                    'books_weighted': {b: 0.0 for b in range(_bc)},
                    'total': None, 'average': None, 'median': None,
                    'normalized_average': 0.0, 'normalized_median': 0.0,
                    'normalized_total': 0.0,
                    'activity_weighted_normalized_median': 0.0,
                    'penalty': 0.0, 'score': 0.0,
                }
                for _uid in range(self.effective_max_uids):
                    if _uid not in self.miner_stats:
                        self.miner_stats[_uid] = {
                            'requests': 0, 'timeouts': 0, 'failures': 0,
                            'rejections': 0, 'call_time': [],
                        }
                    if _uid not in self.activity_factors:
                        self.activity_factors[_uid] = {b: 0.0 for b in range(_bc)}
                    if _uid not in self.pnl_factors:
                        self.pnl_factors[_uid] = {b: 1.0 for b in range(_bc)}
                    if _uid not in self.kappa_values:
                        self.kappa_values[_uid] = dict(_kappa_default)
                        self.kappa_values[_uid]['books'] = {b: None for b in range(_bc)}
                        self.kappa_values[_uid]['books_weighted'] = {b: 0.0 for b in range(_bc)}
                    if _uid not in self.unnormalized_scores:
                        self.unnormalized_scores[_uid] = 0.0
                    if _uid not in self.initial_balances:
                        self.initial_balances[_uid] = {
                            b: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                            for b in range(_bc)
                        }
                    if _uid not in self.initial_balances_published:
                        self.initial_balances_published[_uid] = False
                    if _uid not in self.inventory_history:
                        self.inventory_history[_uid] = {}
                    if _uid not in self.recent_miner_trades:
                        self.recent_miner_trades[_uid] = {b: [] for b in range(_bc)}

            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
            bt.logging.success(
                f"Metagraph resync complete: {len(self.hotkeys)} network hotkeys, "
                f"{len(self.benchmark_agents)} benchmark agents, "
                f"effective_max_uids={self.effective_max_uids}"
            )

        async def _maintain(self) -> None:
            """
            Executes metagraph sync and maintenance operations asynchronously.

            Actions:
                - Marks the validator as in maintenance mode.
                - Runs synchronous maintenance work in an executor thread.
                - Logs timing and reports issues via PagerDuty.

            Returns:
                None
            """
            try:
                self.maintaining = True
                bt.logging.info(f"Synchronizing at Step {self.step}...")
                start = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.maintenance_executor,
                    self._sync_and_check
                )
                bt.logging.info(f"Synchronized ({time.time()-start:.4f}s)")

            except Exception as ex:
                self.pagerduty_alert(f"Failed to sync: {ex}", details={"trace": traceback.format_exc()})
            finally:
                self.maintaining = False

        async def _deliver_gentrx_assignments(self, assignments: dict) -> None:
            """Deliver GenTRX assignments to miners via dendrite."""
            try:
                from taos.im.protocol.gentrx import GenTRXAssignment

                my_hotkey = self.wallet.hotkey.ss58_address
                try:
                    my_uid = self.metagraph.hotkeys.index(my_hotkey)
                except ValueError:
                    my_uid = -1
                deliveries = []
                for uid, assignment in assignments.items():
                    if uid >= len(self.metagraph.axons):
                        # Benchmark miner — look up axon from config
                        bm_idx = uid - self.benchmark_start_uid
                        if bm_idx < 0 or bm_idx >= len(self.benchmark_agents):
                            continue
                        axon = self.benchmark_agents[bm_idx]['axon']
                    else:
                        axon = self.metagraph.axons[uid]
                    if axon.hotkey == my_hotkey:
                        continue  # skip self
                    if axon.ip == "0.0.0.0" or axon.port == 0:
                        continue  # skip dead/unserved nodes
                    try:
                        deliveries.append((
                            uid,
                            axon,
                            GenTRXAssignment(
                                round=assignment.get("round", 0),
                                model_version=assignment.get("model_version", 0),
                                books=assignment.get("books", []),
                                ts_start=assignment.get("ts_start", 0),
                                ts_end=assignment.get("ts_end", 0),
                                data=assignment.get("data", []),
                                data_source=assignment.get("data_source", "s3"),
                                data_endpoint=assignment.get("data_endpoint", ""),
                                data_bucket=assignment.get("data_bucket", ""),
                                data_access_key=assignment.get("data_access_key", ""),
                                data_secret_key=assignment.get("data_secret_key", ""),
                                validator_uid=my_uid,
                            ),
                        ))
                    except Exception as exc:
                        gtx_log.warning(f"build assignment for uid {uid} failed: {exc}")

                if not deliveries:
                    return

                # Single round summary
                round_id = next(iter(assignments.values())).get("round", "?")
                gtx_log.info(f"round={round_id}: delivering to uids={[u for u,_,_ in deliveries]}")

                # Use async-native dendrite call (matches taos/im/validator/query.py pattern).
                # Reuse self.dendrite — creating a new one per call leaks aiohttp sessions
                # and causes "attached to different loop" on subsequent cycles.
                send_ok = 0
                send_fail = 0

                async def _send(uid, axon, synapse):
                    nonlocal send_ok, send_fail
                    t_start = time.time()
                    try:
                        resp = await self.dendrite(
                            axons=axon, synapse=synapse, timeout=5, deserialize=False
                        )
                        t = time.time() - t_start
                        status = getattr(getattr(resp, 'dendrite', None), 'status_code', None)
                        msg = getattr(getattr(resp, 'dendrite', None), 'status_message', '')
                        gtx_log.info(f"round={round_id} uid={uid} {axon.ip}:{axon.port} status={status} msg={msg} t={t:.2f}s")
                        if status == 200:
                            send_ok += 1
                        else:
                            send_fail += 1
                    except Exception as exc:
                        t = time.time() - t_start
                        gtx_log.warning(f"round={round_id} uid={uid} send failed after {t:.2f}s: {exc}")
                        send_fail += 1

                t_deliver_start = time.time()
                await asyncio.gather(*[_send(u, a, s) for u, a, s in deliveries])
                t_deliver = time.time() - t_deliver_start
                gtx_log.info(
                    f"deliver round={round_id} n={len(deliveries)} ok={send_ok} fail={send_fail} t_total={t_deliver:.2f}s"
                )
            except Exception as exc:
                gtx_log.error(f"delivery failed: {exc}")
                import traceback
                gtx_log.error(traceback.format_exc())

        def _sync_and_check(self):
            """
            Performs synchronous metagraph maintenance and simulator health checks.

            Steps:
                - Runs Bittensor sync (without saving state).
                - Verifies simulator health.
                - Restarts simulator if unhealthy.

            Returns:
                None
            """
            self.sync(save_state=False)
            if not check_simulator(self):
                restart_simulator(self)

        def maintain(self) -> None:
            """
            Schedules asynchronous maintenance work from the maintenance thread.

            Behavior:
                - Ensures maintenance is not already running.
                - Triggers only at specific simulation timestamps.
                - Sends a coroutine to the main event loop thread-safely.

            Returns:
                None
            """
            if not self.maintaining and self.last_state and self.last_state.timestamp % self.config.scoring.interval == 2_000_000_000:
                bt.logging.debug(f"[MAINT] Scheduling from thread: {threading.current_thread().name}")
                bt.logging.debug(f"[MAINT] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
                self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._maintain()))

        def _prepare_reporting_data(self):
            bt.logging.debug(f"Retrieving fundamental prices...")
            start = time.time()
            self.load_fundamental()
            bt.logging.debug(f"Retrieved fundamental prices ({time.time()-start:.4f}s).")

            book_count = self.simulation.book_count
            bt.logging.debug("Computing realized P&L totals...")
            pnl_start = time.time()
            total_realized_pnl = {}
            realized_pnl_by_book = {}

            for uid, hist in self.realized_pnl_history.items():
                if not hist:
                    total_realized_pnl[uid] = 0.0
                    realized_pnl_by_book[uid] = {book_id: 0.0 for book_id in range(book_count)}
                    continue

                book_totals = [0.0] * book_count
                for ts, timestamp_data in hist.items():
                    for book_id, pnl in timestamp_data.items():
                        if book_id < len(book_totals):
                            book_totals[book_id] += pnl

                realized_pnl_by_book[uid] = {book_id: book_totals[book_id] for book_id in range(book_count)}
                total_realized_pnl[uid] = sum(book_totals)

            bt.logging.debug(f"Computed realized P&L totals ({time.time()-pnl_start:.4f}s)")

            bt.logging.debug("Building minimal inventory...")
            inv_start = time.time()
            minimal_inventory = {}
            for uid, hist in self.inventory_history.items():
                if not hist:
                    continue
                timestamps = sorted(hist.keys())
                n = len(timestamps)
                if n >= 3:
                    minimal_inventory[uid] = {
                        timestamps[0]: hist[timestamps[0]],
                        timestamps[-2]: hist[timestamps[-2]],
                        timestamps[-1]: hist[timestamps[-1]]
                    }
                elif n > 0:
                    minimal_inventory[uid] = {ts: hist[ts] for ts in timestamps}
            bt.logging.debug(f"Built minimal inventory ({time.time()-inv_start:.4f}s)")

            bt.logging.debug("Serializing volume sums...")
            serialize_start = time.time()

            def serialize_nested_dict(d):
                return {
                    f"{uid}:{book_id}": vol
                    for uid, books in d.items()
                    for book_id, vol in books.items()
                }

            volume_sums_flat = serialize_nested_dict(self.volume_sums)
            maker_volume_sums_flat = serialize_nested_dict(self.maker_volume_sums)
            taker_volume_sums_flat = serialize_nested_dict(self.taker_volume_sums)
            self_volume_sums_flat = serialize_nested_dict(self.self_volume_sums)
            roundtrip_volume_sums_flat = serialize_nested_dict(self.roundtrip_volume_sums)

            bt.logging.debug(f"Serialized volume sums ({time.time()-serialize_start:.4f}s)")

            bt.logging.debug("Building metagraph data...")
            meta_start = time.time()
            metagraph_data = {
                'hotkeys': [str(hk) for hk in self.metagraph.hotkeys],
                'stake': self.metagraph.stake.tolist(),
                'trust': self.metagraph.trust.tolist() if hasattr(self.metagraph, 'trust') else [],
                'consensus': self.metagraph.consensus.tolist(),
                'incentive': self.metagraph.incentive.tolist(),
                'emission': self.metagraph.emission.tolist(),
                'validator_trust': self.metagraph.validator_trust.tolist(),
                'dividends': self.metagraph.dividends.tolist(),
                'active': self.metagraph.active.tolist(),
                'last_update': self.metagraph.last_update.tolist(),
            }
            bt.logging.debug(f"Built metagraph data ({time.time()-meta_start:.4f}s)")

            bt.logging.debug("Building recent trades...")
            trades_start = time.time()
            recent_trades = {
                bookId: [t.model_dump() for t in trades[-25:]]
                for bookId, trades in self.recent_trades.items()
            }

            recent_miner_trades = {
                uid: {
                    bookId: [
                        {'trade': miner_trade.model_dump(), 'role': role}
                        for miner_trade, role in trades[-5:]
                    ]
                    for bookId, trades in book_trades.items()
                }
                for uid, book_trades in self.recent_miner_trades.items()
            }
            bt.logging.debug(f"Built minimal recent trades ({time.time()-trades_start:.4f}s)")

            bt.logging.debug("Building minimal state...")
            state_start = time.time()
            minimal_state = {
                'accounts': self.last_state.accounts,
                'books': {
                    bookId: {
                        'b': book['b'][:5] if book['b'] else [],
                        'a': book['a'][:5] if book['a'] else [],
                        'e': [e for e in book['e'] if e['y'] == 't'][-10:] if book['e'] else [],
                        'mtr': book.get('mtr', 0.0)
                    }
                    for bookId, book in self.last_state.books.items()
                },
                'notices': {
                    uid: [n for n in notices if n['y'] in ['EVENT_TRADE', 'ET']][-10:]
                    for uid, notices in self.last_state.notices.items()
                }
            }
            bt.logging.debug(f"Built minimal state ({time.time()-state_start:.4f}s)")

            bt.logging.debug("Building position summary...")
            pos_start = time.time()
            open_positions = {
                uid: {
                    book_id: {
                        'longs_count': len(pos['longs']),
                        'shorts_count': len(pos['shorts'])
                    }
                    for book_id, pos in books.items()
                }
                for uid, books in self.open_positions.items()
            }
            bt.logging.debug(f"Built position summary ({time.time()-pos_start:.4f}s)")

            bt.logging.debug("Assembling final data structure...")
            final_start = time.time()

            data = {
                'metagraph_data': metagraph_data,
                'simulation': self.simulation.model_dump(),
                'last_state': minimal_state,
                'simulation_timestamp': self.simulation_timestamp,
                'step': self.step,
                'step_rates': list(self.step_rates[-100:]),
                'volume_sums': volume_sums_flat,
                'maker_volume_sums': maker_volume_sums_flat,
                'taker_volume_sums': taker_volume_sums_flat,
                'self_volume_sums': self_volume_sums_flat,
                'roundtrip_volume_sums': roundtrip_volume_sums_flat,
                'inventory_history': minimal_inventory,
                'total_realized_pnl': total_realized_pnl,
                'realized_pnl_by_book': realized_pnl_by_book,
                'book_count': book_count,
                'activity_factors': self.activity_factors,
                'pnl_factors': self.pnl_factors,
                'kappa_values': self.kappa_values,
                'unnormalized_scores': self.unnormalized_scores,
                'scores': {i: score.item() for i, score in enumerate(self.scores)},
                'gentrx_scores': {i: score.item() for i, score in enumerate(self.gentrx_scores)},
                'gentrx_enabled': self._gentrx is not None,
                'gentrx_training': self._gentrx.get_training_stats() if self._gentrx is not None else {},
                'miner_stats': self.miner_stats,
                'initial_balances': self.initial_balances,
                'initial_balances_published': self.initial_balances_published,
                'recent_trades': recent_trades,
                'recent_miner_trades': recent_miner_trades,
                'open_positions': open_positions,
                'fundamental_price': self.fundamental_price,
                'shared_state_rewarding': self.shared_state_rewarding,
                'current_block': self.current_block,
                'uid': self.uid,
                'validator_config' : {
                    'scoring': {
                        'interval': self.config.scoring.interval,
                        'max_instructions_per_book': self.config.scoring.max_instructions_per_book,
                        'min_delay': self.config.scoring.min_delay,
                        'max_delay': self.config.scoring.max_delay,
                        'min_instruction_delay': self.config.scoring.min_instruction_delay,
                        'max_instruction_delay': self.config.scoring.max_instruction_delay,
                        'kappa_weight': self.config.scoring.kappa.weight,
                        'kappa_lookback': self.config.scoring.kappa.lookback,
                        'kappa_min_lookback': self.config.scoring.kappa.min_lookback,
                        'kappa_tau': self.config.scoring.kappa.tau,
                        'kappa_min_realized_observations': self.config.scoring.kappa.min_realized_observations,
                        'kappa_normalization_min': self.config.scoring.kappa.normalization_min,
                        'kappa_normalization_max': self.config.scoring.kappa.normalization_max,
                        'kappa_pnl_impact': self.config.scoring.kappa.pnl.impact,
                        'pnl_weight': self.config.scoring.pnl.weight,
                        'gentrx_simulation_share': getattr(getattr(self.config.scoring, 'gentrx', None), 'simulation_share', 0.0) or 0.0,
                        'pnl_normalization_min_daily_return': self.config.scoring.pnl.min_daily_return,
                        'pnl_normalization_max_daily_return': self.config.scoring.pnl.max_daily_return,
                        'activity_impact': self.config.scoring.activity.impact,
                        'activity_trade_volume_sampling_interval': self.config.scoring.activity.trade_volume_sampling_interval,
                        'activity_trade_volume_assessment_period': self.config.scoring.activity.trade_volume_assessment_period,
                        'activity_decay_grace_period': self.config.scoring.activity.decay_grace_period,
                        'activity_decay_rate': self.config.scoring.activity.decay_rate,
                        'activity_capital_turnover_cap': self.config.scoring.activity.capital_turnover_cap,
                        'activity_max_volume': self.config.scoring.activity.capital_turnover_cap * self.simulation.miner_wealth,
                        'activity_decay_rate': self.config.scoring.activity.decay_rate,
                    }
                }
            }

            bt.logging.debug(f"Assembled final structure ({time.time()-final_start:.4f}s)")

            return data

        async def _report(self):
            if self._reporting:
                bt.logging.warning(f"Previous reporting still in progress, skipping step {self.step}")
                return

            if self.reporting_process.poll() is not None:
                bt.logging.error(f"Reporting service died with exit code {self.reporting_process.returncode}")
                bt.logging.error("Attempting to restart reporting service...")
                self._start_reporting_service()
                if self.reporting_process.poll() is not None:
                    self.pagerduty_alert("Failed to restart reporting service")
                    return

            self._reporting = True
            reporting_step = self.step
            bt.logging.info(f"Starting Reporting at step {reporting_step}...")
            start = time.time()
            try:
                drain_start = time.time()
                drained = 0
                while True:
                    try:
                        await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                self.reporting_ipc_executor,
                                lambda: self.reporting_response_queue.receive(timeout=0.001)
                            ),
                            timeout=0.05
                        )
                        drained += 1
                    except (posix_ipc.BusyError, asyncio.TimeoutError):
                        break

                if drained > 0:
                    bt.logging.warning(f"Drained {drained} stale messages from reporting response queue ({time.time()-drain_start:.4f}s)")

                prep_start = time.time()
                async with self._reward_lock:
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(
                        self.reporting_ipc_executor,
                        self._prepare_reporting_data
                    )
                prep_time = time.time() - prep_start
                bt.logging.info(f"Prepared reporting data ({prep_time:.4f}s)")

                serialize_start = time.time()
                data_bytes = await asyncio.get_event_loop().run_in_executor(
                    self.reporting_ipc_executor,
                    lambda: msgpack.packb(data, use_bin_type=True)
                )
                serialize_time = time.time() - serialize_start
                data_mb = len(data_bytes) / 1024 / 1024
                bt.logging.info(f"Reporting data: {data_mb:.2f} MB (prep={prep_time:.4f}s, serialize={serialize_time:.4f}s)")

                write_start = time.time()
                write_success = await self._write_ipc_nonblocking(
                    self.reporting_request_mem,
                    self.reporting_request_queue,
                    data_bytes,
                    "Reporting",
                    self.reporting_ipc_executor
                )

                if not write_success:
                    self.pagerduty_alert("Failed to write reporting data")
                    return

                bt.logging.info(f"Wrote reporting data ({time.time()-write_start:.4f}s)")

                await asyncio.get_event_loop().run_in_executor(
                    self.reporting_ipc_executor,
                    self.reporting_request_queue.send,
                    b'publish'
                )

                receive_start = time.time()
                max_wait = 120.0
                elapsed = 0
                message = None
                poll_count = 0
                while elapsed < max_wait:
                    poll_count += 1
                    try:
                        message, _ = self.reporting_response_queue.receive(timeout=0.1)
                        bt.logging.info(f"Received reporting response after {poll_count} polls ({time.time()-receive_start:.4f}s)")
                        break
                    except posix_ipc.BusyError:
                        elapsed = time.time() - receive_start
                        if poll_count % 10 == 0:
                            await asyncio.sleep(0.001)
                            bt.logging.debug(f"Still polling for reporting response ({elapsed:.1f}s, {poll_count} polls)")

                        continue
                else:
                    self.pagerduty_alert(f"Reporting response timeout after {max_wait}s")
                    return
                if message is None:
                    self.pagerduty_alert(f"Reporting response timeout after {max_wait}s")
                    return

                read_start = time.time()
                self.reporting_response_mem.seek(0)
                size_bytes = self.reporting_response_mem.read(8)
                data_size = struct.unpack('Q', size_bytes)[0]
                result_bytes = self.reporting_response_mem.read(data_size)
                result_mb = len(result_bytes) / 1024 / 1024
                deserialize_start = time.time()
                result = msgpack.unpackb(result_bytes, raw=False, strict_map_key=False)
                bt.logging.info(f"Read reporting response data ({time.time()-read_start:.4f}s | {result_mb:.2f}MB)")
                self.initial_balances_published = result['initial_balances_published']
                self.miner_stats = result['miner_stats']

            except Exception as e:
                self.pagerduty_alert(f"Error sending to reporting service: {e}", details={"trace": traceback.format_exc()})
            finally:
                self._reporting = False
                bt.logging.info(f"Completed reporting for step {reporting_step} ({time.time() - start:.4f}s)")

        def report(self) -> None:
            if self.config.reporting.disabled or not self.last_state or self.last_state.timestamp % self.config.scoring.interval != 0:
                return
            if self._reporting:
                bt.logging.warning(f"Skipping reporting at step {self.step} — previous report still running.")
                return
            bt.logging.debug(f"[REPORT] Scheduling from thread: {threading.current_thread().name}")
            self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._report()))

        def _match_trade_fifo(self, uid: int, book_id: int, is_buy: bool, quantity: float,
                            price: float, fee: float, timestamp: int) -> tuple[float, float]:
            """
            FIFO matching including fee accounting.
            Args:
                uid: Miner UID
                book_id: Book identifier
                is_buy: True if buying (going long), False if selling (going short)
                quantity: Trade quantity
                price: Trade price
                fee: Fee paid for this trade (positive = cost, negative = rebate)
                timestamp: Trade timestamp

            Returns:
                tuple[float, float]: (realized_pnl, roundtrip_volume)
                    - realized_pnl: Realized P&L from matched trades (including fees)
                    - roundtrip_volume: Total quantity that completed a round-trip
            """
            positions = self.open_positions[uid][book_id]

            if is_buy:
                shorts = positions['shorts']
                if not shorts:
                    positions['longs'].append((timestamp, quantity, price, fee))
                    return 0.0, 0.0
            else:
                longs = positions['longs']
                if not longs:
                    positions['shorts'].append((timestamp, quantity, price, fee))
                    return 0.0, 0.0

            realized_pnl = 0.0
            roundtrip_volume = 0.0
            remaining_qty = quantity

            quantity_inv = 1.0 / quantity if quantity > 0 else 0.0

            if is_buy:
                # Buying: close shorts first (FIFO), then open longs
                while remaining_qty > 0 and shorts:
                    old_ts, old_qty, old_price, old_fee = shorts[0]

                    if old_qty <= remaining_qty:
                        # Fully close this short position
                        price_pnl = (old_price - price) * old_qty
                        close_fee = fee * old_qty * quantity_inv
                        realized_pnl += price_pnl - old_fee - close_fee
                        roundtrip_volume += old_qty
                        remaining_qty -= old_qty
                        shorts.popleft()
                    else:
                        # Partially close short position
                        old_qty_inv = 1.0 / old_qty

                        price_pnl = (old_price - price) * remaining_qty
                        close_fee = fee  # Entire trade closes positions
                        open_fee = old_fee * remaining_qty * old_qty_inv
                        realized_pnl += price_pnl - open_fee - close_fee
                        roundtrip_volume += remaining_qty

                        # Update remaining position with reduced fee
                        remaining_position_fee = old_fee - open_fee
                        shorts[0] = (old_ts, old_qty - remaining_qty, old_price, remaining_position_fee)
                        remaining_qty = 0

                # Any remaining quantity opens new long position
                if remaining_qty > 0:
                    open_fee = fee * remaining_qty * quantity_inv
                    positions['longs'].append((timestamp, remaining_qty, price, open_fee))

            else:
                # Selling: close longs first (FIFO), then open shorts
                while remaining_qty > 0 and longs:
                    old_ts, old_qty, old_price, old_fee = longs[0]

                    if old_qty <= remaining_qty:
                        # Fully close this long position
                        price_pnl = (price - old_price) * old_qty
                        close_fee = fee * old_qty * quantity_inv
                        realized_pnl += price_pnl - old_fee - close_fee
                        roundtrip_volume += old_qty
                        remaining_qty -= old_qty
                        longs.popleft()
                    else:
                        # Partially close long position
                        old_qty_inv = 1.0 / old_qty

                        price_pnl = (price - old_price) * remaining_qty
                        close_fee = fee  # Entire trade closes positions
                        open_fee = old_fee * remaining_qty * old_qty_inv
                        realized_pnl += price_pnl - open_fee - close_fee
                        roundtrip_volume += remaining_qty

                        # Update remaining position with reduced fee
                        remaining_position_fee = old_fee - open_fee
                        longs[0] = (old_ts, old_qty - remaining_qty, old_price, remaining_position_fee)
                        remaining_qty = 0

                # Any remaining quantity opens new short position
                if remaining_qty > 0:
                    open_fee = fee * remaining_qty * quantity_inv
                    positions['shorts'].append((timestamp, remaining_qty, price, open_fee))

            return realized_pnl, roundtrip_volume

        def _update_trade_volumes(self, state: MarketSimulationStateUpdate):
            """
            Updates and maintains all trade volume tracking and position accounting structures.

            This function processes raw trade events from the simulator state and updates
            the following per-UID per-book time series:

            **Volume Tracking:**
            • **total** — total traded notional value
            • **maker** — maker-side volume
            • **taker** — taker-side volume
            • **self** — trades where maker == taker
            • **roundtrip_volumes** — volume from completed round-trip trades (open + close)
            • **volume_sums** / **maker_volume_sums** / **taker_volume_sums** / **self_volume_sums** / **roundtrip_volume_sums**

            **Position Accounting (FIFO):**
            • **open_positions** — tracks open long/short positions with (timestamp, quantity, price, fee)
            • **realized_pnl_history** — realized profit/loss from closed positions (fee-adjusted)
            • Matches trades via FIFO to calculate realized P&L and round-trip volume

            **Inventory & History:**
            • **inventory_history** — mark-to-market inventory value changes over time
            • **recent_trades** — rolling buffer of last 25 trades per book
            • **recent_miner_trades** — rolling buffer of last 5 trades per miner per book
            • **initial_balances** — baseline balances for inventory value calculations

            **Operations:**
            • Samples volume at aligned timestamps (trade_volume_sampling_interval)
            • Prunes old volume entries outside assessment window (trade_volume_assessment_period)
            • Prunes old inventory and realized P&L history outside Kappa lookback window
            • Batch processes updates for performance (deferred rounding)
            • Ensures all nested structures are initialized dynamically

            Args:
                state (MarketSimulationStateUpdate):
                    Full simulation tick state containing books, accounts, and notices.

            Returns:
                None

            Raises:
                Logs errors when UID-level processing fails but continues processing remaining UIDs.
            """
            total_start = time.time()

            books = state.books
            timestamp = state.timestamp
            accounts = state.accounts
            notices = state.notices

            volume_decimals = self.simulation.volumeDecimals
            book_count = self.simulation.book_count

            sampled_timestamp = (timestamp // self.config.scoring.activity.trade_volume_sampling_interval) * self.config.scoring.activity.trade_volume_sampling_interval

            if not hasattr(self, '_last_prune_timestamp'):
                self._last_prune_timestamp = None

            if self._last_prune_timestamp:
                time_since_prune = timestamp - self._last_prune_timestamp
                prune_interval = 60_000_000_000
                should_prune = time_since_prune >= prune_interval
            else:
                should_prune = True
            if should_prune:
                self._last_prune_timestamp = timestamp
                bt.logging.info(f"Pruning at step {self.step} (timestamp {timestamp})")
            volume_prune_threshold = timestamp - self.config.scoring.activity.trade_volume_assessment_period

            for bookId, book in books.items():
                trades = [event for event in book['e'] if event['y'] == 't']
                if trades:
                    recent_trades_book = self.recent_trades[bookId]
                    recent_trades_book.extend([TradeInfo.model_construct(**t) for t in trades])
                    del recent_trades_book[:-25]

            volume_deltas = {}
            realized_pnl_updates = {}
            roundtrip_volume_updates = {}
            uids_to_round = set()

            uid_count = 0
            for uid_item in range(self.effective_max_uids):
                uid_count += 1
                try:
                    # Initialize trade volumes structure if needed
                    if uid_item not in self.trade_volumes:
                        self.trade_volumes[uid_item] = {
                            book_id: {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                            for book_id in range(book_count)
                        }
                    trade_volumes_uid = self.trade_volumes[uid_item]

                    # Prune old volumes and update sums
                    if should_prune:
                        for book_id, role_trades in trade_volumes_uid.items():
                            for role, trades in role_trades.items():
                                if not trades:
                                    continue
                                old_count = len(trades)
                                pruned = {t: v for t, v in trades.items() if t >= volume_prune_threshold}
                                if len(pruned) < old_count:
                                    pruned_volume = sum(v for t, v in trades.items() if t < volume_prune_threshold)
                                    if pruned_volume > 0:
                                        if role == 'total':
                                            self.volume_sums[uid_item][book_id] = max(0.0, self.volume_sums[uid_item][book_id] - pruned_volume)
                                        elif role == 'maker':
                                            self.maker_volume_sums[uid_item][book_id] = max(0.0, self.maker_volume_sums[uid_item][book_id] - pruned_volume)
                                        elif role == 'taker':
                                            self.taker_volume_sums[uid_item][book_id] = max(0.0, self.taker_volume_sums[uid_item][book_id] - pruned_volume)
                                        elif role == 'self':
                                            self.self_volume_sums[uid_item][book_id] = max(0.0, self.self_volume_sums[uid_item][book_id] - pruned_volume)
                                        uids_to_round.add(uid_item)
                                    trade_volumes_uid[book_id][role] = pruned

                    # Initialize sampled timestamp entries
                    for book_id in range(book_count):
                        if book_id not in trade_volumes_uid:
                            trade_volumes_uid[book_id] = {'total': {}, 'maker': {}, 'taker': {}, 'self': {}}
                        book_trade_volumes = trade_volumes_uid[book_id]
                        if sampled_timestamp not in book_trade_volumes['total']:
                            book_trade_volumes['total'][sampled_timestamp] = 0.0
                            book_trade_volumes['maker'][sampled_timestamp] = 0.0
                            book_trade_volumes['taker'][sampled_timestamp] = 0.0
                            book_trade_volumes['self'][sampled_timestamp] = 0.0

                    # Process trade notices
                    if uid_item in notices:
                        trades = [notice for notice in notices[uid_item] if notice['y'] in ['EVENT_TRADE', "ET"]]
                        if trades:
                            if uid_item not in self.recent_miner_trades:
                                self.recent_miner_trades[uid_item] = {b: [] for b in range(self.simulation.book_count)}
                            recent_miner_trades_uid = self.recent_miner_trades[uid_item]
                            if uid_item not in volume_deltas:
                                volume_deltas[uid_item] = {}

                            for trade in trades:
                                is_maker = trade['Ma'] == uid_item
                                is_taker = trade['Ta'] == uid_item
                                book_id = trade['b']

                                # Update recent miner trades
                                if is_maker:
                                    recent_miner_trades_uid[book_id].append([TradeEvent.model_construct(**trade), "maker"])
                                if is_taker:
                                    recent_miner_trades_uid[book_id].append([TradeEvent.model_construct(**trade), "taker"])
                                if len(recent_miner_trades_uid[book_id]) > 5:
                                    del recent_miner_trades_uid[book_id][:-5]

                                book_volumes = trade_volumes_uid[book_id]
                                trade_value = trade['q'] * trade['p']
                                if book_id not in volume_deltas[uid_item]:
                                    volume_deltas[uid_item][book_id] = {'total': 0.0, 'maker': 0.0, 'taker': 0.0, 'self': 0.0}

                                book_volumes['total'][sampled_timestamp] += trade_value
                                volume_deltas[uid_item][book_id]['total'] += trade_value

                                if trade['Ma'] == trade['Ta']:
                                    book_volumes['self'][sampled_timestamp] += trade_value
                                    volume_deltas[uid_item][book_id]['self'] += trade_value
                                elif is_maker:
                                    book_volumes['maker'][sampled_timestamp] += trade_value
                                    volume_deltas[uid_item][book_id]['maker'] += trade_value
                                elif is_taker:
                                    book_volumes['taker'][sampled_timestamp] += trade_value
                                    volume_deltas[uid_item][book_id]['taker'] += trade_value

                                uids_to_round.add(uid_item)

                                # FIFO Matching: Calculate realized P&L and round-trip volume
                                quantity = trade['q']
                                price = trade['p']
                                side = trade['s']
                                is_buy = (is_taker and side == 0) or (is_maker and side == 1)
                                fee = trade['Mf'] if is_maker else trade['Tf']

                                realized_pnl, roundtrip_volume = self._match_trade_fifo(
                                    uid_item, book_id, is_buy, quantity, price, fee, timestamp
                                )

                                if realized_pnl != 0.0:
                                    if uid_item not in realized_pnl_updates:
                                        realized_pnl_updates[uid_item] = {}
                                    if timestamp not in realized_pnl_updates[uid_item]:
                                        realized_pnl_updates[uid_item][timestamp] = {}
                                    if book_id not in realized_pnl_updates[uid_item][timestamp]:
                                        realized_pnl_updates[uid_item][timestamp][book_id] = 0.0
                                    realized_pnl_updates[uid_item][timestamp][book_id] += realized_pnl

                                if roundtrip_volume > 0:
                                    roundtrip_value = roundtrip_volume * price
                                    if uid_item not in roundtrip_volume_updates:
                                        roundtrip_volume_updates[uid_item] = {}
                                    if sampled_timestamp not in roundtrip_volume_updates[uid_item]:
                                        roundtrip_volume_updates[uid_item][sampled_timestamp] = {}
                                    if book_id not in roundtrip_volume_updates[uid_item][sampled_timestamp]:
                                        roundtrip_volume_updates[uid_item][sampled_timestamp][book_id] = 0.0
                                    roundtrip_volume_updates[uid_item][sampled_timestamp][book_id] += roundtrip_value

                            for book_id, deltas in volume_deltas[uid_item].items():
                                self.volume_sums[uid_item][book_id] = self.volume_sums[uid_item].get(book_id, 0.0) + deltas['total']
                                self.maker_volume_sums[uid_item][book_id] = self.maker_volume_sums[uid_item].get(book_id, 0.0) + deltas['maker']
                                self.taker_volume_sums[uid_item][book_id] = self.taker_volume_sums[uid_item].get(book_id, 0.0) + deltas['taker']
                                self.self_volume_sums[uid_item][book_id] = self.self_volume_sums[uid_item].get(book_id, 0.0) + deltas['self']
                    # Initialize zero P&L for timestamps with no trades
                    if timestamp not in self.realized_pnl_history[uid_item]:
                        self.realized_pnl_history[uid_item][timestamp] = {}

                    # Update inventory history
                    if uid_item in accounts:
                        if uid_item not in self.initial_balances:
                            self.initial_balances[uid_item] = {
                                b: {'BASE': None, 'QUOTE': None, 'WEALTH': None}
                                for b in range(self.simulation.book_count)
                            }
                        initial_balances_uid = self.initial_balances[uid_item]
                        accounts_uid = accounts[uid_item]

                        for bookId, account in accounts_uid.items():
                            initial_balance_book = initial_balances_uid[bookId]
                            if initial_balance_book['BASE'] is None:
                                initial_balance_book['BASE'] = account['bb']['t']
                            if initial_balance_book['QUOTE'] is None:
                                initial_balance_book['QUOTE'] = account['qb']['t']
                            if initial_balance_book['WEALTH'] is None:
                                initial_balance_book['WEALTH'] = get_inventory_value(account, books[bookId])

                        current_inventory = {
                            book_id: get_inventory_value(accounts_uid[book_id], book) - initial_balances_uid[book_id]['WEALTH']
                            for book_id, book in books.items()
                        }
                        if uid_item not in self.inventory_history:
                            self.inventory_history[uid_item] = {}
                        hist = self.inventory_history[uid_item]
                        if not hist:
                            hist[timestamp] = current_inventory
                        else:
                            timestamps = sorted(hist.keys())
                            if len(timestamps) == 1:
                                hist[timestamp] = current_inventory
                            else:
                                first_ts = timestamps[0]
                                self.inventory_history[uid_item] = {
                                    first_ts: hist[first_ts],
                                    timestamps[-1]: hist[timestamps[-1]],
                                    timestamp: current_inventory
                                }
                    else:
                        self.inventory_history[uid_item][timestamp] = {book_id: 0.0 for book_id in books}
                except Exception as ex:
                    self.pagerduty_alert(f"Failed to update trade data for UID {uid_item}: {ex}", details={"trace": traceback.format_exc()})

            if should_prune:
                lookback_time = self.config.scoring.kappa.lookback
                lookback_threshold = timestamp - lookback_time
                for uid_item in self.realized_pnl_history:
                    pnl_hist = self.realized_pnl_history[uid_item]
                    if not pnl_hist:
                        continue
                    self.realized_pnl_history[uid_item] = {
                        ts: books 
                        for ts, books in pnl_hist.items() 
                        if ts >= lookback_threshold
                    }
                for uid_item in self.roundtrip_volumes:
                    roundtrip_volumes_uid = self.roundtrip_volumes[uid_item]
                    
                    for book_id, rt_volumes in roundtrip_volumes_uid.items():
                        if not rt_volumes:
                            continue
                        old_count = len(rt_volumes)
                        pruned = {t: v for t, v in rt_volumes.items() if t >= volume_prune_threshold}
                        if len(pruned) < old_count:
                            pruned_rt_volume = sum(v for t, v in rt_volumes.items() if t < volume_prune_threshold)
                            if pruned_rt_volume > 0:
                                current = self.roundtrip_volume_sums[uid_item][book_id]
                                self.roundtrip_volume_sums[uid_item][book_id] = max(0.0, current - pruned_rt_volume)
                                uids_to_round.add(uid_item)
                            roundtrip_volumes_uid[book_id] = pruned

            for uid_item, timestamps in realized_pnl_updates.items():
                if uid_item not in self.realized_pnl_history:
                    self.realized_pnl_history[uid_item] = {}
                for ts, books in timestamps.items():
                    if ts not in self.realized_pnl_history[uid_item]:
                        self.realized_pnl_history[uid_item][ts] = {}
                    ts_pnl = self.realized_pnl_history[uid_item][ts]
                    for book_id, pnl in books.items():
                        rounded_pnl = round(pnl, volume_decimals)
                        if rounded_pnl == 0.0:
                            continue
                        current = ts_pnl.get(book_id, 0.0)
                        new_value = round(current + rounded_pnl, volume_decimals)
                        if new_value != 0.0:
                            ts_pnl[book_id] = new_value
                        elif book_id in ts_pnl:
                            del ts_pnl[book_id]
            for uid_item, timestamps in roundtrip_volume_updates.items():
                for ts, books in timestamps.items():
                    for book_id, rt_vol in books.items():
                        if uid_item not in self.roundtrip_volumes:
                            self.roundtrip_volumes[uid_item] = defaultdict(lambda: defaultdict(float))
                        if book_id not in self.roundtrip_volumes[uid_item]:
                            self.roundtrip_volumes[uid_item][book_id] = defaultdict(float)
                        if ts not in self.roundtrip_volumes[uid_item][book_id]:
                            self.roundtrip_volumes[uid_item][book_id][ts] = 0.0                        
                        self.roundtrip_volumes[uid_item][book_id][ts] += rt_vol
                        self.roundtrip_volume_sums[uid_item][book_id] = self.roundtrip_volume_sums[uid_item].get(book_id, 0.0) + rt_vol
                        uids_to_round.add(uid_item)
            for uid_item in uids_to_round:
                changed_books = set(volume_deltas.get(uid_item, {}).keys())

                if uid_item in roundtrip_volume_updates:
                    for ts_books in roundtrip_volume_updates[uid_item].values():
                        changed_books.update(ts_books.keys())
                if not changed_books:
                    changed_books = range(book_count)

                for book_id in changed_books:
                    if uid_item in self.trade_volumes and book_id in self.trade_volumes[uid_item]:
                        book_vols = self.trade_volumes[uid_item][book_id]
                        for role in ['total', 'maker', 'taker', 'self']:
                            if sampled_timestamp in book_vols[role]:
                                book_vols[role][sampled_timestamp] = round(book_vols[role][sampled_timestamp], volume_decimals)

                    if book_id in self.volume_sums[uid_item]:
                        self.volume_sums[uid_item][book_id] = round(self.volume_sums[uid_item][book_id], volume_decimals)
                    if book_id in self.maker_volume_sums[uid_item]:
                        self.maker_volume_sums[uid_item][book_id] = round(self.maker_volume_sums[uid_item][book_id], volume_decimals)
                    if book_id in self.taker_volume_sums[uid_item]:
                        self.taker_volume_sums[uid_item][book_id] = round(self.taker_volume_sums[uid_item][book_id], volume_decimals)
                    if book_id in self.self_volume_sums[uid_item]:
                        self.self_volume_sums[uid_item][book_id] = round(self.self_volume_sums[uid_item][book_id], volume_decimals)
                    if book_id in self.roundtrip_volume_sums[uid_item]:
                        self.roundtrip_volume_sums[uid_item][book_id] = round(self.roundtrip_volume_sums[uid_item][book_id], volume_decimals)

                    if uid_item in realized_pnl_updates:
                        for ts in realized_pnl_updates[uid_item]:
                            if book_id < book_count and ts in self.realized_pnl_history[uid_item]:
                                if book_id in self.realized_pnl_history[uid_item][ts]:
                                    self.realized_pnl_history[uid_item][ts][book_id] = round(
                                        self.realized_pnl_history[uid_item][ts][book_id],
                                        volume_decimals
                                    )
            total_time = time.time() - total_start
            if should_prune:
                bt.logging.debug(f"[UPDATE_VOLUMES] Total: {total_time:.4f}s (pruned, {uid_count} UIDs)")
            else:
                bt.logging.debug(f"[UPDATE_VOLUMES] Total: {total_time:.4f}s ({uid_count} UIDs)")

        def should_block_queries(self) -> bool:
            """Block queries if reward is lagging."""
            if self._pending_reward_tasks >= 5:
                return True
            return False

        async def _reward(self, state: MarketSimulationStateUpdate):
            """
            Asynchronously perform the full reward computation pipeline.
            """
            if not hasattr(self, "_reward_lock"):
                self._reward_lock = asyncio.Lock()

            with self._rewarding_lock:
                self._pending_reward_tasks += 1

            start_wait = time.time()
            rewarding_step = self.step
            try:
                async with self._reward_lock:
                    waited = time.time() - start_wait
                    if waited > 0:
                        bt.logging.debug(f"Acquired reward lock after waiting {waited:.3f}s")

                    self.shared_state_rewarding = True

                    timestamp = state.timestamp
                    duration = duration_from_timestamp(timestamp)
                    bt.logging.info(f"Starting reward calculation for step {rewarding_step}...")
                    start = time.time()

                    try:
                        self._update_trade_volumes(state)
                        if timestamp % self.config.scoring.interval != 0:
                            bt.logging.info(f"Agent Scores Data Updated for {duration} ({time.time()-start:.4f}s)")
                            return
                        bt.logging.info("Starting reward calculation...")
                        calc_start = time.time()

                        loop = asyncio.get_event_loop()
                        trading_rewards, gentrx_rewards, updated_data, all_uids = await loop.run_in_executor(
                            self.reward_executor,
                            get_rewards,
                            self
                        )

                        bt.logging.info(f"Reward calculation completed ({time.time()-calc_start:.4f}s)")

                        self.kappa_values = updated_data['kappa_values']
                        self.activity_factors = updated_data['activity_factors']
                        self.pnl_factors = updated_data['pnl_factors']

                        bt.logging.debug(
                            f"Agent Rewards Recalculated for {duration} ({time.time()-start:.4f}s):\n"
                            f"trading={trading_rewards}\ngentrx={gentrx_rewards}"
                        )
                        self.update_scores(trading_rewards, all_uids, gentrx_rewards=gentrx_rewards)
                        bt.logging.info(f"Agent Scores Updated for {duration} ({time.time()-start:.4f}s)")
                        self._last_rewarded_sim_timestamp = timestamp

                    except Exception as ex:
                        self.pagerduty_alert(f"Rewarding failed: {ex}", details={"trace": traceback.format_exc()})
                    finally:
                        self.shared_state_rewarding = False
                        bt.logging.debug(f"Completed rewarding (TOTAL {time.time()-start_wait:.4f}s).")
            finally:
                with self._rewarding_lock:
                    self._pending_reward_tasks -= 1

        def reward(self, state : MarketSimulationStateUpdate) -> None:
            """
            Schedule asynchronous reward calculation on the validator's main event loop.
            Offloads work to `_reward()` to ensure that:

            • Reward computation always occurs in the correct asyncio event loop
            • CPU-intensive work does not block the calling thread
            • Reward logic executes with proper async locking

            Args:
                state (MarketSimulationStateUpdate):
                    Simulation state for the current tick, forwarded to `_reward`.
            """
            bt.logging.debug(f"[REWARD] Scheduling from thread: {threading.current_thread().name}")
            bt.logging.debug(f"[REWARD] Main loop ID: {id(self.main_loop)}, Current loop ID: {id(asyncio.get_event_loop())}")
            self.main_loop.call_soon_threadsafe(lambda: self.main_loop.create_task(self._reward(state)))

        async def handle_state(self, message : dict, state : MarketSimulationStateUpdate, receive_start : int) -> dict:
            """
            Handle a full simulator state update, enrich it with validator data, compute responses,
            update internal validator state, and return instructions back to the simulator.


            This method is the central processing loop for each simulation step. It performs:
            - Periodic validator configuration reloads.
            - Per‑account volume injection.
            - Simulation metadata updates and logging.
            - State forwarding to miners and response aggregation.
            - Reward calculation, scoring, persistence, and metric publication.


            Args:
            message (dict): The raw simulator state message as received (typically msgpack‑decoded).
            state (MarketSimulationStateUpdate): Parsed simulation state model containing orderbooks, accounts, timestamps, etc.
            receive_start (int): Timestamp marking when the simulator delivered the message, used for latency metrics.


            Returns:
            dict: Serialized response batch to be returned to the simulator.
            """
            # Every 1H of simulation time, check if there are any changes to the validator - if updates exist, pull them and restart.
            if self.simulation_timestamp % 3600_000_000_000 == 0 and self.simulation_timestamp != 0:
                bt.logging.info("Checking for validator updates...")
                self.update_repo()
            state.version = __spec_version__
            start = time.time()
            for uid, accounts in state.accounts.items():
                for book_id in accounts:
                    state.accounts[uid][book_id]['v'] = self.volume_sums.get((uid, book_id), 0.0)
            bt.logging.info(f"Volumes added to state ({time.time()-start:.4f}s).")

            # Update variables
            if not self.start_time:
                self.start_time = time.time()
                self.start_timestamp = state.timestamp
            if self.simulation.logDir != message['logDir']:
                bt.logging.info(
                    f"Simulation log directory changed : {self.simulation.logDir} -> {message['logDir']}"
                )
                self.simulation.logDir = message['logDir']
                self._notify_seed_log_dir_change(message['logDir'])

            self.simulation_timestamp = state.timestamp
            self.step_rates.append((state.timestamp - (self.last_state.timestamp if self.last_state else self.start_timestamp)) / (time.time() - (self.last_state_time if self.last_state_time else self.start_time)))
            self.last_state = state
            if self.simulation:
                self.simulation.simulation_id = os.path.basename(self.simulation.logDir)[:13]
                state.config = self.simulation.model_copy()
                state.config.logDir = None
            self.step += 1

            if self.simulation_timestamp % self.simulation.log_window == self.simulation.publish_interval:
                self.compress_outputs()

            # Log received state data
            bt.logging.info(f"STATE UPDATE RECEIVED | VALIDATOR STEP : {self.step} | TIME : {duration_from_timestamp(state.timestamp)} (T={state.timestamp})")
            if self.config.logging.debug or self.config.logging.trace:
                debug_text = ''
                for bookId, book in state.books.items():
                    debug_text += '-' * 50 + "\n"
                    debug_text += f"BOOK {bookId}" + "\n"
                    if book['b'] and book['a']:
                        debug_text += ' | '.join([f"{level['q']:.4f}@{level['p']}" for level in reversed(book['b'][:5])]) + '||' + ' | '.join([f"{level['q']:.4f}@{level['p']}" for level in book['a'][:5]]) + "\n"
                    else:
                        debug_text += "EMPTY" + "\n"
                bt.logging.debug("\n" + debug_text.strip("\n"))

            # Process deregistration notices
            self.process_resets(state)

            # GenTRX: save state to S3 + poll/deliver assignments
            if self._gentrx is not None:
                try:
                    self._gentrx.push_state(state)
                    await self._gentrx.poll_and_deliver()
                except Exception as _gex:
                    gtx_log.warning(f"handle_state error: {_gex}")

            # Forward state synapse to miners, populate response data to simulator object and serialize for returning to simulator.
            start = time.time()
            response = SimulatorResponseBatch(await forward(self, state))
            bt.logging.debug(f"Gathered Response Batch ({time.time()-start}s)")
            start = time.time()
            response = response.serialize()
            bt.logging.debug(f"Serialized Response Batch ({time.time()-start}s)")
            # Log response data, start state serialization and reporting threads, and return miner instructions to the simulator
            if len(response['responses']) > 0:
                bt.logging.trace(f"RESPONSE : {response}")
            bt.logging.info(f"RATE : {(self.step_rates[-1] if self.step_rates != [] else 0) / 1e9:.2f} STEPS/s | AVG : {(sum(self.step_rates) / len(self.step_rates) / 1e9 if self.step_rates != [] else 0):.2f}  STEPS/s")
            self.step_rates = self.step_rates[-10000:]
            self.last_state_time = time.time()

            # Calculate latest rewards, update miner scores, save state and publish metrics
            self.maintain()
            self.reward(state)
            self.save_state()
            self.report()
            bt.logging.info(f"State update handled ({time.time()-receive_start}s)")

            return response

        async def _listen(self):
            """
            Continuously listen for simulator state updates via POSIX IPC, unpack them,
            parse them into state objects, and process them with `handle_state`.

            This listener runs the full validator event loop when operating in IPC mode.
            It performs:
            - Receiving shared‑memory pointers via message queues.
            - mmap reads with retry logic.
            - msgpack unpacking with retry logic.
            - Parsing the state dict into a typed model.
            - Handling all state updates and forwarding miner responses.

            The method uses run‑in‑executor offloading for blocking IPC operations.
            """
            def receive(mq_req: posix_ipc.MessageQueue) -> tuple:
                self._receiving_state = True
                try:
                    msg, priority = mq_req.receive()
                    receive_start = time.time()
                    bt.logging.info(f"Received state update from simulator (msgpack)")
                    byte_size_req = int.from_bytes(msg, byteorder="little")
                    shm_req = posix_ipc.SharedMemory("/state")
                    start = time.time()
                    packed_data = None
                    for attempt in range(1, 6):
                        try:
                            with mmap.mmap(shm_req.fd, byte_size_req, mmap.MAP_SHARED, mmap.PROT_READ) as mm:
                                packed_data = mm.read(byte_size_req)
                            break
                        except Exception as ex:
                            if attempt < 5:
                                bt.logging.error(f"mmap read failed (attempt {attempt}/5): {ex}")
                                time.sleep(0.005)
                            else:
                                bt.logging.error(f"mmap read failed on all 5 attempts: {ex}")
                                return None, receive_start
                        finally:
                            if packed_data is not None or attempt >= 5:
                                shm_req.close_fd()
                    bt.logging.info(f"Retrieved State Update ({time.time() - receive_start}s)")
                    start = time.time()
                    for attempt in range(1, 6):
                        try:
                            result = msgpack.unpackb(packed_data, raw=False, use_list=True, strict_map_key=False)
                            bt.logging.info(f"Unpacked state update ({time.time() - start:.4f}s)")
                            break
                        except Exception as ex:
                            if attempt < 5:
                                bt.logging.error(f"Msgpack unpack failed (attempt {attempt}/5): {ex}")
                                time.sleep(0.005)
                            else:
                                bt.logging.error(f"Msgpack unpack failed on all 5 attempts: {ex}")
                                return None, receive_start
                finally:                    
                    self._receiving_state = False
                return result, receive_start

            def respond(response: dict) -> dict:
                self.last_response = response
                packed_res = msgpack.packb(response, use_bin_type=True)
                byte_size_res = len(packed_res)
                mq_res = posix_ipc.MessageQueue("/taosim-res", flags=posix_ipc.O_CREAT, max_messages=1, max_message_size=8)
                shm_res = posix_ipc.SharedMemory("/responses", flags=posix_ipc.O_CREAT, size=byte_size_res)
                with mmap.mmap(shm_res.fd, byte_size_res, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ) as mm:
                    shm_res.close_fd()
                    mm.write(packed_res)
                mq_res.send(byte_size_res.to_bytes(8, byteorder="little"))
                mq_res.close()

            mq_req = posix_ipc.MessageQueue("/taosim-req", flags=posix_ipc.O_CREAT, max_messages=1, max_message_size=8)
            thread_pool = ThreadPoolExecutor(max_workers=4)
            try:
                while True:
                    response = {"responses": []}
                    try:
                        loop = asyncio.get_event_loop()
                        t1 = time.time()
                        bt.logging.debug(f"[LISTEN] Starting receive at {t1:.3f}")
                        message, receive_start = await loop.run_in_executor(thread_pool, receive, mq_req)
                        if message:
                            t2 = time.time()
                            bt.logging.debug(f"[LISTEN] Received message in {t2-t1:.4f}s")
                            state = MarketSimulationStateUpdate.parse_dict(message)
                            t3 = time.time()
                            bt.logging.info(f"Parsed state dict ({t3-t2:.4f}s)")
                            response = await self.handle_state(message, state, receive_start)
                            t4 = time.time()
                            bt.logging.debug(f"[LISTEN] handle_state completed in {t4-t3:.4f}s")
                    except Exception as ex:
                        traceback.print_exc()
                        self.pagerduty_alert(f"Exception in posix listener loop : {ex}", details={"trace": traceback.format_exc()})
                    finally:
                        t5 = time.time()
                        bt.logging.debug(f"[LISTEN] Starting respond at {t5:.3f}")
                        await loop.run_in_executor(thread_pool, respond, response)
                        t6 = time.time()
                        bt.logging.debug(f"[LISTEN] Respond completed in {t6-t5:.4f}s")
                        bt.logging.debug(f"[LISTEN] Total loop iteration: {t6-t1:.4f}s")
            finally:
                mq_req.close()
                thread_pool.shutdown(wait=True)

        def listen(self):
            """
            Synchronous wrapper for the asynchronous `_listen` method.
            """
            try:
                os.nice(-10)
            except PermissionError:
                bt.logging.warning("Cannot set process priority (need sudo for negative nice values)")
            try:
                asyncio.run(self._listen())
            except KeyboardInterrupt:
                print("Listening stopped by user.")

        async def orderbook(self, request : Request) -> dict:
            """
            HTTP route endpoint that receives a complete simulator state update over HTTP,
            parses it, and forwards it to `handle_state`.


            This is the HTTP equivalent of the IPC listener used when running a
            distributed or containerized simulator. It performs:
            - Streaming request‑body read.
            - Basic JSON‑structure validation.
            - Construction of a Ypy‑backed state object.
            - Conversion into a typed MarketSimulationStateUpdate model.
            - Delegation to the main validator processing pipeline.


            Args:
            request (Request): Incoming HTTP request containing a JSON‑encoded simulation state update.


            Returns:
            dict: Serialized simulation response batch.
            """
            bt.logging.info("Received state update from simulator.")
            global_start = time.time()
            start = time.time()
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
            bt.logging.info(f"Retrieved request body ({time.time()-start:.4f}s).")
            if body[-3:].decode() != "]}}":
                raise Exception(f"Incomplete JSON!")
            message = YpyObject(body, 1)
            bt.logging.info(f"Constructed YpyObject ({time.time()-start:.4f}s).")
            state = MarketSimulationStateUpdate.from_ypy(message)
            bt.logging.info(f"Synapse populated ({time.time()-start:.4f}s).")
            del body

            response = await self.handle_state(message, state)

            bt.logging.info(f"State update processed ({time.time()-global_start}s)")
            return response

        async def account(self, request : Request) -> None:
            """
            HTTP route endpoint for receiving event‑level notifications from the simulator
            (e.g., simulation start, simulation end, error reports, market notices).


            Responsibilities:
            - Immediately forward simulation‑start events to miners.
            - Handle simulation‑end markers.
            - Record and persist error‑report batches.
            - Forward all other event notifications to miners.
            - Trigger alerting for msgpack or simulation integrity errors.

            Args:
            request (Request): HTTP request containing a batch of simulator event messages.

            Returns:
            None | dict: `{"continue": True/False}` when error‑report limits are reached.
            Otherwise returns `None`.
            """
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
            batch = msgspec.json.decode(body)
            bt.logging.info(f"NOTICE : {batch}")
            notices = []
            ended = False
            for message in batch['messages']:
                if message['type'] == 'EVENT_SIMULATION_START':
                    self.onStart(message['timestamp'], FinanceEventNotification.from_json(message).event)
                    continue
                elif message['type'] == 'EVENT_SIMULATION_END':
                    ended = True
                elif message['type'] == 'RESPONSES_ERROR_REPORT':
                    dump_file = self.config.neuron.full_path + f"/{self.last_state.config.simulation_id}.{message['timestamp']}.responses.json"
                    with open(dump_file, "w") as f:
                        json.dump(self.last_response, f, indent=4)
                    error_file = self.config.neuron.full_path + f"/{self.last_state.config.simulation_id}.{message['timestamp']}.error.json"
                    with open(error_file, "w") as f:
                        json.dump(message, f, indent=4)
                    self.msgpack_error_counter += len(message) - 3
                    if self.msgpack_error_counter < 10:
                        self.pagerduty_alert(f"{self.msgpack_error_counter} msgpack deserialization errors encountered in simulator - continuing.", details=message)
                        return { "continue": True }
                    else:
                        self.pagerduty_alert(f"{self.msgpack_error_counter} msgpack deserialization errors encountered in simulator - terminating simulation.", details=message)
                        return { "continue": False }
                notice = FinanceEventNotification.from_json(message)
                if not notice:
                    bt.logging.error(f"Unrecognized notification : {message}")
                else:
                    notices.append(notice)
            await notify(self, notices)
            if ended:
                self.onEnd()

        def cleanup_ipc(self):
            """
            Shuts down the query service and releases all POSIX IPC resources.

            Behavior:
                - Attempts to send a shutdown message to the query service.
                - Waits for graceful termination, falling back to terminate/kill.
                - Closes memory maps and shared memory file descriptors.
                - Closes message queues.
                - Logs detailed warnings for any partial cleanup failures.

            Returns:
                None
            """
            try:
                bt.logging.info("Cleaning up query service...")
                if hasattr(self, 'request_queue'):
                    try:
                        self.request_queue.send(b'shutdown', timeout=1.0)
                        bt.logging.info("Sent shutdown command to query service")
                    except Exception as e:
                        bt.logging.warning(f"Failed to send shutdown command: {e}")
                if hasattr(self, 'query_process') and self.query_process:
                    try:
                        self.query_process.wait(timeout=5.0)
                        bt.logging.info(f"Query service exited with code {self.query_process.returncode}")
                    except subprocess.TimeoutExpired:
                        bt.logging.warning("Query service did not exit gracefully, terminating...")
                        self.query_process.terminate()
                        try:
                            self.query_process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            bt.logging.error("Query service did not terminate, killing...")
                            self.query_process.kill()

                if hasattr(self, 'request_mem'):
                    try:
                        self.request_mem.close()
                        bt.logging.debug("Closed request memory map")
                    except Exception as e:
                        bt.logging.warning(f"Error closing request memory map: {e}")

                if hasattr(self, 'response_mem'):
                    try:
                        self.response_mem.close()
                        bt.logging.debug("Closed response memory map")
                    except Exception as e:
                        bt.logging.warning(f"Error closing response memory map: {e}")

                if hasattr(self, 'request_shm'):
                    try:
                        self.request_shm.close_fd()
                        bt.logging.debug("Closed request shared memory fd")
                    except Exception as e:
                        bt.logging.warning(f"Error closing request shared memory fd: {e}")

                if hasattr(self, 'response_shm'):
                    try:
                        self.response_shm.close_fd()
                        bt.logging.debug("Closed response shared memory fd")
                    except Exception as e:
                        bt.logging.warning(f"Error closing response shared memory fd: {e}")

                if hasattr(self, 'request_queue'):
                    try:
                        self.request_queue.close()
                        bt.logging.debug("Closed request queue")
                    except Exception as e:
                        bt.logging.warning(f"Error closing request queue: {e}")

                if hasattr(self, 'response_queue'):
                    try:
                        self.response_queue.close()
                        bt.logging.debug("Closed response queue")
                    except Exception as e:
                        bt.logging.warning(f"Error closing response queue: {e}")

                bt.logging.info("Query service cleanup complete")

                bt.logging.info("Cleaning up reporting service...")

                if hasattr(self, 'reporting_request_queue'):
                    try:
                        self.reporting_request_queue.send(b'shutdown', timeout=1.0)
                        bt.logging.info("Sent shutdown command to reporting service")
                    except Exception as e:
                        bt.logging.warning(f"Failed to send shutdown command to reporting: {e}")

                if hasattr(self, 'reporting_process') and self.reporting_process:
                    try:
                        self.reporting_process.wait(timeout=5.0)
                        bt.logging.info(f"Reporting service exited with code {self.reporting_process.returncode}")
                    except subprocess.TimeoutExpired:
                        bt.logging.warning("Reporting service did not exit gracefully, terminating...")
                        self.reporting_process.terminate()
                        try:
                            self.reporting_process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            bt.logging.error("Reporting service did not terminate, killing...")
                            self.reporting_process.kill()

                if hasattr(self, 'reporting_request_mem'):
                    try:
                        self.reporting_request_mem.close()
                        bt.logging.debug("Closed reporting request memory map")
                    except Exception as e:
                        bt.logging.warning(f"Error closing reporting request memory map: {e}")

                if hasattr(self, 'reporting_response_mem'):
                    try:
                        self.reporting_response_mem.close()
                        bt.logging.debug("Closed reporting response memory map")
                    except Exception as e:
                        bt.logging.warning(f"Error closing reporting response memory map: {e}")

                if hasattr(self, 'reporting_request_shm'):
                    try:
                        self.reporting_request_shm.close_fd()
                        bt.logging.debug("Closed reporting request shared memory fd")
                    except Exception as e:
                        bt.logging.warning(f"Error closing reporting request shared memory fd: {e}")

                if hasattr(self, 'reporting_response_shm'):
                    try:
                        self.reporting_response_shm.close_fd()
                        bt.logging.debug("Closed reporting response shared memory fd")
                    except Exception as e:
                        bt.logging.warning(f"Error closing reporting response shared memory fd: {e}")

                if hasattr(self, 'reporting_request_queue'):
                    try:
                        self.reporting_request_queue.close()
                        bt.logging.debug("Closed reporting request queue")
                    except Exception as e:
                        bt.logging.warning(f"Error closing reporting request queue: {e}")

                if hasattr(self, 'reporting_response_queue'):
                    try:
                        self.reporting_response_queue.close()
                        bt.logging.debug("Closed reporting response queue")
                    except Exception as e:
                        bt.logging.warning(f"Error closing reporting response queue: {e}")

                bt.logging.info("Reporting service cleanup complete")

                bt.logging.info("Cleaning up seed service...")
                if hasattr(self, 'seed_process') and self.seed_process:
                    try:
                        self.seed_process.terminate()
                        self.seed_process.wait(timeout=5.0)
                        bt.logging.info(f"Seed service exited with code {self.seed_process.returncode}")
                    except subprocess.TimeoutExpired:
                        bt.logging.warning("Seed service did not exit gracefully, killing...")
                        self.seed_process.kill()
                bt.logging.info("Seed service cleanup complete")

            except Exception as e:
                bt.logging.error(f"Error during validator cleanup: {e}")
                bt.logging.error(traceback.format_exc())

        def cleanup_executors(self):
            """
            Shuts down thread and process executors used by the validator.

            Executors cleaned:
                - reward_executor (ProcessPoolExecutor)
                - save_state_executor (ThreadPoolExecutor)
                - maintenance_executor (ThreadPoolExecutor)
                - multiprocessing manager (if present)

            Behavior:
                - Each executor is shut down gracefully with wait=True
                - For ProcessPoolExecutor, attempts graceful shutdown first
                - Falls back to immediate termination if graceful fails
                - Logs success or failure for each executor

            Returns:
                None
            """
            if hasattr(self, 'reward_executor') and self.reward_executor is not None:
                try:
                    bt.logging.info("Shutting down reward_executor...")
                    self.reward_executor.shutdown(wait=True, cancel_futures=False)
                    bt.logging.info("reward_executor shut down successfully")
                except Exception as ex:
                    bt.logging.error(f"Error shutting down reward_executor: {ex}")
                    try:
                        bt.logging.warning("Attempting to terminate reward_executor processes...")
                        for process in self.reward_executor._processes.values():
                            if process.is_alive():
                                process.terminate()
                                process.join(timeout=2.0)
                                if process.is_alive():
                                    process.kill()
                        bt.logging.info("reward_executor processes terminated")
                    except Exception as term_ex:
                        bt.logging.error(f"Error terminating reward_executor: {term_ex}")

            if hasattr(self, 'query_ipc_executor') and self.query_ipc_executor is not None:
                try:
                    bt.logging.info("Shutting down query_ipc_executor...")
                    self.query_ipc_executor.shutdown(wait=True, cancel_futures=False)
                    bt.logging.info("query_ipc_executor shut down successfully")
                except Exception as ex:
                    bt.logging.error(f"Error shutting down query_ipc_executor: {ex}")

            if hasattr(self, 'reporting_ipc_executor') and self.reporting_ipc_executor is not None:
                try:
                    bt.logging.info("Shutting down reporting_ipc_executor...")
                    self.reporting_ipc_executor.shutdown(wait=True, cancel_futures=False)
                    bt.logging.info("reporting_ipc_executor shut down successfully")
                except Exception as ex:
                    bt.logging.error(f"Error shutting down reporting_ipc_executor: {ex}")

            thread_executors = {
                'save_state_executor': getattr(self, 'save_state_executor', None),
                'maintenance_executor': getattr(self, 'maintenance_executor', None),
            }

            for name, executor in thread_executors.items():
                if executor is not None:
                    try:
                        bt.logging.info(f"Shutting down {name}...")
                        executor.shutdown(wait=True, cancel_futures=False)
                        bt.logging.info(f"{name} shut down successfully")
                    except Exception as ex:
                        bt.logging.error(f"Error shutting down {name}: {ex}")

            if hasattr(self, 'manager'):
                try:
                    bt.logging.info("Shutting down multiprocessing manager...")
                    self.manager.shutdown()
                    bt.logging.info("Manager shut down successfully")
                except Exception as ex:
                    bt.logging.error(f"Error shutting down manager: {ex}")

            bt.logging.info("Executor cleanup complete")

        def cleanup_event_loop(self):
            """
            Gracefully shuts down the main event loop and any pending tasks.

            Behavior:
                - Cancels all pending tasks in the main loop
                - Waits for task cancellation to complete
                - Stops the event loop if still running
                - Closes the event loop

            Returns:
                None
            """
            try:
                if hasattr(self, 'main_loop') and self.main_loop and not self.main_loop.is_closed():
                    bt.logging.info("Shutting down main event loop...")

                    pending = asyncio.all_tasks(self.main_loop)
                    if pending:
                        bt.logging.info(f"Cancelling {len(pending)} pending tasks...")
                        for task in pending:
                            task.cancel()

                        self.main_loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )

                    if self.main_loop.is_running():
                        self.main_loop.stop()

                    self.main_loop.close()
                    bt.logging.info("Main event loop shut down successfully")
            except Exception as ex:
                bt.logging.error(f"Error shutting down main event loop: {ex}")
                bt.logging.error(traceback.format_exc())

        def cleanup(self):
            """
            Performs full resource cleanup for the validator during shutdown.
            """
            if self._cleanup_done:
                bt.logging.debug("Cleanup already completed, skipping")
                return

            bt.logging.info("Starting validator cleanup...")
            self._cleanup_done = True

            try:
                bt.logging.info("Waiting for active operations to complete...")
                wait_timeout = 30.0
                wait_start = time.time()

                while (self.shared_state_rewarding or
                    self.shared_state_saving or
                    self.shared_state_reporting or
                    self.maintaining or
                    self.compressing or
                    self.querying):

                    elapsed = time.time() - wait_start
                    if elapsed > wait_timeout:
                        bt.logging.warning(
                            f"Timeout waiting for operations after {elapsed:.2f}s"
                        )
                        break
                    time.sleep(0.1)

                self.cleanup_executors()
                self.cleanup_ipc()
                self.cleanup_event_loop()

                bt.logging.success("Validator cleanup completed successfully")

            except Exception as ex:
                bt.logging.error(f"Error during cleanup: {ex}")
                bt.logging.error(traceback.format_exc())


if __name__ == "__main__":
    from taos.im.validator.update import check_repo, update_validator, check_simulator, rebuild_simulator, restart_simulator
    from taos.im.validator.forward import forward, notify
    from taos.im.validator.reward import get_rewards

    if float(platform.freedesktop_os_release()['VERSION_ID']) < 22.04:
        raise Exception(f"taos validator requires Ubuntu >= 22.04!")

    bt.logging.info("Initializing validator...")
    app = FastAPI()
    validator = Validator()
    try:
        app.include_router(validator.router)

        bt.logging.info("Starting background threads...")
        threads = []
        for name, target in [('Monitor', validator.monitor), ('Listen', validator.listen)]:
            try:
                bt.logging.info(f"Starting {name} thread...")
                thread = Thread(target=target, daemon=True, name=name)
                thread.start()
                threads.append(thread)
            except Exception as ex:
                validator.pagerduty_alert(f"Exception starting {name} thread: {ex}")
                raise

        time.sleep(1)
        for thread in threads:
            if not thread.is_alive():
                validator.pagerduty_alert(f"Failed to start {thread.name} thread!")
                raise RuntimeError(f"Thread '{thread.name}' failed to start")

        bt.logging.info(f"All threads running. Starting FastAPI server and main event loop...")

        def run_main_loop():
            """Run the pre-created main event loop."""
            async def keep_alive():
                bt.logging.info(f"Main event loop started for background tasks")
                bt.logging.debug(f"[MAINLOOP] Thread: {threading.current_thread().name}")
                bt.logging.debug(f"[MAINLOOP] Loop: {id(validator.main_loop)}")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    bt.logging.info("Main event loop stopping...")
            loop = validator.main_loop
            asyncio.set_event_loop(loop)
            validator._main_loop_ready.set()
            bt.logging.debug(f"[MAINLOOP] Running loop: {id(loop)}")
            try:
                loop.run_until_complete(keep_alive())
            finally:
                loop.close()

        main_loop_thread = Thread(target=run_main_loop, daemon=True, name='main')
        main_loop_thread.start()
        threads.append(main_loop_thread)
        time.sleep(0.5)
        bt.logging.info(f"Starting FastAPI server on port {validator.config.port}...")
        uvicorn.run(app, host="0.0.0.0", port=validator.config.port)
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt received")
    except Exception as ex:
        bt.logging.error(f"Fatal error: {ex}")
        bt.logging.debug(traceback.format_exc())
        sys.exit(1)