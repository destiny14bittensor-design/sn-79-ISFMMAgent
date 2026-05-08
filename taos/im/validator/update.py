# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Validator self-update helpers: git repo checking, Python package reinstall,
C++ simulator rebuild, and simulator process lifecycle management.
"""
import time
import traceback
import json

# Bittensor
import bittensor as bt

from typing import Tuple

import subprocess
import psutil

from taos.common.utils.misc import run_process
from taos.im.neurons.validator import Validator
        
def check_repo(self : Validator) -> Tuple[bool, bool, bool, bool]:
    """
    Check the git repository for updates with timeout protection.

    Args:
        self (Validator): The intelligent markets simulation validator.

    Returns:
        Tuple[bool, bool, bool, bool]: (validator_py_changed, simulator_config_changed,
            simulator_py_changed, simulator_cpp_changed). All False on error.
    """
    check_timeout = 30.0
    try:
        bt.logging.info("Checking repo for updates...")
        start_time = time.time()
        remote = self.repo.remotes[self.config.repo.remote]
        fetch_start = time.time()
        fetch = remote.fetch(self.repo.active_branch.name)
        fetch_time = time.time() - fetch_start
        if fetch_time > 10.0:
            bt.logging.warning(f"Git fetch took {fetch_time:.1f}s (slow network?)")
        local_commit = self.repo.head.commit
        remote_commit = remote.refs[self.repo.active_branch.name].commit
        validator_py_files_changed = False
        simulator_config_changed = False
        simulator_py_files_changed = False
        simulator_cpp_files_changed = False
        if local_commit != remote_commit:
            diff_start = time.time()
            diff = remote_commit.diff(local_commit)
            for cht in diff.change_type:
                changes = list(diff.iter_change_type(cht))
                for c in changes:
                    if str(self.repo_path / c.b_path) == self.simulator_config_file:
                        simulator_config_changed = True
                    if c.b_path.endswith('.cpp'):
                        simulator_cpp_files_changed = True
                    if c.b_path.endswith('.py'):
                        if 'simulate/trading' in c.b_path:
                            simulator_py_files_changed = True
                        else:
                            validator_py_files_changed = True
            diff_time = time.time() - diff_start
            bt.logging.debug(f"Git diff processed in {diff_time:.1f}s")
        total_time = time.time() - start_time
        if total_time > check_timeout:
            bt.logging.warning(f"Repo check took {total_time:.1f}s (timeout threshold: {check_timeout}s)")
        if not any([validator_py_files_changed, simulator_config_changed, 
                   simulator_py_files_changed, simulator_cpp_files_changed]):
            bt.logging.info(f"Nothing to update (checked in {total_time:.1f}s)")
        else:
            bt.logging.info(
                f"Changes to pull (checked in {total_time:.1f}s): "
                f"[{validator_py_files_changed=}, {simulator_config_changed=}, "
                f"{simulator_py_files_changed=}, {simulator_cpp_files_changed=}]"
            )
        return (validator_py_files_changed, simulator_config_changed, 
                simulator_py_files_changed, simulator_cpp_files_changed)
    except Exception as ex:
        bt.logging.error(f"Failed to check repo: {ex}")
        bt.logging.error(traceback.format_exc())
        self.pagerduty_alert(
            f"Failed to check repo: {ex}", 
            details={"traceback": traceback.format_exc()}
        )
        return False, False, False, False

def update_validator(self : Validator) -> None:
    """
    Pull the latest code, reinstall the package, and restart the validator process.

    Attempts a PM2-managed restart first; falls back to killing the Python process
    directly if PM2 is not available.

    Args:
        self (Validator): The intelligent markets simulation validator.

    Raises:
        subprocess.TimeoutExpired: If the restart command exceeds 30 s.
        Exception: On any other pip install or process-management failure.
    """
    try:
        py_cmd = ["pip", "install", "-e", "."]
        bt.logging.info(f"UPDATING VALIDATOR (PY)...")
        
        update_start = time.time()
        make = run_process(py_cmd, cwd=(self.repo_path).resolve())
        update_time = time.time() - update_start
        
        if make.returncode == 0:
            bt.logging.success(f"VALIDATOR PY UPDATE SUCCESSFUL ({update_time:.1f}s). RESTARTING...")
        else:
            raise Exception(f"FAILED TO COMPLETE VALIDATOR PY UPDATE:\n{make.stderr}")
        try:
            pm2_result = subprocess.run(
                ['pm2', 'jlist'], 
                capture_output=True, 
                text=True, 
                timeout=10.0
            )
            pm2_json = pm2_result.stdout
            pm2_js = json.loads(pm2_json) if pm2_json else []
        except subprocess.TimeoutExpired:
            bt.logging.warning("PM2 jlist timed out after 10s")
            pm2_js = []
        except json.JSONDecodeError as e:
            bt.logging.warning(f"Failed to parse PM2 JSON: {e}")
            pm2_js = []
        
        restart_cmd = None
        if len(pm2_js) > 0:
            pm2_processes = {p['name']: p for p in pm2_js}
            if 'validator' in pm2_processes:
                bt.logging.info("FOUND VALIDATOR IN pm2 PROCESSES.")
                restart_cmd = ["pm2", "restart", "validator"]
        
        if not restart_cmd:
            killed_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'python validator.py' in cmdline:
                        bt.logging.info(f"FOUND VALIDATOR PROCESS `{proc.info['name']}` WITH PID {proc.info['pid']}")
                        proc.kill()
                        proc.wait(timeout=5.0)
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.TimeoutExpired) as e:
                    bt.logging.warning(f"Error killing process: {e}")
            
            if killed_count > 0:
                bt.logging.info(f"Killed {killed_count} validator process(es)")
                time.sleep(2.0)  # Brief pause after kill
            
            restart_cmd = [
                "pm2", "start", "--name=validator",
                f"python validator.py --netuid {self.config.netuid} "
                f"--subtensor.chain_endpoint {self.config.subtensor.chain_endpoint} "
                f"--wallet.path {self.config.wallet.path} "
                f"--wallet.name {self.config.wallet.name} "
                f"--wallet.hotkey {self.config.wallet.hotkey}"
            ]
        
        bt.logging.info(f"RESTARTING VALIDATOR: {' '.join(restart_cmd)}")
        validator = subprocess.run(
            restart_cmd, 
            cwd=str((self.repo_path / 'taos' / 'im' / 'neurons').resolve()), 
            shell=isinstance(restart_cmd, str),
            capture_output=True,
            timeout=30.0
        )
        
        if validator.returncode != 0:
            bt.logging.warning(
                f"Validator restart returned code {validator.returncode}\n"
                f"STDOUT: {validator.stdout}\nSTDERR: {validator.stderr}"
            )
        
        return
        
    except subprocess.TimeoutExpired:
        bt.logging.error("Validator restart command timed out after 30s")
        self.pagerduty_alert("Validator restart timeout")
        raise
    except Exception as ex:
        bt.logging.error(f"Failed to update validator: {ex}")
        bt.logging.error(traceback.format_exc())
        self.pagerduty_alert(
            f"Failed to update validator: {ex}",
            details={"traceback": traceback.format_exc()}
        )
        raise

def rebuild_simulator(self : Validator) -> None:
    """
    Recompile the C++ simulator and reinstall its Python bindings.

    Runs cmake + cmake --build using g++-14, then reinstalls the Python
    package from simulate/trading.

    Args:
        self (Validator): The intelligent markets simulation validator.

    Raises:
        subprocess.TimeoutExpired: If any subprocess exceeds its timeout.
        Exception: On cmake, build, or pip install failures.
    """
    try:
        gcc_version_proc = subprocess.run(
            ['g++', '-dumpversion'], 
            capture_output=True, 
            timeout=5.0
        )
        
        if gcc_version_proc.returncode == 0:
            gcc_version = gcc_version_proc.stdout.decode().strip()
            gcc14_check_proc = subprocess.run(
                ['g++-14', '-dumpversion'], 
                capture_output=True, 
                timeout=5.0
            )
            if gcc14_check_proc.returncode != 0:
                raise Exception(
                    f"Could not find g++-14 on system: "
                    f"{gcc14_check_proc.stderr.decode()}"
                )
        else:
            raise Exception(
                f"Could not find g++ version: "
                f"{gcc_version_proc.stderr.decode()}"
            )
        
        if gcc_version != '14':
            make_cmd = [
                "cmake", "-DENABLE_TRACES=1", "-DCMAKE_BUILD_TYPE=Release", 
                "..", "-D", "CMAKE_CXX_COMPILER=g++-14"
            ]
        else:
            make_cmd = [
                "cmake", "-DENABLE_TRACES=1", "-DCMAKE_BUILD_TYPE=Release", ".."
            ]

        bt.logging.info(f"REBUILDING SIMULATOR (MAKE)...")
        make_start = time.time()
        make = run_process(
            make_cmd, 
            (self.repo_path / 'simulate' / 'trading' / 'build').resolve()
        )
        make_time = time.time() - make_start
        
        if make.returncode == 0:
            bt.logging.success(f"MAKE PROCESS SUCCESSFUL ({make_time:.1f}s). BUILDING...")
            
            build_cmd = ["cmake", "--build", "."]
            bt.logging.info(f"REBUILDING SIMULATOR (BUILD)...")
            build_start = time.time()
            build = run_process(
                build_cmd, 
                cwd=(self.repo_path / 'simulate' / 'trading' / 'build').resolve()
            )
            build_time = time.time() - build_start
            
            if build.returncode == 0:
                bt.logging.success(f"REBUILT SIMULATOR ({build_time:.1f}s).")
            else:
                raise Exception(
                    f"FAILED TO COMPLETE SIMULATOR BUILD:\n{build.stderr}"
                )
        else:
            raise Exception(f"FAILED TO COMPLETE SIMULATOR MAKE:\n{make.stderr}")

        py_cmd = ["pip", "install", "-e", "."]
        bt.logging.info(f"REBUILDING SIMULATOR (PY)...")
        py_start = time.time()
        py = run_process(
            py_cmd, 
            cwd=(self.repo_path / 'simulate' / 'trading').resolve()
        )
        py_time = time.time() - py_start
        
        if py.returncode == 0:
            bt.logging.success(f"PY UPDATE SUCCESSFUL ({py_time:.1f}s).")
        else:
            raise Exception(
                f"FAILED TO COMPLETE SIMULATOR PY UPDATE:\n{py.stderr}"
            )
            
    except subprocess.TimeoutExpired as e:
        bt.logging.error(f"Simulator rebuild timeout: {e}")
        self.pagerduty_alert(f"Simulator rebuild timeout: {e}")
        raise
    except Exception as ex:
        bt.logging.error(f"Failed to rebuild simulator: {ex}")
        bt.logging.error(traceback.format_exc())
        self.pagerduty_alert(
            f"Failed to rebuild simulator: {ex}",
            details={"traceback": traceback.format_exc()}
        )
        raise

def restart_simulator(self : Validator, end : bool = False) -> None:
    """
    Stop the running simulator and start a new one, optionally resuming from checkpoint.

    Kills any existing PM2 or bare simulator process, then attempts to resume
    from the latest checkpoint unless `end` is True or the checkpoint fails health
    check, in which case a fresh simulation is started from the config file.

    Args:
        self (Validator): The intelligent markets simulation validator.
        end (bool): If True, skip checkpoint resume and start a new simulation.
            Defaults to False.

    Raises:
        subprocess.TimeoutExpired: If any subprocess exceeds its timeout.
        Exception: On any process-management failure.
    """
    try:
        try:
            pm2_result = subprocess.run(
                ['pm2', 'jlist'], 
                capture_output=True, 
                text=True, 
                timeout=10.0
            )
            pm2_json = pm2_result.stdout
            pm2_js = json.loads(pm2_json) if pm2_json else []
        except subprocess.TimeoutExpired:
            bt.logging.warning("PM2 jlist timed out after 10s")
            pm2_js = []
        except json.JSONDecodeError as e:
            bt.logging.warning(f"Failed to parse PM2 JSON: {e}")
            pm2_js = []
        pm2_processes = {p['name']: p for p in pm2_js}
        if 'simulator' in pm2_processes:
            bt.logging.info("STOPPING EXISTING SIMULATOR IN PM2...")
            try:
                subprocess.run(
                    ['pm2', 'delete', 'simulator'],
                    capture_output=True,
                    timeout=10.0
                )
                time.sleep(1.0)
            except subprocess.TimeoutExpired:
                bt.logging.warning("PM2 delete timed out after 10s")

        killed_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if '../build/src/cpp/taosim' in cmdline:
                    bt.logging.info(
                        f"FOUND SIMULATOR PROCESS `{proc.info['name']}` "
                        f"WITH PID {proc.info['pid']}"
                    )
                    proc.kill()
                    proc.wait(timeout=5.0)
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.TimeoutExpired) as e:
                bt.logging.warning(f"Error killing simulator process: {e}")
        
        if killed_count > 0:
            bt.logging.info(f"Killed {killed_count} simulator process(es)")
            time.sleep(2.0)

        if not end:
            resume_cmd = [
                "pm2", "start", "--no-autorestart", "--name=simulator",
                "../build/src/cpp/taosim -c latest"
            ]
            
            bt.logging.info(f"ATTEMPTING TO RESUME SIMULATOR FROM CHECKPOINT: {' '.join(resume_cmd)}")
            resume_result = subprocess.run(
                resume_cmd, 
                cwd=str((self.repo_path / 'simulate' / 'trading' / 'run').resolve()), 
                shell=isinstance(resume_cmd, str),
                capture_output=True,
                timeout=30.0
            )

            checkpoint_success = False
            if resume_result.returncode == 0:
                time.sleep(2.0)
                if check_simulator(self):
                    bt.logging.success("SIMULATOR RESUMED FROM CHECKPOINT SUCCESSFULLY.")
                    checkpoint_success = True
                else:
                    bt.logging.warning("Checkpoint resume started but simulator not healthy. Falling back to new simulation.")
                    try:
                        subprocess.run(['pm2', 'delete', 'simulator'], capture_output=True, timeout=10.0)
                        time.sleep(1.0)
                    except:
                        pass
            else:
                bt.logging.warning(
                    f"Failed to resume from checkpoint (returncode={resume_result.returncode}). "
                    f"Falling back to new simulation.\n"
                    f"STDERR: {resume_result.stderr.decode() if resume_result.stderr else 'N/A'}"
                )
        if end or not checkpoint_success:
            fallback_cmd = [
                "pm2", "start", "--no-autorestart", "--name=simulator",
                f"../build/src/cpp/taosim -f {self.simulator_config_file}"
            ]
            bt.logging.info(f"STARTING NEW SIMULATION: {' '.join(fallback_cmd)}")
            simulator = subprocess.run(
                fallback_cmd, 
                cwd=str((self.repo_path / 'simulate' / 'trading' / 'run').resolve()), 
                shell=isinstance(fallback_cmd, str),
                capture_output=True,
                timeout=30.0
            )
            
            if simulator.returncode == 0:
                time.sleep(2.0)
                if check_simulator(self):
                    bt.logging.success("NEW SIMULATION STARTED SUCCESSFULLY.")
                else:
                    self.pagerduty_alert(
                        "FAILED TO START SIMULATOR! NOT FOUND IN PM2 AFTER RESTART."
                    )
            else:
                raise Exception(
                    f"FAILED TO START NEW SIMULATION:\n"
                    f"STDOUT: {simulator.stdout}\n"
                    f"STDERR: {simulator.stderr}"
                )
            
    except subprocess.TimeoutExpired:
        bt.logging.error("Simulator restart command timed out after 30s")
        self.pagerduty_alert("Simulator restart timeout")
        raise
    except Exception as ex:
        bt.logging.error(f"Failed to restart simulator: {ex}")
        bt.logging.error(traceback.format_exc())
        self.pagerduty_alert(
            f"Failed to restart simulator: {ex}",
            details={"traceback": traceback.format_exc()}
        )
        raise

def check_simulator(self : Validator) -> bool:
    """
    Check whether the simulator process is alive and healthy.

    Returns True immediately if `last_state_time` indicates a recent heartbeat.
    Otherwise queries PM2 for status, then falls back to scanning the process
    table for the taosim binary.

    Args:
        self (Validator): The intelligent markets simulation validator.

    Returns:
        bool: True if the simulator is healthy, False otherwise.
    """
    try:
        if not self.last_state_time or self.last_state_time >= time.time() - 300:
            return True
        try:
            pm2_result = subprocess.run(
                ['pm2', 'jlist'], 
                capture_output=True, 
                text=True, 
                timeout=10.0
            )
            pm2_json = pm2_result.stdout
            pm2_js = json.loads(pm2_json) if pm2_json else []
        except subprocess.TimeoutExpired:
            bt.logging.error("PM2 jlist timed out after 10s during health check")
            return False
        except json.JSONDecodeError as e:
            bt.logging.error(f"Failed to parse PM2 JSON during health check: {e}")
            pm2_js = []
        pm2_processes = {p['name']: p for p in pm2_js}
        if 'simulator' in pm2_processes:
            status = pm2_processes['simulator']['pm2_env']['status']
            if status != 'online':
                self.pagerduty_alert(
                    f"Simulator process (PM2) has stopped! Status={status}"
                )
                return False
            return True
        found = False
        try:
            for proc in psutil.process_iter(['cmdline']):
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if '../build/src/cpp/taosim' in cmdline:
                    found = True
                    break
        except Exception as e:
            bt.logging.error(f"Error checking simulator processes: {e}")
            return False
        
        if not found:
            self.pagerduty_alert("Simulator process (No PM2) has stopped!")
            return False
        
        return True
    except Exception as ex:
        bt.logging.error(f"Error during simulator health check: {ex}")
        bt.logging.error(traceback.format_exc())
        return False