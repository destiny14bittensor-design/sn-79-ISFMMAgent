# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Validator and simulation state persistence: atomic msgpack serialisation to
disk, designed for use in a ProcessPoolExecutor worker.
"""
import os
import time
import traceback
from typing import Dict
import msgpack

def save_state_worker(simulation_state_data: Dict, validator_state_data: Dict, 
                     simulation_state_file: str, validator_state_file: str) -> Dict:
    """
    Worker function for saving validator and simulation state to disk - picklable for ProcessPoolExecutor.
    
    Args:
        simulation_state_data (Dict): Dictionary containing simulation state to save
        validator_state_data (Dict): Dictionary containing validator state to save
        simulation_state_file (str): Path to save the simulation state file
        validator_state_file (str): Path to save the validator state file
        
    Returns:
        Dict: Result with success status and timing information
    """
    import msgpack
    import os
    import time
    import traceback
    
    result = {
        'success': False,
        'error': None,
        'simulation_save_time': 0,
        'validator_save_time': 0,
        'total_time': 0
    }
    
    total_start = time.time()
    
    try:
        sim_start = time.time()
        packed_data = msgpack.packb(simulation_state_data, use_bin_type=True)
        
        with open(simulation_state_file + ".tmp", "wb") as file:
            file.write(packed_data)        
        if os.path.exists(simulation_state_file):
            os.remove(simulation_state_file)
        os.rename(simulation_state_file + ".tmp", simulation_state_file)
        
        result['simulation_save_time'] = time.time() - sim_start
        val_start = time.time()
        packed_data = msgpack.packb(validator_state_data, use_bin_type=True)
        
        with open(validator_state_file + ".tmp", "wb") as file:
            file.write(packed_data)
        
        if os.path.exists(validator_state_file):
            os.remove(validator_state_file)
        os.rename(validator_state_file + ".tmp", validator_state_file)
        
        result['validator_save_time'] = time.time() - val_start
        result['total_time'] = time.time() - total_start
        result['success'] = True
        
    except Exception as ex:
        result['error'] = str(ex)
        result['traceback'] = traceback.format_exc()
        for tmp in [simulation_state_file + ".tmp", validator_state_file + ".tmp"]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except:
                    pass    
    return result