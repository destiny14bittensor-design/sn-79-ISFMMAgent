# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
CPU core allocation: distributes logical cores across validator sub-processes
(validator, query, reward, reporting, IPC, and optional gradient server).
"""
import os
import multiprocessing

def get_core_allocation(grad_server_cores: int = 0):
    """
    Allocate CPU cores across validator components using percentage-based allocation.

    Reserves the last `grad_server_cores` logical cores for the gradient server
    and distributes the remainder across the validator, query, reward, reporting,
    and IPC sub-processes.

    Args:
        grad_server_cores (int): Number of logical cores to reserve for the gradient
            server process. Defaults to 0 (no reservation).

    Returns:
        dict: Mapping of component name to list of core indices. Keys are 'validator',
            'query', 'reward', 'reporting', 'ipc', and optionally 'gradient_server'.

    Raises:
        Exception: If fewer than 8 cores remain after the gradient server reservation.
    """
    total_cores = multiprocessing.cpu_count()
    available_cores = total_cores - grad_server_cores
    grad_cores = list(range(available_cores, total_cores)) if grad_server_cores > 0 else []

    if available_cores < 8:
        raise Exception(
            f"Validator requires a minimum of 8 cores to run! "
            f"(total={total_cores}, grad_server_cores={grad_server_cores}, available={available_cores})"
        )

    if available_cores == 8:
        result = {
            'validator': [0, 1],
            'query': [2, 3],
            'reward': [4, 5],
            'reporting': [6],
            'ipc': [7],
        }
        if grad_cores:
            result['gradient_server'] = grad_cores
        return result

    validator_pct = 0.20
    query_pct = 0.20
    reward_pct = 0.25
    reporting_pct = 0.1
    ipc_pct = 0.15

    validator_count = max(2, int(available_cores * validator_pct))
    query_count = max(2, int(available_cores * query_pct))
    reward_count = max(2, int(available_cores * reward_pct))
    reporting_count = max(1, int(available_cores * reporting_pct))
    ipc_count = max(2, int(available_cores * ipc_pct))

    allocated = validator_count + query_count + reward_count + reporting_count + ipc_count
    if allocated > available_cores:
        scale = available_cores / allocated
        validator_count = max(2, int(validator_count * scale))
        query_count = max(2, int(query_count * scale))
        reward_count = max(2, int(reward_count * scale))
        reporting_count = max(1, int(reporting_count * scale))
        ipc_count = max(2, int(ipc_count * scale))

    offset = 0

    validator_cores = list(range(offset, offset + validator_count))
    offset += validator_count

    query_cores = list(range(offset, offset + query_count))
    offset += query_count

    reward_cores = list(range(offset, offset + reward_count))
    offset += reward_count

    reporting_cores = list(range(offset, min(available_cores, offset + reporting_count)))
    offset = min(available_cores, offset + reporting_count)

    ipc_cores = list(range(offset, min(available_cores, offset + ipc_count)))

    result = {
        'validator': validator_cores,
        'query': query_cores,
        'reward': reward_cores,
        'reporting': reporting_cores,
        'ipc': ipc_cores,
    }
    if grad_cores:
        result['gradient_server'] = grad_cores
    return result