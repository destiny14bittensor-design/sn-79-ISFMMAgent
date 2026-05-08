# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Legacy Sharpe ratio calculation: unrealized (mark-to-market) and realized
(round-trip P&L) Sharpe per miner, with parallel batch processing via loky.
"""
import os
import numpy as np
import traceback
from functools import partial
from loky.backend.context import set_start_method
set_start_method('forkserver', force=True)
from loky import get_reusable_executor

from taos.im.utils import normalize


def sharpe(uid, inventory_values, realized_pnl_values, lookback, norm_min, norm_max, 
           min_lookback, min_realized_observations, grace_period, deregistered_uids) -> dict:
    """
    Calculates both unrealized and realized Sharpe ratios.
    
    Unrealized: Based on inventory value changes (mark-to-market)
    Realized: Based on actual P&L from completed round-trip trades
    
    Both use the SAME observation window and timestamps for consistency.
    
    Args:
        uid: Miner UID
        inventory_values: Dict of {timestamp: {book_id: inventory_value}}
        realized_pnl_values: Dict of {timestamp: {book_id: realized_pnl}}
        lookback: Number of periods to look back
        norm_min: Minimum value for normalization
        norm_max: Maximum value for normalization
        min_lookback: Minimum required periods for valid unrealized Sharpe calculation
        min_realized_observations: Minimum required non-zero trades for valid realized Sharpe calculation
        grace_period: Time threshold for detecting simulation changeovers
        deregistered_uids: List of UIDs that are deregistered
        
    Returns:
        Dict containing unrealized and realized Sharpe metrics, or None on error
    """
    try:
        num_values = len(inventory_values)
        if uid in deregistered_uids or num_values < min(min_lookback, lookback):
            return None
        
        timestamps = list(inventory_values.keys())
        book_ids = list(next(iter(inventory_values.values())).keys())
        num_books = len(book_ids)
        
        # ===== UNREALIZED SHARPE =====
        np_inventory_values = np.zeros((num_books, num_values), dtype=np.float64)
        for i, ts in enumerate(timestamps):
            ts_values = inventory_values[ts]
            for j, book_id in enumerate(book_ids):
                np_inventory_values[j, i] = ts_values[book_id]
        
        # Detect changeover periods (simulation restarts)
        changeover_mask = None
        if grace_period > 0:
            ts_array = np.array(timestamps, dtype=np.int64)
            time_diffs = np.diff(ts_array)
            changeover_indices = np.where(time_diffs >= grace_period)[0]
            
            if len(changeover_indices) > 0:
                changeover_mask = np.ones(len(timestamps) - 1, dtype=bool)
                changeover_mask[changeover_indices] = False
        
        # Calculate unrealized returns (period-over-period changes)
        returns = np.diff(np_inventory_values, axis=1)
        
        # Apply changeover mask to exclude restart periods
        if changeover_mask is not None:
            returns = returns[:, changeover_mask]
        
        # Vectorized unrealized Sharpe calculation: sqrt(n) * (mean / std)
        means = returns.mean(axis=1)
        stds = returns.std(axis=1, ddof=0)
        sharpe_ratios = np.divide(means, stds, out=np.zeros_like(means), where=(stds != 0.0))
        
        # ===== REALIZED SHARPE =====
        sharpe_ratios_realized = np.full(num_books, np.nan)

        if realized_pnl_values and len(realized_pnl_values) > 0:
            np_realized_pnl = np.zeros((num_books, num_values), dtype=np.float64)
            for i, ts in enumerate(timestamps):
                pnl_at_ts = realized_pnl_values.get(ts, {})
                for j, book_id in enumerate(book_ids):
                    np_realized_pnl[j, i] = pnl_at_ts.get(book_id, 0.0)
            
            # Drop first timestamp to align with returns (after diff)
            realized_returns = np_realized_pnl[:, 1:]
            
            if changeover_mask is not None:
                realized_returns = realized_returns[:, changeover_mask]
            
            # Vectorized realized Sharpe calculation (per book)
            non_zero_counts = np.count_nonzero(realized_returns, axis=1)
            sufficient_mask = non_zero_counts >= min_realized_observations
            
            if np.any(sufficient_mask):
                realized_means = realized_returns.mean(axis=1)
                realized_stds = realized_returns.std(axis=1, ddof=0)
                
                # Use ALL returns (including zeros) for Sharpe calculation
                # This treats each timestamp as an observation period
                valid_mask = sufficient_mask & (realized_stds != 0.0)
                sharpe_ratios_realized[valid_mask] = realized_means[valid_mask] / realized_stds[valid_mask]
                
                # Zero std means constant returns (all zeros or all same value)
                # If mean is positive, perfect consistency; if zero, no activity
                zero_std_mask = sufficient_mask & (realized_stds == 0.0)
                sharpe_ratios_realized[zero_std_mask] = np.where(
                    realized_means[zero_std_mask] == 0.0, np.nan, 0.0
                )
        
        sharpe_values = {
            'books': {book_ids[i]: float(sharpe_ratios[i]) for i in range(num_books)},
            'books_realized': {
                book_ids[i]: (float(sharpe_ratios_realized[i]) if not np.isnan(sharpe_ratios_realized[i]) else None)
                for i in range(num_books)
            }
        }
        
        # Aggregate values across books (only for unrealized)
        sharpe_values['average'] = float(sharpe_ratios.mean())
        sharpe_values['median'] = float(np.median(sharpe_ratios))
        
        # Aggregate realized values (only if we have valid data)
        valid_realized = sharpe_ratios_realized[~np.isnan(sharpe_ratios_realized)]
        if len(valid_realized) > 0:
            sharpe_values['average_realized'] = float(valid_realized.mean())
            sharpe_values['median_realized'] = float(np.median(valid_realized))
        else:
            sharpe_values['average_realized'] = None
            sharpe_values['median_realized'] = None
        
        # ===== TOTAL PORTFOLIO SHARPE (UNREALIZED) =====
        total_inventory = np_inventory_values.sum(axis=0)
        total_returns = np.diff(total_inventory)
        
        if changeover_mask is not None:
            total_returns = total_returns[changeover_mask]
        
        total_std = total_returns.std(ddof=0)
        total_mean = total_returns.mean()
        sharpe_values['total'] = float(total_mean / total_std if total_std != 0.0 else 0.0)
        
        # ===== TOTAL PORTFOLIO SHARPE (REALIZED) =====
        if realized_pnl_values and len(realized_pnl_values) > 0:
            total_realized_pnl = np_realized_pnl.sum(axis=0)[1:]
            if changeover_mask is not None:
                total_realized_pnl = total_realized_pnl[changeover_mask]
            
            non_zero_total = total_realized_pnl[total_realized_pnl != 0.0]
            count_multiplier = min(len(non_zero_total) / min_realized_observations, 1.0)
            if len(non_zero_total) > 0:
                realized_total_std = non_zero_total.std(ddof=0)
                realized_total_mean = non_zero_total.mean()
                sharpe_values['total_realized'] = count_multiplier * float(realized_total_mean / realized_total_std if realized_total_std != 0.0 else 0.0)
            else:
                sharpe_values['total_realized'] = None  # No round trips
        else:
            sharpe_values['total_realized'] = None  # No realized P&L data
        
        # ===== NORMALIZE ALL VALUES =====
        sharpe_values['normalized_average'] = normalize(norm_min, norm_max, sharpe_values['average'])
        sharpe_values['normalized_median'] = normalize(norm_min, norm_max, sharpe_values['median'])
        sharpe_values['normalized_total'] = normalize(norm_min, norm_max, sharpe_values['total'])
        
        # Normalize realized values (only if defined)
        sharpe_values['normalized_average_realized'] = (
            normalize(norm_min, norm_max, sharpe_values['average_realized'])
        )
        sharpe_values['normalized_median_realized'] = (
            normalize(norm_min, norm_max, sharpe_values['median_realized'])
        )
        sharpe_values['normalized_total_realized'] = (
            normalize(norm_min, norm_max, sharpe_values['total_realized'])
        )
        
        return sharpe_values
        
    except Exception as ex:
        print(f"Failed to calculate Sharpe for UID {uid}: {traceback.format_exc()}")
        return None


def sharpe_batch(inventory_values, realized_pnl_values, lookback, norm_min, norm_max, 
                 min_lookback, min_realized_observations, grace_period, deregistered_uids):
    """
    Process a batch of UIDs for Sharpe calculation with realized P&L.
    
    Args:
        inventory_values: Dict of {uid: {timestamp: {book_id: value}}}
        realized_pnl_values: Dict of {uid: {timestamp: {book_id: pnl}}}
        lookback: Number of periods to look back
        norm_min: Minimum value for normalization
        norm_max: Maximum value for normalization
        min_lookback: Minimum required periods for unrealized Sharpe
        min_realized_observations: Minimum required non-zero trades for realized Sharpe
        grace_period: Time threshold for changeover detection
        deregistered_uids: List of deregistered UIDs
        
    Returns:
        Dict of {uid: sharpe_values}
    """
    return {
        uid: sharpe(uid, inventory_value, realized_pnl_values.get(uid, {}), 
                   lookback, norm_min, norm_max, min_lookback, min_realized_observations, 
                   grace_period, deregistered_uids) 
        for uid, inventory_value in inventory_values.items()
    }


def _init_worker_affinity(cores):
    """
    Worker initializer that sets CPU affinity.
    Must be at module level for pickling.
    
    Args:
        cores: List of CPU cores to bind to
    """
    if cores is not None:
        try:
            os.sched_setaffinity(0, set(cores))
        except (AttributeError, OSError):
            pass


def batch_sharpe(inventory_values, realized_pnl_values, batches, lookback, norm_min, norm_max, 
                 min_lookback, min_realized_observations, grace_period, deregistered_uids, cores=None):
    """
    Parallel processing of Sharpe calculations with realized P&L.
    
    Uses loky for process-based parallelism to avoid GIL limitations
    during NumPy computations.
    
    Args:
        inventory_values: Dict of {uid: {timestamp: {book_id: value}}}
        realized_pnl_values: Dict of {uid: {timestamp: {book_id: pnl}}}
        batches: List of UID batches for parallel processing
        lookback: Number of periods to look back
        norm_min: Minimum value for normalization
        norm_max: Maximum value for normalization
        min_lookback: Minimum required periods for unrealized Sharpe
        min_realized_observations: Minimum required non-zero trades for realized Sharpe
        grace_period: Time threshold for changeover detection
        deregistered_uids: List of deregistered UIDs
        cores: Optional list of CPU cores for worker affinity
        
    Returns:
        Dict of {uid: sharpe_values} for all UIDs
    """
    if cores is not None:
        initializer = partial(_init_worker_affinity, cores)
    else:
        initializer = None
    pool = get_reusable_executor(
        max_workers=len(batches),
        initializer=initializer,
        timeout=300  # Workers timeout after 5 min idle
    )
    
    tasks = [
        pool.submit(
            sharpe_batch,
            {uid: inventory_values[uid] for uid in batch},
            {uid: realized_pnl_values.get(uid, {}) for uid in batch},
            lookback, norm_min, norm_max, min_lookback, min_realized_observations,
            grace_period, deregistered_uids
        )
        for batch in batches
    ]
    
    result = {}
    for task in tasks:
        batch_result = task.result()
        for k, v in batch_result.items():
            result[int(k)] = v
    
    return result