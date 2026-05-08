# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Kappa-3 (lower partial moment) risk-adjusted return calculation: per-UID
realized P&L scoring with parallel batch processing via loky workers.
"""
import os
import numpy as np
import traceback
from functools import partial
from loky.backend.context import set_start_method
set_start_method('forkserver', force=True)
from loky import get_reusable_executor

from taos.im.utils import normalize

def _get_pnl_fingerprint(realized_pnl_values):
    """
    Generate a robust fingerprint of P&L data for cache validation.
    
    Uses multiple independent metrics to ensure uniqueness:
    - Count of non-zero entries
    - Sum of all P&L values
    - Sum of squares (captures magnitude distribution)
    - Min and max values (captures range)
    
    Args:
        realized_pnl_values: Dict of {timestamp: {book_id: pnl}}
        
    Returns:
        Tuple of (count, sum, sum_squares, min_val, max_val)
    """
    if not realized_pnl_values:
        return (0, 0.0, 0.0, 0.0, 0.0)
    
    count = 0
    total = 0.0
    sum_squares = 0.0
    min_val = float('inf')
    max_val = float('-inf')
    
    for books in realized_pnl_values.values():
        for pnl in books.values():
            count += 1
            total += pnl
            sum_squares += pnl * pnl
            min_val = min(min_val, pnl)
            max_val = max(max_val, pnl)

    if count == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    
    return (count, total, sum_squares, min_val, max_val)


def kappa_3(uid, realized_pnl_values, tau, lookback, norm_min, norm_max, 
           min_lookback, min_realized_observations, grace_period, deregistered_uids, book_count,
           cache=None) -> dict:
    """
    Calculates realized Kappa-3 ratio based on actual P&L from completed round-trip trades.
    
    Kappa-3 is defined as: K_3(τ) = (μ - τ) / [LPM_3(τ)]^(1/3)
    where LPM_3(τ) is the third lower partial moment.
    
    For perfect miners (no downside), uses: K_3(τ) = (μ - τ) / [UPM_3(τ)]^(1/3)
    This ensures scale-invariance and consistency with the standard formula.
    
    Args:
        uid: Miner UID
        realized_pnl_values: Dict of {timestamp: {book_id: realized_pnl}}
        tau: Threshold return (minimum acceptable return)
        lookback: Number of periods to look back
        norm_min: Minimum value for normalization
        norm_max: Maximum value for normalization
        min_lookback: Minimum required periods for valid calculation
        min_realized_observations: Minimum required non-zero trades for valid calculation
        grace_period: Time threshold for detecting simulation changeovers
        deregistered_uids: List of UIDs that are deregistered
        book_count: Total number of books in simulation
        cache: Optional dict for caching results {uid: (fingerprint, kappa_values)}
        
    Returns:
        Dict containing realized Kappa-3 metrics, or None on error
    """
    try:
        if cache is not None:
            current_fingerprint = _get_pnl_fingerprint(realized_pnl_values)
            if uid in cache:
                cached_fingerprint, cached_kappa = cache[uid]
                if cached_fingerprint == current_fingerprint:
                    return cached_kappa

        if uid in deregistered_uids or not realized_pnl_values:
            return None
        timestamps = sorted(realized_pnl_values.keys())
        if timestamps[-1] - timestamps[0] < min_lookback:
            return None
        
        num_values = len(timestamps)
        book_ids = list(range(book_count))
        num_books = len(book_ids)
        
        np_realized_pnl = np.zeros((num_books, num_values), dtype=np.float64)
        for i, ts in enumerate(timestamps):
            pnl_at_ts = realized_pnl_values.get(ts, {})
            for j, book_id in enumerate(book_ids):
                np_realized_pnl[j, i] = pnl_at_ts.get(book_id, 0.0)
        
        # Detect changeover periods (simulation restarts)
        changeover_mask = None
        if grace_period > 0:
            ts_array = np.array(timestamps, dtype=np.int64)
            time_diffs = np.diff(ts_array)
            changeover_indices = np.where(time_diffs >= grace_period)[0]
            
            if len(changeover_indices) > 0:
                changeover_mask = np.ones(len(timestamps) - 1, dtype=bool)
                changeover_mask[changeover_indices] = False
        
        # Normalize returns by MAD for scale-invariance
        median_per_book = np.median(np_realized_pnl, axis=1, keepdims=True)
        mad_per_book = np.median(np.abs(np_realized_pnl - median_per_book), axis=1, keepdims=True)
        mad_per_book = np.maximum(mad_per_book, 1e-6)
        realized_returns = np_realized_pnl / mad_per_book
        
        # Apply changeover mask if needed
        if changeover_mask is not None:
            full_mask = np.concatenate([[True], changeover_mask])
            realized_returns = realized_returns[:, full_mask]
        
        # Vectorized realized Kappa-3 calculation (per book)
        non_zero_counts = np.count_nonzero(realized_returns, axis=1)
        sufficient_mask = non_zero_counts >= min_realized_observations
        
        kappa_ratios_realized = np.full(num_books, np.nan)
        if np.any(sufficient_mask):
            realized_means = realized_returns.mean(axis=1)
            realized_downside = np.maximum(tau - realized_returns, 0.0)
            realized_lpm3 = np.power(realized_downside, 3).mean(axis=1)
            realized_upside = np.maximum(realized_returns - tau, 0.0)
            realized_upm3 = np.power(realized_upside, 3).mean(axis=1)
            
            # Data-driven regularization to prevent division by near-zero
            typical_scale = np.abs(realized_means) + np.std(realized_returns, axis=1)
            regularization = np.power(typical_scale * 0.1, 3)
            
            # Adaptive epsilon based on mean direction
            # If mean is positive (winning), be generous with epsilon (ignore tiny losses)
            # If mean is negative (losing), be strict with epsilon (don't ignore real losses)
            epsilon_per_book = np.where(
                realized_means > tau,
                1e-2,
                1e-6
            )
            
            # Standard formula (meaningful downside) with regularization
            valid_mask = sufficient_mask & (realized_lpm3 > epsilon_per_book)
            kappa_ratios_realized[valid_mask] = (
                (realized_means[valid_mask] - tau) / np.cbrt(realized_lpm3[valid_mask] + regularization[valid_mask])
            )
            
            # Perfect formula (negligible downside AND positive mean) with regularization
            perfect_mask = sufficient_mask & (realized_lpm3 <= epsilon_per_book) & (realized_means > tau)
            kappa_ratios_realized[perfect_mask] = (
                (realized_means[perfect_mask] - tau) / np.cbrt(realized_upm3[perfect_mask] + regularization[perfect_mask])
            )
            
            # Zero score (no meaningful downside but negative mean)
            zero_mask = sufficient_mask & (realized_lpm3 <= epsilon_per_book) & (realized_means <= tau)
            kappa_ratios_realized[zero_mask] = 0.0
        
        kappa_values = {
            'books': {
                book_ids[i]: (float(kappa_ratios_realized[i]) if not np.isnan(kappa_ratios_realized[i]) else None)
                for i in range(num_books)
            }
        }
        
        # Aggregate realized values (only if we have valid data)
        valid_realized = kappa_ratios_realized[~np.isnan(kappa_ratios_realized)]
        if len(valid_realized) > 0:
            kappa_values['average'] = float(valid_realized.mean())
            kappa_values['median'] = float(np.median(valid_realized))
        else:
            kappa_values['average'] = None
            kappa_values['median'] = None
        
        # ===== TOTAL PORTFOLIO KAPPA-3 (REALIZED) =====
        total_realized_pnl = np_realized_pnl.sum(axis=0)
        
        if changeover_mask is not None:
            full_mask = np.concatenate([[True], changeover_mask])
            total_realized_pnl = total_realized_pnl[full_mask]
        
        # Normalize portfolio by MAD
        total_median = np.median(total_realized_pnl)
        total_mad = np.median(np.abs(total_realized_pnl - total_median))
        total_mad = max(total_mad, 1e-6)
        total_realized_normalized = total_realized_pnl / total_mad
        
        non_zero_total = total_realized_normalized[total_realized_normalized != 0.0]
        count_multiplier = min(len(non_zero_total) / min_realized_observations, 1.0)
        
        if len(non_zero_total) > 0:
            realized_total_mean = total_realized_normalized.mean()
            realized_total_downside = np.maximum(tau - total_realized_normalized, 0.0)
            realized_total_lpm3 = np.power(realized_total_downside, 3).mean()
            realized_total_upside = np.maximum(total_realized_normalized - tau, 0.0)
            realized_total_upm3 = np.power(realized_total_upside, 3).mean()
            
            # Regularization for portfolio
            total_typical_scale = abs(realized_total_mean) + np.std(total_realized_normalized)
            total_regularization = (total_typical_scale * 0.1) ** 3
            
            # Adaptive epsilon for portfolio
            epsilon_portfolio = 1e-2 if realized_total_mean > tau else 1e-6
            
            if realized_total_lpm3 > epsilon_portfolio:
                kappa_values['total'] = count_multiplier * float(
                    (realized_total_mean - tau) / np.cbrt(realized_total_lpm3 + total_regularization)
                )
            elif realized_total_mean > tau:
                kappa_values['total'] = count_multiplier * float(
                    (realized_total_mean - tau) / np.cbrt(realized_total_upm3 + total_regularization)
                )
            else:
                kappa_values['total'] = count_multiplier * 0.0
        else:
            kappa_values['total'] = None
        
        # Normalize all values
        kappa_values['normalized_average'] = (
            normalize(norm_min, norm_max, kappa_values['average'])
        )
        kappa_values['normalized_median'] = (
            normalize(norm_min, norm_max, kappa_values['median'])
        )
        kappa_values['normalized_total'] = (
            normalize(norm_min, norm_max, kappa_values['total'])
        )

        if cache is not None:
            cache[uid] = (_get_pnl_fingerprint(realized_pnl_values), kappa_values)
        
        return kappa_values
        
    except Exception as ex:
        print(f"Failed to calculate Kappa-3 for UID {uid}: {traceback.format_exc()}")
        return None


def kappa_3_batch(realized_pnl_values, tau, lookback, norm_min, norm_max,
                  min_lookback, min_realized_observations, grace_period, deregistered_uids, book_count,
                  cache=None):
    """
    Process a batch of UIDs for Kappa-3 calculation with realized P&L only.
    
    Returns both results and cache updates for parent process.
    
    Args:
        realized_pnl_values: Dict of {uid: {timestamp: {book_id: pnl}}}
        tau: Threshold return
        lookback: Number of periods to look back
        norm_min: Minimum value for normalization
        norm_max: Maximum value for normalization
        min_lookback: Minimum required periods for unrealized Kappa
        min_realized_observations: Minimum required non-zero trades for valid calculation
        grace_period: Time threshold for changeover detection
        deregistered_uids: List of deregistered UIDs
        book_count: Total number of books in simulation
        cache: Optional cache dict (passed from parent, read-only in worker)
        
    Returns:
        Tuple of (results_dict, cache_updates_dict)
    """
    results = {}
    cache_updates = {}    
    for uid, realized_pnl_value in realized_pnl_values.items():
        kappa_values = kappa_3(
            uid, realized_pnl_value, tau, lookback, norm_min, norm_max,
            min_lookback, min_realized_observations, grace_period, 
            deregistered_uids, book_count, cache=cache
        )
        results[uid] = kappa_values
        fingerprint = _get_pnl_fingerprint(realized_pnl_value)
        cache_updates[uid] = (fingerprint, kappa_values)        
    return results, cache_updates

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

def batch_kappa_3(realized_pnl_values, tau, batches, lookback, norm_min, norm_max,
                  min_lookback, min_realized_observations, grace_period, deregistered_uids, 
                  book_count, cache=None, cores=None):
    """
    Parallel processing of Kappa-3 calculations with realized P&L only.
    
    Returns results and cache updates to avoid Manager overhead.
    
    Args:
        realized_pnl_values: Dict of {uid: {timestamp: {book_id: pnl}}}
        tau: Threshold return
        batches: List of UID batches for parallel processing
        lookback: Number of periods to look back
        norm_min: Minimum value for normalization
        norm_max: Maximum value for normalization
        min_lookback: Minimum required periods for unrealized Kappa
        min_realized_observations: Minimum required non-zero trades for valid calculation
        grace_period: Time threshold for changeover detection
        deregistered_uids: List of deregistered UIDs
        book_count: Total number of books in simulation
        cache: Optional dict for caching (read-only in workers)
        cores: Optional list of CPU cores for worker affinity
        
    Returns:
        Tuple of (results_dict, cache_updates_dict)
    """
    if cores is not None:
        initializer = partial(_init_worker_affinity, cores)
    else:
        initializer = None
    pool = get_reusable_executor(
        max_workers=len(batches),
        initializer=initializer,
        timeout=300
    )
    
    tasks = [
        pool.submit(
            kappa_3_batch,
            {uid: realized_pnl_values.get(uid, {}) for uid in batch},
            tau, lookback, norm_min, norm_max, min_lookback, min_realized_observations,
            grace_period, deregistered_uids, book_count,
            cache=cache
        )
        for batch in batches
    ]
    
    result = {}
    cache_updates = {}
    
    for task in tasks:
        batch_result, batch_cache_updates = task.result()
        for k, v in batch_result.items():
            result[int(k)] = v
        cache_updates.update(batch_cache_updates)
    
    return result, cache_updates