# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""
Per-UID reward computation: Kappa-3, P&L, and GenTRX scores combined into
two-pool (trading + training) reward tensors.
"""

import time
import torch
import random
import bittensor as bt
import numpy as np
from typing import Dict, Tuple
from collections import defaultdict
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.utils import normalize
from taos.im.utils.kappa import kappa_3, batch_kappa_3, _get_pnl_fingerprint

def calculate_kappa_score(
    uid: int,
    kappa_values: Dict,
    activity_factors: Dict,
    pnl_factors: Dict,
    roundtrip_volumes: Dict,
    realized_pnl_history: Dict,
    config: Dict,
    simulation_config: Dict,
    simulation_timestamp: int
) -> float:
    """
    Calculate Kappa-based score with activity and P&L factor weighting.
    
    This function measures risk-adjusted trading quality (Kappa-3 metric) and weights it by:
    1. Trading activity (volume-based): Rewards active participation
    2. Profitability (P&L-based): Boosts scores for profitable trading
    
    The Kappa score component represents "quality of trading WHERE the miner traded."
    Activity and P&L factors are applied per-book to weight the quality measure.
    
    Scoring Flow:
    1. Normalize raw Kappa values per book to [0, 1] range
    2. Calculate volume-based activity factors per book (decay for inactivity)
    3. Calculate P&L-based multipliers per book (if enabled via config)
    4. Combine activity × P&L factors and apply to normalized Kappa per book
    5. Aggregate weighted Kappas across books with outlier penalty
    6. Return final Kappa score in [0, 1] range
    
    Args:
        uid: UID of miner being scored
        kappa_values: Kappa calculation results from kappa_3() {uid: {books: {book_id: kappa_value}}}
        activity_factors: Activity factor storage (updated in-place) {uid: {book_id: factor}}
        pnl_factors: P&L factor storage (updated in-place) {uid: {book_id: factor}}
        roundtrip_volumes: Trading volume history {uid: {book_id: {timestamp: volume}}}
        realized_pnl_history: Realized P&L from completed trades {uid: {timestamp: {book_id: pnl}}}
        config: Scoring configuration dict
        simulation_config: Simulation parameters (miner_wealth, publish_interval, etc)
        simulation_timestamp: Current simulation timestamp in nanoseconds
        
    Returns:
        float: Kappa score in [0, 1] range
    """
    # Early exit if no Kappa data for this miner (also guards new UIDs not yet in the dict)
    if not kappa_values.get(uid):
        return 0.0

    uid_kappa = kappa_values[uid]
    kappas = uid_kappa['books']  # Raw Kappa values per book from kappa_3()

    # ===== STEP 1: NORMALIZE KAPPA VALUES =====
    # Maps raw Kappa values to [0, 1] range for fair comparison
    # Normalization allows combining Kappas from different books on a common scale
    norm_min = config['kappa']['normalization_min']
    norm_max = config['kappa']['normalization_max']
    norm_range = norm_max - norm_min
    norm_range_inv = 1.0 / norm_range if norm_range != 0 else 0.0

    # Normalize kappas per book - keep None for books with insufficient data
    # None indicates the miner hasn't traded enough on that book to calculate reliable Kappa
    normalized_kappas = {book_id: None for book_id in activity_factors.get(uid, {}).keys()}
    
    for book_id, kappa_val in kappas.items():
        if kappa_val is not None:
            # Clamp to [0, 1] range: (kappa - min) / (max - min)
            normalized_kappas[book_id] = max(0.0, min(1.0, (kappa_val - norm_min) * norm_range_inv))

    # ===== STEP 2: CALCULATE ACTIVITY FACTOR PARAMETERS =====
    # Activity factors reward miners for trading volume relative to their capital
    # Volume cap together with impact parameter defines the threshold for maximum activity boost
    volume_cap = round(
        config['activity']['capital_turnover_cap'] * simulation_config['miner_wealth'],
        simulation_config['volumeDecimals']
    )
    volume_cap_inv = 1.0 / volume_cap

    lookback = config['kappa']['lookback']  # Number of intervals to look back
    publish_interval = simulation_config['publish_interval']  # Nanoseconds per interval
    lookback_threshold = simulation_timestamp - (lookback * publish_interval)
    
    # Decay parameters: Activity factors decay for inactive books to incentivize consistent trading
    decay_grace_period = config['activity'].get('decay_grace_period', 600_000_000_000)
    activity_impact = config['activity'].get('impact', 0.33)  # Max boost from activity
    time_acceleration_power = 2.0  # Exponential decay acceleration
    
    # Calculate decay parameters
    # The decay window is the lookback period minus the grace period
    # During grace period: no decay (allows brief pauses without penalty)
    # After grace period: exponential decay kicks in
    scoring_interval_seconds = config['interval'] / 1e9
    total_intervals = lookback // scoring_interval_seconds
    grace_intervals = decay_grace_period / config['interval']
    decay_window_intervals = total_intervals - grace_intervals
    
    # Safety check for decay window
    if decay_window_intervals <= 0:
        bt.logging.warning(f"UID {uid}: Invalid decay window (total={total_intervals}, grace={grace_intervals})")
        base_decay_factor = 0.999  # Fallback to very slow decay
    else:
        # Calculate base decay factor: 2^(-1/decay_window_intervals)
        # This creates exponential decay that reaches ~0.5 after decay_window_intervals periods
        base_decay_factor = 2 ** (-1 / decay_window_intervals)
        base_decay_factor = max(0.5, min(0.9999, base_decay_factor))  # Clamp to safe range
    
    decay_window_ns = (lookback * config['interval']) - decay_grace_period
    decay_window_ns_inv = 1.0 / decay_window_ns if decay_window_ns > 0 else 0.0

    # ===== STEP 3: COMPUTE ROUNDTRIP VOLUMES PER BOOK =====
    # Roundtrip volume = sum of trade sizes for completed buy-sell cycles
    # This measures actual trading activity, not just open positions
    sampling_interval = config['activity']['trade_volume_sampling_interval']
    sampled_timestamp = (simulation_timestamp // sampling_interval) * sampling_interval
    
    # Get number of books from normalized_kappas keys
    num_books = len(normalized_kappas)
    
    # Compute compact roundtrip volumes for this UID
    miner_roundtrip_volumes = {}
    latest_roundtrip_volumes = {}
    latest_roundtrip_timestamps = {}
    
    if uid in roundtrip_volumes:
        uid_rt_volumes = roundtrip_volumes[uid]
        
        for book_id in range(num_books):
            if book_id in uid_rt_volumes:
                rt_volumes = uid_rt_volumes[book_id]
                
                if rt_volumes:
                    lookback_volume = 0.0
                    latest_time = 0
                    latest_volume = 0.0
                    
                    # Sum all volumes within lookback period
                    # Find the most recent trading timestamp
                    for ts, vol in rt_volumes.items():
                        if ts >= lookback_threshold:
                            lookback_volume += vol
                        if vol > 0 and ts <= sampled_timestamp and ts > latest_time:
                            latest_time = ts
                    
                    # Check if there was recent activity (within sampling interval)
                    if latest_time > 0 and latest_time >= sampled_timestamp - sampling_interval:
                        latest_volume = rt_volumes[latest_time]
                    
                    miner_roundtrip_volumes[book_id] = lookback_volume
                    latest_roundtrip_volumes[book_id] = latest_volume
                    latest_roundtrip_timestamps[book_id] = latest_time
                else:
                    # No volume history for this book
                    miner_roundtrip_volumes[book_id] = 0.0
                    latest_roundtrip_volumes[book_id] = 0.0
                    latest_roundtrip_timestamps[book_id] = 0
            else:
                # Book not in roundtrip volumes
                miner_roundtrip_volumes[book_id] = 0.0
                latest_roundtrip_volumes[book_id] = 0.0
                latest_roundtrip_timestamps[book_id] = 0
    else:
        # UID not in roundtrip_volumes, initialize all books to zero
        for book_id in range(num_books):
            miner_roundtrip_volumes[book_id] = 0.0
            latest_roundtrip_volumes[book_id] = 0.0
            latest_roundtrip_timestamps[book_id] = 0

    # ===== STEP 4: CALCULATE VOLUME-BASED ACTIVITY FACTORS =====
    # Activity factors range from 0 to 2.0:
    # - 1.0 = neutral (no boost or penalty)
    # - <1.0 = penalty for inactivity (via exponential decay)
    # - >1.0 = boost for high volume (up to 2.0x at volume_cap)
    activity_factors_uid = activity_factors.get(uid, {b: 0.0 for b in normalized_kappas})
    decay_rate = config['activity'].get('decay_rate', 1.0)

    for book_id, roundtrip_volume in miner_roundtrip_volumes.items():
        if latest_roundtrip_volumes[book_id] > 0:
            # ACTIVE BOOK: Calculate activity boost based on volume
            # Formula: 1 + (volume/volume_cap × activity_impact)
            activity_factors_uid[book_id] = min(
                1 + ((roundtrip_volume * volume_cap_inv) * activity_impact), 
                2.0
            )
        else:
            # INACTIVE BOOK: Apply exponential decay
            if decay_rate == 0.0:
                # Decay disabled, skip this book
                continue
                
            latest_time = latest_roundtrip_timestamps[book_id]
            
            if latest_time > 0:
                # Calculate time since last activity
                inactive_time = max(0, simulation_timestamp - latest_time)
            else:
                # Never traded on this book, use full simulation time
                inactive_time = simulation_timestamp
            
            current_factor = activity_factors_uid[book_id]
            activity_multiplier = max(current_factor, 1.0)

            # Time acceleration: Decay accelerates based on how long inactive
            if inactive_time <= decay_grace_period:
                # Within grace period: no decay acceleration
                time_acceleration = 1.0
            else:
                # Beyond grace period: accelerated decay
                # The longer inactive, the faster the decay
                time_beyond_grace = inactive_time - decay_grace_period
                time_ratio = time_beyond_grace * decay_window_ns_inv
                # Quadratic acceleration (time_acceleration_power = 2.0)
                time_acceleration = 1 + (time_ratio ** time_acceleration_power) * decay_rate
            
            # Total acceleration combines current factor with time acceleration
            # Higher current factors decay faster (to prevent "coasting" on past activity)
            total_acceleration = activity_multiplier * time_acceleration
            total_acceleration = min(total_acceleration, 100.0)  # Safety cap
            
            # Apply exponential decay: factor *= base_decay_factor^acceleration
            try:
                decay_factor = base_decay_factor ** total_acceleration
                if not np.isfinite(decay_factor):
                    bt.logging.error(f"UID {uid} book {book_id}: Non-finite decay_factor")
                    decay_factor = 0.0
            except (OverflowError, ValueError) as e:
                bt.logging.error(f"UID {uid} book {book_id}: Decay overflow - {e}")
                decay_factor = 0.0
            
            # Update activity factor with decay
            activity_factors_uid[book_id] *= decay_factor
    
    # ===== STEP 5: CALCULATE P&L-BASED MULTIPLIERS =====
    # P&L factors boost/penalize Kappa scores based on realized profitability per book
    # This rewards miners who achieve good risk-adjusted returns AND make money
    # Range: 0.0 to (1 + config.kappa.pnl.impact)
    pnl_factors_uid = pnl_factors.get(uid, {b: 1.0 for b in normalized_kappas})
    pnl_impact = config.get('kappa', {}).get('pnl', {}).get('impact', 0.0)
    
    if pnl_impact > 0:
        # Normalization: 100% DAILY return is the baseline for max boost
        # This makes P&L factors comparable across different assessment windows
        DAILY_NS = 86400_000_000_000  # 24 hours in nanoseconds
        assessment_window_ns = lookback * config['interval']
        window_fraction = assessment_window_ns / DAILY_NS
        pnl_reference = simulation_config['miner_wealth'] * window_fraction
        
        # Compute realized P&L per book over lookback window
        # Only counts completed trades (realized gains/losses)
        book_realized_pnl = {}
        if uid in realized_pnl_history:
            for timestamp, books_dict in realized_pnl_history[uid].items():
                if timestamp >= lookback_threshold:
                    for book_id, pnl in books_dict.items():
                        if book_id not in book_realized_pnl:
                            book_realized_pnl[book_id] = 0.0
                        book_realized_pnl[book_id] += pnl
        
        # Calculate P&L factor per book
        for book_id in normalized_kappas.keys():
            realized_pnl_book = book_realized_pnl.get(book_id, 0.0)
            
            # Normalize by daily-scaled reference
            # pnl_ratio = 1.0 means they earned enough to imply 100% daily return
            pnl_ratio = realized_pnl_book / pnl_reference
            
            # Raw P&L factor calculation:
            #   0% daily return → 1.0x (neutral)
            #   +100% daily return → (1+impact)x
            #   -100% daily return → (1+impact)x
            raw_pnl_factor = max(1.0 + pnl_ratio, 0.0)
            
            # Apply impact scaling: controls how much P&L affects final score
            pnl_factor = 1.0 + ((raw_pnl_factor - 1.0) * pnl_impact)
            
            # Cap at 2x boost
            pnl_factor = min(pnl_factor, 2.0)
            
            pnl_factors_uid[book_id] = pnl_factor
    else:
        # P&L weighting disabled - set all to neutral (1.0 = no effect)
        for book_id in normalized_kappas.keys():
            pnl_factors_uid[book_id] = 1.0
    
    # ===== STEP 6: COMBINE ACTIVITY AND P&L FACTORS, APPLY TO KAPPA =====
    # Both factors are multiplicative: combined_factor = activity × pnl
    # This means miners need BOTH good activity AND good profitability for max boost
    # Asymmetric weighting is applied based on factor magnitude and Kappa value
    activity_weighted_normalized_kappas = {}
    
    for book_id in normalized_kappas.keys():
        activity_factor = activity_factors_uid[book_id]
        pnl_factor = pnl_factors_uid[book_id]
        combined_factor = activity_factor * pnl_factor
        
        norm_kappa = normalized_kappas[book_id]
        
        if norm_kappa is None:
            # No Kappa data for this book (insufficient trading history)
            activity_weighted_normalized_kappas[book_id] = None
        else:
            # Apply combined factor with asymmetric weighting
            if combined_factor < 1 or norm_kappa > 0.5:
                weighted = combined_factor * norm_kappa
            else:
                weighted = (2 - combined_factor) * norm_kappa
            
            # Cap at 1.0 (normalized Kappa range)
            activity_weighted_normalized_kappas[book_id] = min(weighted, 1)

    uid_kappa['books_weighted'] = activity_weighted_normalized_kappas
    
    # ===== STEP 7: HANDLE INACTIVE BOOKS =====
    # Miners are allowed a certain number of inactive books without penalty
    max_inactive_books_ratio = config['max_inactive_books_ratio']
    total_books = len(normalized_kappas)
    max_inactive_books = int(max_inactive_books_ratio * total_books)
    
    # Separate books into: valid kappa vs no kappa (None)
    books_with_scores = []
    books_with_no_kappa = []
    
    for book_id, score in activity_weighted_normalized_kappas.items():
        if score is None:
            books_with_no_kappa.append(book_id)
        else:
            books_with_scores.append(score)
    
    num_books_no_kappa = len(books_with_no_kappa)
    
    # Determine scoring data
    if num_books_no_kappa <= max_inactive_books:
        # Can ignore all books with no kappa (within allowed limit)
        data = np.array(books_with_scores)
        
        uid_kappa['inactive_books'] = books_with_no_kappa
        uid_kappa['penalty_books'] = []
        
        bt.logging.trace(
            f"UID {uid}: Ignoring {num_books_no_kappa} books with no kappa "
            f"(≤ max_inactive_books={max_inactive_books}, {max_inactive_books_ratio*100:.1f}% of {total_books} books)"
        )
    else:
        # More than max_inactive_books have no kappa - excess contribute 0.0
        excess_inactive = num_books_no_kappa - max_inactive_books
        
        # Include valid scores + zeros for excess inactive books
        data = np.array(books_with_scores + [0.0] * excess_inactive)
        
        uid_kappa['inactive_books'] = books_with_no_kappa[:max_inactive_books]
        uid_kappa['penalty_books'] = books_with_no_kappa[max_inactive_books:]
        
        bt.logging.debug(
            f"UID {uid}: {num_books_no_kappa} books with no kappa "
            f"(> max_inactive_books={max_inactive_books}, {max_inactive_books_ratio*100:.1f}% of {total_books} books). "
            f"Ignoring {max_inactive_books}, penalizing {excess_inactive} as 0.0"
        )
    
    # ===== STEP 8: CALCULATE OUTLIER PENALTY =====
    # Use the 1.5×IQR rule to detect left-hand outliers in the activity-weighted Kappas
    # Outliers indicate books where the miner performed significantly worse than their median
    # This penalizes inconsistent performance across books
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    
    # Apply minimum IQR to prevent division issues and scale penalty appropriately
    min_iqr = 0.01
    effective_iqr = max(iqr, min_iqr)
    lower_threshold = q1 - 1.5 * effective_iqr
    outliers = data[data < lower_threshold]
    
    # Outliers detected here are activity-weighted Kappas which are significantly lower than those achieved on other books
    # A penalty equal to 67% of the difference between the mean outlier value and the value at the centre of the possible activity weighted Kappa values is calculated
    # Penalty is scaled by consistency: tight clusters (low IQR) get reduced penalty to reward consistent performance
    if len(outliers) > 0 and np.median(outliers) < 0.5:
        base_penalty = (0.5 - np.median(outliers)) / 1.5
        consistency_bonus = 1.0 - np.exp(-5 * iqr)  # Sigmoid-like scaling
        outlier_penalty = base_penalty * consistency_bonus
    else:
        outlier_penalty = 0
    
    # ===== STEP 9: FINAL KAPPA SCORE =====
    # The median of the activity-weighted Kappas provides the base score for the miner
    # Median is robust to outliers and represents "typical" performance across books
    activity_weighted_normalized_median = np.median(data)
    
    # The penalty factor is subtracted from the base score to punish particularly  poor performance on any particular book
    kappa_score = max(activity_weighted_normalized_median - abs(outlier_penalty), 0.0)

    # Store metadata for debugging/reporting
    uid_kappa['activity_weighted_normalized_median'] = activity_weighted_normalized_median
    uid_kappa['penalty'] = abs(outlier_penalty)
    uid_kappa['score'] = kappa_score
    uid_kappa['num_scored_books'] = len(data)
    
    return kappa_score


def calculate_pnl_score(
    realized_pnl_history: Dict,
    uid: int,
    lookback: int,
    interval: int,
    lookback_threshold: int,
    miner_wealth: float,
    book_count: int,
    max_inactive_books_ratio: float,
    config: Dict
) -> float:
    """
    Calculate normalized P&L score using per-book daily returns with median aggregation.
    
    This function measures absolute profitability per book:
    - Calculates daily return for EACH book independently
    - Uses per-book capital allocation (miner_wealth / book_count) as reference
    - Allows up to max_inactive_books to have zero P&L without penalty
    - Excess inactive books contribute 0.0 to the median calculation
    - Takes MEDIAN across scored books (consistent with Kappa methodology)
    - Maps to [-0.5, 0.5] with 0.0 as neutral (breakeven)
    
    Scoring Philosophy:
    - Per-book calculation: Prevents one profitable book from masking losses elsewhere
    - Inactive book tolerance: Miners can focus on subset of books without penalty
    - Excess inactive penalty: Too many inactive books dilute the score toward 0.0
    - Median aggregation: Robust to outliers, represents "typical" book performance
    - Consistent with Kappa: Both use per-book → median with inactive tolerance
    
    Args:
        realized_pnl_history: P&L history from completed trades
            Format: {uid: {timestamp: {book_id: pnl}}}
        uid: UID to score
        lookback: Lookback period in intervals
        interval: Interval duration in nanoseconds
        lookback_threshold: Minimum timestamp to include (pre-calculated)
        miner_wealth: Total initial capital across all books
        book_count: Number of books in simulation
        max_inactive_books_ratio: Ratio of books allowed to be inactive (e.g., 0.375)
        config: Normalization config with optional keys:
            - min_daily_return: Floor for daily return (default: -1.0 = -100%)
            - max_daily_return: Cap for daily return (default: 1.0 = +100%)
        
    Returns:
        float: P&L score in [-0.5, 0.5] range where:
            -0.5 = worst possible (median book lost 100% daily)
             0.0 = neutral (median book breakeven)
            +0.5 = best possible (median book gained 100% daily)
    """
    # Early exit if no P&L data
    if uid not in realized_pnl_history:
        return 0.0  # Neutral score for no P&L data
    
    # ===== STEP 1: CALCULATE REALIZED P&L PER BOOK =====
    # Aggregate realized P&L per book over lookback window
    book_realized_pnl = {}
    for timestamp, books_dict in realized_pnl_history[uid].items():
        if timestamp >= lookback_threshold:
            for book_id, pnl in books_dict.items():
                if book_id not in book_realized_pnl:
                    book_realized_pnl[book_id] = 0.0
                book_realized_pnl[book_id] += pnl
    
    # If no realized P&L in lookback window, return neutral
    if not book_realized_pnl:
        return 0.0
    
    # ===== STEP 2: CALCULATE DAILY RETURN RATIO PER BOOK =====
    # Per-book capital allocation
    DAILY_NS = 86400_000_000_000  # 24 hours in nanoseconds
    assessment_window_ns = lookback * interval
    window_fraction = assessment_window_ns / DAILY_NS
    
    # Capital per book (equal allocation across books)
    capital_per_book = miner_wealth / book_count
    pnl_reference_per_book = capital_per_book * window_fraction
    
    if pnl_reference_per_book == 0:
        bt.logging.warning(
            f"UID {uid}: Per-book P&L reference is zero "
            f"(miner_wealth={miner_wealth}, book_count={book_count})"
        )
        return 0.0
    
    # Get normalization bounds
    min_daily = config.get('min_daily_return', -1.0)
    max_daily = config.get('max_daily_return', 1.0)
    
    # Calculate daily return ratio per book
    books_with_pnl = []  # Books that traded
    books_inactive = []  # Books with no P&L
    
    for book_id in range(book_count):
        book_pnl = book_realized_pnl.get(book_id, 0.0)
        
        if book_pnl == 0.0:
            # No P&L on this book (inactive)
            books_inactive.append(book_id)
        else:
            # Normalize to daily return for this book
            daily_return_ratio = book_pnl / pnl_reference_per_book
            
            # Clip to expected bounds
            daily_return_ratio = max(min_daily, min(max_daily, daily_return_ratio))
            
            books_with_pnl.append(daily_return_ratio)
    
    # ===== STEP 3: HANDLE INACTIVE BOOKS =====
    # Calculate max allowed inactive books
    max_inactive_books = int(max_inactive_books_ratio * book_count)
    num_inactive = len(books_inactive)
    
    # Determine which books to include in scoring
    if num_inactive <= max_inactive_books:
        # Can ignore all inactive books (within allowed limit)
        scoring_data = books_with_pnl
        
        bt.logging.trace(
            f"UID {uid} P&L: Ignoring {num_inactive} inactive books "
            f"(≤ max_inactive_books={max_inactive_books}, "
            f"{max_inactive_books_ratio*100:.1f}% of {book_count} books)"
        )
    else:
        # More than max_inactive_books are inactive - excess contribute 0.0
        excess_inactive = num_inactive - max_inactive_books
        
        # Include active books + zeros for excess inactive books
        scoring_data = books_with_pnl + [0.0] * excess_inactive
        
        bt.logging.debug(
            f"UID {uid} P&L: {num_inactive} inactive books "
            f"(> max_inactive_books={max_inactive_books}, "
            f"{max_inactive_books_ratio*100:.1f}% of {book_count} books). "
            f"Ignoring {max_inactive_books}, penalizing {excess_inactive} as 0.0"
        )
    
    # ===== STEP 4: TAKE MEDIAN ACROSS SCORED BOOKS =====
    # If no books to score, return neutral
    if not scoring_data:
        return 0.0
    
    # Median is robust to outliers and represents "typical" book performance
    median_daily_return = np.median(scoring_data)
    
    # ===== STEP 5: MAP TO [-0.5, 0.5] SCORE =====
    # Map median daily return to P&L score
    pnl_score = median_daily_return / 2.0
    
    bt.logging.trace(
        f"UID {uid}: P&L score calculation - "
        f"active_books={len(books_with_pnl)}, "
        f"inactive_books={num_inactive}, "
        f"scored_books={len(scoring_data)}, "
        f"median_return={median_daily_return*100:.1f}%, "
        f"pnl_score={pnl_score:.4f}"
    )
    
    return pnl_score


def _gentrx_rank_normalize(gentrx_scores: Dict) -> Dict[int, float]:
    """Rank-normalize raw GenTRX gradient scores to [0, 1].

    Maps raw scores (arbitrary range, often negative) to a uniform [0, 1]
    distribution based on rank among all scored miners this round.
    Ties get the same rank. Miners not in gentrx_scores are absent from output.

    Returns {uid: normalized_score} where 1.0 = best, 0.0 = worst.
    """
    if not gentrx_scores:
        return {}

    # Extract (uid, raw_score) pairs
    scored = {}
    for uid_key, entry in gentrx_scores.items():
        uid_int = int(uid_key) if isinstance(uid_key, str) else uid_key
        if isinstance(entry, dict):
            scored[uid_int] = entry.get('score', 0.0)
        else:
            scored[uid_int] = float(entry) if entry is not None else 0.0

    if not scored:
        return {}
    if len(scored) == 1:
        uid = next(iter(scored))
        # Single miner: 1.0 if positive, 0.0 if negative
        return {uid: 1.0 if scored[uid] > 0 else 0.0}

    # Sort by score ascending — rank 0 = worst, rank N-1 = best
    sorted_uids = sorted(scored.keys(), key=lambda u: scored[u])
    n = len(sorted_uids)
    result = {}
    for rank, uid in enumerate(sorted_uids):
        result[uid] = rank / (n - 1)  # [0.0, 1.0]
    return result


def score_uid(validator_data: Dict, uid: int) -> Tuple[float, float]:
    """
    Computes the per-UID trading and gentrx scores for the two-pool allocation.

    Two pools, two scores:

    TRADING SCORE = kappa_weight * kappa + pnl_weight * pnl
        - kappa: per-book risk-adjusted returns (Kappa-3 metric), already
          weighted by activity and P&L factors and aggregated across books
        - pnl: aggregate realized P&L normalized to daily return
        - kappa_weight + pnl_weight must equal 1.0 (validated at init)

    GENTRX SCORE = post-EMA rank-normalized gradient quality
        - Rank-normalized to [0, 1] across miners scored this round
        - Per-UID EMA smoothing (alpha=0.1) over rounds
        - 0 for miners that did not submit gradients this round
        - Computed only when --scoring.gentrx.simulation_share > 0

    Pool combination, Pareto multiplication, slow EMA, and burn allocation
    happen downstream in `prepare_weights`. This function returns the raw
    per-UID inputs for both pools.

    Args:
        validator_data (Dict): Snapshot of validator state used for scoring,
            including kappa_values, activity_factors, pnl_factors,
            roundtrip_volumes, realized_pnl_history, config, simulation_config,
            simulation_timestamp, gentrx_scores, and gentrx_ema.
        uid (int): UID of the miner to score.

    Returns:
        Tuple[float, float]: (trading_score, gentrx_score), each in [0, 1].
    """
    # Extract required data from validator state
    kappa_values = validator_data['kappa_values']
    activity_factors = validator_data['activity_factors']
    pnl_factors = validator_data['pnl_factors']
    roundtrip_volumes = validator_data['roundtrip_volumes']
    realized_pnl_history = validator_data['realized_pnl_history']
    config = validator_data['config']['scoring']
    simulation_config = validator_data['simulation_config']
    simulation_timestamp = validator_data['simulation_timestamp']

    # ===== STEP 1: CALCULATE KAPPA SCORE COMPONENT =====
    # This calculates risk-adjusted returns with activity and P&L weighting
    # The Kappa score already includes:
    # - Per-book activity factors (volume-based participation weighting)
    # - Per-book P&L factors (profitability-based quality weighting)
    # - Outlier penalty (consistency enforcement)
    kappa_score = calculate_kappa_score(
        uid=uid,
        kappa_values=kappa_values,
        activity_factors=activity_factors,
        pnl_factors=pnl_factors,
        roundtrip_volumes=roundtrip_volumes,
        realized_pnl_history=realized_pnl_history,
        config=config,
        simulation_config=simulation_config,
        simulation_timestamp=simulation_timestamp
    )
    
    # ===== STEP 2: P&L COMPONENT =====
    pnl_config = config.get('pnl', {})
    pnl_score_weight = pnl_config.get('weight', 0.0)
    pnl_score = 0.0

    if pnl_score_weight > 0:
        lookback = config['kappa']['lookback']
        publish_interval = simulation_config['publish_interval']
        lookback_threshold = simulation_timestamp - (lookback * publish_interval)

        pnl_score = calculate_pnl_score(
            realized_pnl_history=realized_pnl_history,
            uid=uid,
            lookback=lookback,
            interval=config['interval'],
            lookback_threshold=lookback_threshold,
            miner_wealth=simulation_config['miner_wealth'],
            book_count=simulation_config['book_count'],
            max_inactive_books_ratio=config['max_inactive_books_ratio'],
            config=pnl_config.get('normalization', {})
        )

    # ===== STEP 3: GenTRX COMPONENT =====
    # Computed only when the gentrx pool is funded. Output is the raw
    # rank-normalized + per-UID EMA-smoothed score in [0, 1]. Pool sizing
    # (simulation_pool * gentrx_simulation_share * N/N_target) is applied
    # downstream in prepare_weights.
    gentrx_config = config.get('gentrx', {})
    gentrx_sim_share = gentrx_config.get('simulation_share', 0.0)
    gentrx_score = 0.0

    if gentrx_sim_share > 0:
        gentrx_scores = validator_data.get('gentrx_scores', {})
        gentrx_ranked = _gentrx_rank_normalize(gentrx_scores)
        round_score = gentrx_ranked.get(uid, 0.0)
        gentrx_ema = validator_data.get('gentrx_ema', {})
        alpha = gentrx_config.get('ema_alpha', 0.1)
        prev = gentrx_ema.get(uid, round_score)
        gentrx_score = alpha * round_score + (1.0 - alpha) * prev
        gentrx_ema[uid] = gentrx_score

    # ===== STEP 4: TRADING SCORE (kappa + pnl, weighted-sum, clamp) =====
    # kappa_weight + pnl_weight must sum to 1.0 within the trading pool
    # (validated at validator init).
    kappa_weight = config['kappa'].get('weight', 0.0)

    trading_score = (kappa_weight * kappa_score) + (pnl_score_weight * pnl_score)
    trading_score = max(0.0, min(1.0, trading_score))
    gentrx_score = max(0.0, min(1.0, gentrx_score))

    if kappa_values.get(uid):
        uid_kappa = kappa_values[uid]
        uid_kappa['pnl_score'] = pnl_score if pnl_score_weight > 0 else None
        uid_kappa['gentrx_score'] = gentrx_score if gentrx_sim_share > 0 else None
        uid_kappa['kappa_weight'] = kappa_weight
        uid_kappa['pnl_score_weight'] = pnl_score_weight
        uid_kappa['gentrx_simulation_share'] = gentrx_sim_share
        uid_kappa['trading_score'] = trading_score
        uid_kappa['final_score'] = trading_score

    bt.logging.trace(
        f"UID {uid}: score - "
        f"kappa={kappa_score:.4f} (w={kappa_weight:.2f}), "
        f"pnl={pnl_score:.4f} (w={pnl_score_weight:.2f}), "
        f"trading={trading_score:.4f}, "
        f"gentrx={gentrx_score:.4f} (sim_share={gentrx_sim_share:.4f})"
    )

    return trading_score, gentrx_score

def score_uids(validator_data: Dict) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Computes per-UID trading and gentrx scores for all UIDs in this round.

    Orchestrates the Kappa-3 calculation, then iterates UIDs through
    `score_uid` to produce the two-pool inputs.

    Returns:
        Tuple[Dict[int, float], Dict[int, float]]:
            (trading_scores, gentrx_scores) keyed by UID, each in [0, 1].
    """
    config = validator_data['config']['scoring']
    uids = validator_data['uids']
    deregistered_uids = validator_data['deregistered_uids']
    simulation_config = validator_data['simulation_config']
    kappa_values = validator_data['kappa_values']
    kappa_cache = validator_data['kappa_cache']
    realized_pnl_history = validator_data['realized_pnl_history']
    tau = config['kappa']['tau']

    if config['kappa']['parallel_workers'] == 0:
        cache_updates = {}
        for uid in uids:
            realized_pnl_value = realized_pnl_history.get(uid, {})
            kappa_result = kappa_3(
                uid,
                realized_pnl_value,
                tau,
                config['kappa']['lookback'],
                config['kappa']['normalization_min'],
                config['kappa']['normalization_max'],
                config['kappa']['min_lookback'],
                config['kappa']['min_realized_observations'],
                simulation_config['grace_period'],
                deregistered_uids,
                simulation_config['book_count'],
                cache=kappa_cache
            )
            kappa_values[uid] = kappa_result
            fingerprint = _get_pnl_fingerprint(realized_pnl_value)
            cache_updates[uid] = (fingerprint, kappa_result)
        kappa_cache.update(cache_updates)        
    else:
        if config['kappa']['parallel_workers'] == -1:
            num_processes = len(config['kappa']['reward_cores'])
        else:
            num_processes = config['kappa']['parallel_workers']
        total_uids = len(uids)
        batch_size = max(1, total_uids // num_processes)
        batches = []
        for i in range(0, total_uids, batch_size):
            batch = uids[i:i+batch_size]
            if batch:
                batches.append(batch)
        actual_cores = config['kappa']['reward_cores'][:len(batches)]
        bt.logging.debug(
            f"Parallel kappa calculation: {total_uids} UIDs split into {len(batches)} batches "
            f"(batch sizes: {[len(b) for b in batches]}) across {len(actual_cores)} cores"
        )
        
        kappa_results, cache_updates = batch_kappa_3(
            realized_pnl_history,
            tau,
            batches,
            config['kappa']['lookback'],
            config['kappa']['normalization_min'],
            config['kappa']['normalization_max'],
            config['kappa']['min_lookback'],
            config['kappa']['min_realized_observations'],
            simulation_config['grace_period'],
            deregistered_uids,
            simulation_config['book_count'],
            cache=kappa_cache,
            cores=actual_cores
        )
        kappa_values.update(kappa_results)
        kappa_cache.update(cache_updates)

    validator_data['kappa_values'] = kappa_values

    trading_scores: Dict[int, float] = {}
    gentrx_scores: Dict[int, float] = {}
    for uid in uids:
        t, g = score_uid(validator_data, uid)
        trading_scores[uid] = t
        gentrx_scores[uid] = g
    return trading_scores, gentrx_scores

def distribute_rewards(rewards: list, config: Dict) -> torch.FloatTensor:
    """
    Distributes rewards using a Pareto distribution to create variance in reward allocation.

    Args:
        rewards (list): List of raw reward scores for each UID
        config (Dict): Configuration dictionary containing rewarding parameters with keys:
            - rewarding.seed (int): Random seed for reproducibility
            - rewarding.pareto.shape (float): Shape parameter for Pareto distribution
            - rewarding.pareto.scale (float): Scale parameter for Pareto distribution

    Returns:
        torch.FloatTensor: Tensor of distributed rewards maintaining the original order of UIDs
    """
    rng = np.random.default_rng(config['rewarding']['seed'])
    num_uids = len(rewards)
    distribution = torch.FloatTensor(sorted(
        config['rewarding']['pareto']['scale'] *
        rng.pareto(config['rewarding']['pareto']['shape'], num_uids)
    ))
    rewards_tensor = torch.FloatTensor(rewards)
    sorted_rewards, sorted_indices = rewards_tensor.sort()
    distributed_rewards = distribution * sorted_rewards
    return torch.gather(distributed_rewards, 0, sorted_indices.argsort())

def get_rewards(self: 'Validator') -> Tuple[torch.FloatTensor, torch.FloatTensor, Dict]:
    """
    Calculate per-round trading and gentrx rewards for all UIDs.

    Two-pool architecture:
    - Trading rewards: kappa+pnl combine, then Pareto sort-multiply
    - GenTRX rewards:  rank-norm + per-UID EMA, NO Pareto

    Args:
        self (Validator): The intelligent markets simulation validator.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor, Dict, List[int]]:
            (trading_rewards_pareto, gentrx_rewards, updated_data, all_uids).
            Both tensors are length-effective_max_uids in UID order.
            all_uids is returned so the caller uses the same snapshot, not a
            re-evaluated range that may have grown if the metagraph synced
            concurrently.
    """
    roundtrip_volumes = self.roundtrip_volumes
    realized_pnl_history = self.realized_pnl_history
    all_uids = list(range(self.effective_max_uids))

    validator_data = {
        'kappa_values': self.kappa_values,
        'kappa_cache': self.kappa_cache,
        'activity_factors': self.activity_factors,
        'pnl_factors': self.pnl_factors,
        'roundtrip_volumes': roundtrip_volumes,
        'realized_pnl_history': realized_pnl_history,
        'config': {
            'scoring': {
                'kappa': {
                    'weight': self.config.scoring.kappa.weight,
                    'normalization_min': self.config.scoring.kappa.normalization_min,
                    'normalization_max': self.config.scoring.kappa.normalization_max,
                    'min_lookback': self.config.scoring.kappa.min_lookback,
                    'lookback': self.config.scoring.kappa.lookback,
                    'min_realized_observations': self.config.scoring.kappa.min_realized_observations,
                    'parallel_workers': self.config.scoring.kappa.parallel_workers,
                    'reward_cores': self.reward_cores,
                    'tau': self.config.scoring.kappa.tau,
                    'pnl_impact': self.config.scoring.kappa.pnl.impact
                },
                'pnl': {
                    'weight': self.config.scoring.pnl.weight,
                    'normalization': {
                        'min_daily_return': self.config.scoring.pnl.normalization.min_daily_return,
                        'max_daily_return': self.config.scoring.pnl.normalization.max_daily_return,
                    }
                },
                'gentrx': {
                    'simulation_share': getattr(getattr(self.config.scoring, 'gentrx', None), 'simulation_share', 0.0) or 0.0,
                    'ema_alpha': getattr(getattr(self.config.scoring, 'gentrx', None), 'ema_alpha', 0.1) or 0.1,
                },
                'activity': {
                    'capital_turnover_cap': self.config.scoring.activity.capital_turnover_cap,
                    'trade_volume_sampling_interval': self.config.scoring.activity.trade_volume_sampling_interval,
                    'trade_volume_assessment_period': self.config.scoring.activity.trade_volume_assessment_period,
                    'decay_grace_period': self.config.scoring.activity.decay_grace_period,
                    'impact' : self.config.scoring.activity.impact,
                    'decay_rate': self.config.scoring.activity.decay_rate
                },
                'max_inactive_books_ratio': self.config.scoring.max_inactive_books,
                'interval': self.config.scoring.interval,
            },
            'rewarding': {
                'seed': self.config.rewarding.seed,
                'pareto': {
                    'shape': self.config.rewarding.pareto.shape,
                    'scale': self.config.rewarding.pareto.scale,
                }
            },
        },
        'simulation_config': {
            'miner_wealth': self.simulation.miner_wealth,
            'publish_interval': self.simulation.publish_interval,
            'volumeDecimals': self.simulation.volumeDecimals,
            'grace_period': self.simulation.grace_period,
            'book_count': self.simulation.book_count, 
        },
        'simulation_timestamp': self.simulation_timestamp,
        'uids': all_uids,
        'deregistered_uids': self.deregistered_uids,
        'device': self.device,
        # GenTRX gradient-training scores (empty dict when GenTRX disabled).
        # Shape: {uid: {"score": float, "accepted": bool, "books": [...]}}
        'gentrx_scores': (
            self._gentrx.get_scores()
            if hasattr(self, '_gentrx') and self._gentrx is not None
            else {}
        ),
        # EMA state for GenTRX score smoothing — persists across rounds on self.
        'gentrx_ema': getattr(self, '_gentrx_ema', {}),
    }
    
    trading_uid_scores, gentrx_uid_scores = score_uids(validator_data)

    # Trading rewards run through Pareto sort-multiply.
    trading_rewards_list = [trading_uid_scores[uid] for uid in all_uids]
    distributed_trading = distribute_rewards(
        trading_rewards_list, validator_data['config']
    ).to(self.device)

    # GenTRX rewards skip Pareto. Pool sizing happens in prepare_weights.
    gentrx_rewards = torch.tensor(
        [gentrx_uid_scores[uid] for uid in all_uids],
        dtype=torch.float32,
        device=self.device,
    )

    # Persist GenTRX per-round EMA state back to validator for next round
    self._gentrx_ema = validator_data.get('gentrx_ema', {})

    updated_data = {
        'kappa_values': validator_data['kappa_values'],
        'activity_factors': validator_data['activity_factors'],
        'pnl_factors': validator_data['pnl_factors']
    }

    return distributed_trading, gentrx_rewards, updated_data, all_uids

def set_delays(self: 'Validator', synapse_responses: dict[int, MarketSimulationStateUpdate]) -> list[FinanceAgentResponse]:
    """
    Applies base delay based on process time using an exponential mapping,
    and adds a per-book Gaussian-distributed random latency instruction_delay to instructions,
    with zero instruction_delay applied to the first instruction per book.

    Args:
        self (taos.im.neurons.validator.Validator): Validator instance.
        synapse_responses (dict[int, MarketSimulationStateUpdate]): Latest state updates.

    Returns:
        list[FinanceAgentResponse]: Delayed finance responses.
    """
    start_time = time.time()
    responses = []
    timeout = self.config.neuron.timeout
    min_delay = self.config.scoring.min_delay
    max_delay = self.config.scoring.max_delay
    min_instruction_delay = self.config.scoring.min_instruction_delay
    max_instruction_delay = self.config.scoring.max_instruction_delay

    def compute_delay(p_time: float) -> int:
        """Exponential scaling of process time into delay."""
        t = p_time / timeout
        exp_scale = 5
        delay_frac = (np.exp(exp_scale * t) - 1) / (np.exp(exp_scale) - 1)
        delay = min_delay + delay_frac * (max_delay - min_delay)
        return int(delay)
    log_messages = []
    for uid, synapse_response in synapse_responses.items():
        response = synapse_response.response
        if response:
            base_delay = compute_delay(synapse_response.dendrite.process_time)
            seen_books = set()
            for instruction in response.instructions:
                book_id = instruction.bookId

                if book_id not in seen_books:
                    instruction_delay = 0
                    seen_books.add(book_id)
                else:
                    instruction_delay = random.randint(min_instruction_delay, max_instruction_delay)
                instruction.delay += base_delay + instruction_delay
            responses.append(response)
            log_messages.append(
                f"UID {response.agent_id} responded with {len(response.instructions)} instructions "
                f"after {synapse_response.dendrite.process_time:.4f}s – base delay {base_delay}{self.simulation.time_unit if hasattr(self, 'simulation') else 'ns'}"
            )
    if log_messages:
        bt.logging.info("\n".join(log_messages))    
    elapsed = time.time() - start_time
    if elapsed > 0.1:
        bt.logging.warning(f"set_delays took {elapsed:.4f}s for {len(synapse_responses)} responses")
    return responses