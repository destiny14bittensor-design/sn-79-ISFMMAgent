# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Order-book history reconstruction: replays order, trade, and cancel events
onto an LOB snapshot to produce a timestamped sequence of book states.
"""
from loky.backend.context import set_start_method
set_start_method('forkserver', force=True)
from loky import get_reusable_executor, as_completed
import time
import copy

def history(snapshot, events, volume_decimals):
    """
    Replay a sequence of order-book events onto an initial snapshot.

    Applies order placements, trades, and cancellations in chronological order,
    recording a deep copy of the book state at each event timestamp.

    Args:
        snapshot (dict): Initial LOB state with 'bids', 'asks', and 'timestamp' keys.
        events (list[dict]): List of event dicts with 'y' (type), 't' (timestamp),
            'p' (price), 'q' (quantity), and 's' (side) fields.
        volume_decimals (int): Decimal precision for rounding quantity accumulation.

    Returns:
        Tuple[dict, dict]: (history_dict, trades_dict) where history_dict maps
            timestamp to book snapshot and trades_dict maps timestamp to trade event.
    """
    history = {snapshot['timestamp']: copy.deepcopy(snapshot)}
    trades = {}
    # Apply events in chronological order
    for event in sorted(events, key=lambda x: x['t']):
        match event:
            case o if event['y'] == 'o':
                # Place new order
                if o['s'] == 0:
                    if o['p'] not in snapshot['bids']:
                        snapshot['bids'][o['p']] = {'p':o['p'], 'q':0.0, 'o':None}
                    snapshot['bids'][o['p']]['q'] = round(
                        snapshot['bids'][o['p']]['q'] + o['q'],
                        volume_decimals
                    )
                else:
                    if o['p'] not in snapshot['asks']:
                        snapshot['asks'][o['p']] =  {'p':o['p'], 'q':0.0, 'o':None}
                    snapshot['asks'][o['p']]['q'] = round(
                        snapshot['asks'][o['p']]['q'] + o['q'],
                        volume_decimals
                    )

            case t if event['y'] == 't':
                # Record trade
                trades[t['t']] = t
                if t['s'] == 0:
                    if t['p'] in snapshot['asks']:
                        snapshot['asks'][t['p']]['q'] = round(
                            snapshot['asks'][t['p']]['q'] - t['q'],
                            volume_decimals
                        )
                        if snapshot['asks'][t['p']]['q'] == 0.0:
                            del snapshot['asks'][t['p']]
                else:
                    if t['p'] in snapshot['bids']:
                        snapshot['bids'][t['p']]['q'] = round(
                            snapshot['bids'][t['p']]['q'] - t['q'],
                            volume_decimals
                        )
                        if snapshot['bids'][t['p']]['q'] == 0.0:
                            del snapshot['bids'][t['p']]

            case c if event['y'] == 'c':
                # Cancel existing order
                if c['p'] >= min(snapshot['asks'].keys()):
                    if c['p'] in snapshot['asks']:
                        snapshot['asks'][c['p']]['q'] = round(
                            snapshot['asks'][c['p']]['q'] - c['q'],
                            volume_decimals
                        )
                        if snapshot['asks'][c['p']]['q'] == 0.0:
                            del snapshot['asks'][c['p']]
                else:
                    if c['p'] in snapshot['bids']:
                        snapshot['bids'][c['p']]['q'] = round(
                            snapshot['bids'][c['p']]['q'] - c['q'],
                            volume_decimals
                        )
                        if snapshot['bids'][c['p']]['q'] == 0.0:
                            del snapshot['bids'][c['p']]

        # Add snapshot to history after each update
        snapshot['timestamp'] = event['t']
        history[event['t']] = copy.deepcopy(snapshot)
    return history, trades
    
def history_batch(snapshots, events, volume_decimals):
    """
    Compute order-book histories for a batch of books sequentially.

    Args:
        snapshots (dict): Mapping of book_id to initial LOB snapshot.
        events (dict): Mapping of book_id to list of event dicts.
        volume_decimals (int): Decimal precision for volume rounding.

    Returns:
        dict: Mapping of book_id to the (history_dict, trades_dict) tuple from `history`.
    """
    start = time.time()
    result = {book_id : history(snapshot, events[book_id], volume_decimals) for book_id, snapshot in snapshots.items()}
    print(f"Calculated histories for books {list(result.keys())[0]}-{list(result.keys())[-1]} ({time.time() - start}s)")
    return result

def batch_history(snapshots, events, batches, volume_decimals):
    """
    Compute order-book histories for all books in parallel using loky workers.

    Args:
        snapshots (dict): Mapping of book_id to initial LOB snapshot.
        events (dict): Mapping of book_id to list of event dicts.
        batches (list[list[int]]): Partition of book IDs into worker batches.
        volume_decimals (int): Decimal precision for volume rounding.

    Returns:
        dict: Mapping of book_id (int) to (history_dict, trades_dict) tuples.
    """
    history_batches= []
    pool = get_reusable_executor(max_workers=len(batches))
    tasks = [pool.submit(history_batch, {book_id : snapshots[book_id] for book_id in batch}, {book_id : events[book_id] for book_id in batch}, volume_decimals) for batch in batches]
    for task in as_completed(tasks):
        result = task.result()        
        history_batches.append(result)
    return {int(k): v for d in history_batches for k, v in d.items()}