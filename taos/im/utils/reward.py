# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Account inventory valuation: computes instantaneous total account value
using best-bid, midquote, or order-book liquidation pricing.
"""

from typing import Dict

def get_inventory_value(account: Dict, book: Dict, method='midquote') -> float:
    """
    Calculates the instantaneous total value of an account's inventory using the specified method

    Args:
        account (Dict): Account state dict with balance, loan, and collateral fields
            for both base ('bb', 'bl', 'bc') and quote ('qb', 'ql', 'qc') currencies.
        book (Dict): Order-book state dict with 'a' (asks) and 'b' (bids) level lists.
        method (str): Pricing method — 'best_bid' uses the top bid price, 'midquote'
            uses (bid + ask) / 2, 'liquidation' walks the bid side to simulate a
            market sell. Defaults to 'midquote'.

    Returns:
        float: Total inventory value of the account.
    """
    quote_balance = account['qb']['t'] - account['ql'] + account['qc']
    base_balance = account['bb']['t'] - account['bl'] + account['bc']

    book_a = book['a']
    book_b = book['b']
    has_orders = len(book_a) > 0 and len(book_b) > 0

    if method == "best_bid":
        price = book_b[0]['p'] if has_orders else 0.0
        return quote_balance + price * base_balance
    elif method == "midquote":
        price = (book_a[0]['p'] + book_b[0]['p']) / 2 if has_orders else 0.0
        return quote_balance + price * base_balance
    else:  # liquidation
        liq_value = 0.0
        to_liquidate = account['bb']['t']
        for bid in book_b:
            if to_liquidate == 0:
                break
            level_liq = min(to_liquidate, bid['q'])
            liq_value += level_liq * bid['p']
            to_liquidate -= level_liq
        return quote_balance + liq_value