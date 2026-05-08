# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Random market-maker agent: places limit orders at random prices between the
best bid and ask. Supports GenTRX distributed training via agent params.
"""
import bittensor as bt
from taos.common.agents import launch
from taos.im.agents import LimitOrderPlacementEvent, OrderCancellationEvent, OrderPlacementEvent, TradeEvent
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse

from taos.im.agents import GenTRXAgent
import random


class RandomMakerAgent(GenTRXAgent):
    def initialize(self):
        """
        Initializes properties, variables and quantities that will be used by the agent.
        The fields attached to `self.config` are defined in the launch parameters.
        """
        # GenTRX is opt-in: only activates when explicitly configured.
        if not hasattr(self.config, 'gtx_training_enabled'):
            self.config.gtx_training_enabled = False
        if not hasattr(self.config, 'gtx_collect_data'):
            self.config.gtx_collect_data = False
        super().initialize()
        self.min_quantity = self.config.min_quantity
        self.max_quantity = self.config.max_quantity
        self.min_leverage = self.config.min_leverage if hasattr(self.config, 'min_leverage') else 0.0
        self.max_leverage = self.config.max_leverage if hasattr(self.config, 'max_leverage') else 0.0
        self.max_fee_rate = self.config.max_fee_rate if hasattr(self.config, 'max_fee_rate') else 0.005
        self.expiry_period = self.config.expiry_period
        self.open_orders = {}

    def quantity(self):
        """
        Obtains a random quantity for order placement within the bounds defined by the agent strategy parameters.
        """
        return round(random.uniform(self.min_quantity,self.max_quantity),self.simulation_config.volumeDecimals)

    def leverage(self):
        """
        Obtains a random quantity for order placement within the bounds defined by the agent strategy parameters.
        """
        return round(random.uniform(self.min_leverage,self.max_leverage),2)

    def respond(self, state : MarketSimulationStateUpdate) -> FinanceAgentResponse:
        """
        The main logic of the strategy executed when a new state is received from validator.
        Analyses the latest market state data and generates instructions to be submitted.

        Args:
            state (MarketSimulationStateUpdate): The current market state data 
                provided by the simulation validator.

        Returns:
            FinanceAgentResponse: A response object containing the list of 
                instructions (e.g., limit orders) to submit to the market.
        """
        # GenTRX: data collection + training trigger (runs even when training disabled).
        response = super().respond(state)
        # Iterate over all the book realizations in the state message
        for book_id, book in state.books.items():

            # NEW: Maker and taker fees will be dynamically moving under the DIS fee policy
            # The below demonstrates a simple approach for reacting to the changing rates in trading logic
            previous_maker_rate = self.accounts[book_id].fees.maker_fee_rate
            # Positive rate implies a fee is to be paid, negative rate results in rebates to the trader
            # Check the rate against the configured tolerance
            if previous_maker_rate > self.max_fee_rate:
                # If the rate exceeds the tolerance, cancel the order closest to the best level,
                bidOrders = {order.price : order for order in self.accounts[book_id].orders if order.side == OrderDirection.BUY}
                if bidOrders:
                    topBid = max(bidOrders.keys())
                    response.cancel_order(book_id, bidOrders[topBid].id)
                askOrders = {order.price : order for order in self.accounts[book_id].orders if order.side == OrderDirection.SELL}
                if askOrders:
                    topAsk = min(askOrders.keys())
                    response.cancel_order(book_id, askOrders[topAsk].id)
                continue
            
            # If the book is populated (it of course always should be)
            if len(book.bids) > 0 and len(book.asks) > 0:
                # Calculate placement prices for new orders to be a random distance between the current best bid and best ask
                bidprice = round(random.uniform(book.bids[0].price,book.asks[0].price),self.simulation_config.priceDecimals)
                askprice = round(random.uniform(bidprice,book.asks[0].price),self.simulation_config.priceDecimals)
            else:
                # Otherwise, place orders within 0.05 of the 100.0 price level
                bidprice = round(random.uniform(99.95,100.05),self.simulation_config.priceDecimals)
                askprice = round(random.uniform(bidprice,100.05),self.simulation_config.priceDecimals)
            # If the bid and ask prices are different i.e the spread is not too small to place both orders at different prices
            if bidprice != askprice:
                # Select a random side of the book to place an order
                direction = random.choice([OrderDirection.BUY, OrderDirection.SELL])
                bt.logging.info(f"BOOK {book_id} | QUOTE : {self.accounts[book_id].quote_balance.total} [LOAN {self.accounts[book_id].quote_loan} | COLLAT {self.accounts[book_id].quote_collateral}]")
                bt.logging.info(f"BOOK {book_id} | BASE : {self.accounts[book_id].base_balance.total} [LOAN {self.accounts[book_id].base_loan} | COLLAT {self.accounts[book_id].base_collateral}]")
                if direction == OrderDirection.BUY:
                    # Attach a buy limit order placement instruction to the response
                    # On the BUY side, we place leveraged orders according to the config
                    # Obtain a random leverage value if there is no open margin position on sell side
                    leverage = self.leverage() if self.accounts[book_id].base_loan == 0 else 0.0
                    # If an open opposite margin position exists, repay the corresponding loans in order 
                    # from oldest to newest by setting LoanSettlementOption.FIFO
                    settlement = LoanSettlementOption.NONE if self.accounts[book_id].base_loan == 0 else LoanSettlementOption.FIFO
                    # If placing unleveraged order, increase the quantity to better match the average total size of 
                    # leveraged orders on the other side.  This avoids accumulating too much inventory in one currency.
                    quantity =  round(self.quantity() * (1 + self.leverage()), self.simulation_config.volumeDecimals)
                    # If the agent can afford to place the buy order
                    if self.accounts[book_id].quote_balance.free >= quantity * bidprice:
                        response.limit_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            quantity=quantity, 
                            price=bidprice, 
                            stp=STP.CANCEL_BOTH, 
                            timeInForce=TimeInForce.GTT, 
                            expiryPeriod=self.expiry_period,
                            leverage=leverage,
                            settlement_option=settlement
                        )
                        bt.logging.info(f"SUBMITTING BUY LIMIT ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}@{bidprice}")
                    else:
                        print(f"CANNOT SUBMIT BUY ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}@{bidprice} : Insufficient quote balance!")
                if direction == OrderDirection.SELL:
                    # Attach a sell limit order placement instruction to the response
                    # In the SELL case, the order quantity is adjusted to approximate that of the leveraged buy order
                    # Obtain a random leverage value if there is no open margin position on buy side
                    leverage = self.leverage() if self.accounts[book_id].quote_loan == 0 else 0.0
                    # If an open opposite margin position exists, repay the corresponding loans in order 
                    # from oldest to newest by setting LoanSettlementOption.FIFO
                    settlement = LoanSettlementOption.NONE if self.accounts[book_id].quote_loan == 0 else LoanSettlementOption.FIFO
                    # If placing unleveraged order, increase the quantity to better match the average total size of 
                    # leveraged orders on the other side.  This avoids accumulating too much inventory in one currency.
                    quantity =  round(self.quantity() * (1 + self.leverage()), self.simulation_config.volumeDecimals)
                    # If the agent can afford to place the sell order
                    if self.accounts[book_id].base_balance.free >= quantity:
                        response.limit_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            quantity=round(quantity * 1 + leverage, self.simulation_config.volumeDecimals), 
                            price=askprice, 
                            stp=STP.CANCEL_NEWEST, 
                            timeInForce=TimeInForce.GTT, 
                            expiryPeriod=self.expiry_period,
                            leverage=leverage,
                            settlement_option=settlement
                        )
                        bt.logging.info(f"SUBMITTING SELL LIMIT ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}@{askprice}")
                    else:
                        print(f"CANNOT SUBMIT SELL ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}@{askprice} : Insufficient base balance!")
        # Return the response with instructions appended
        # The response will be serialized and sent back to the validator for processing
        return response
    
if __name__ == "__main__":
    """
    Example command for local standalone testing execution using Proxy:
    python RandomMakerAgent.py --port 8888 --agent_id 0 --params min_quantity=0.1 max_quantity=1.0 min_leverage=0.0 max_leverage=1.0 expiry_period=200000000000 max_fee_rate=0.002
    """
    launch(RandomMakerAgent)