# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Random market-taker agent: places market orders at random intervals.
Supports GenTRX distributed training via agent params.
"""
import bittensor as bt
from taos.common.agents import launch
from taos.im.protocol.models import *
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse

from taos.im.agents import GenTRXAgent
import random


class RandomTakerAgent(GenTRXAgent):
    def initialize(self):
        """
        Initialize properties, variables and quantities that will be used by the agent.
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
        self.max_fee_rate = self.config.max_fee_rate if hasattr(self.config, 'max_fee_rate') else 0.002
        # Initialize a variable which allows to maintain the same direction of trade for a defined period
        self.direction = {}

    def quantity(self):
        """
        Obtains a random quantity for order placement within the bounds defined by the agent strategy parameters.
        """
        return round(random.uniform(self.min_quantity,self.max_quantity),self.simulation_config.volumeDecimals)

    def leverage(self):
        """
        Obtains a random quantity for order placement within the bounds defined by the agent strategy parameters.
        """
        return round(random.uniform(self.min_leverage,self.max_leverage),2) if self.min_leverage != self.max_leverage else self.max_leverage

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
            # If we have not set a trade direction for this book, or 100 simulation seconds have elapsed
            if not book_id in self.direction or state.timestamp % 100_000_000_000 == 0:
                # Randomly select a new trade direction for the agent on this book
                self.direction[book_id] = random.choice([OrderDirection.BUY,OrderDirection.SELL])

            # NEW: Maker and taker fees will be dynamically moving under the DIS fee policy
            # The below demonstrates a simple approach for reacting to the changing rates in trading logic
            previous_taker_rate = self.accounts[book_id].fees.taker_fee_rate
            # Positive rate implies a fee is to be paid, negative rate results in rebates to the trader
            # Check the rate against the configured tolerance
            if previous_taker_rate > self.max_fee_rate:
                # If taker rate is above tolerance, do not place orders this round
                continue
             
            # Attach a market order instruction in the current trade direction for a random quantity within bounds defined by the parameters
            bt.logging.info(f"BOOK {book_id} | QUOTE : {self.accounts[book_id].quote_balance.total} [LOAN {self.accounts[book_id].quote_loan} | COLLAT {self.accounts[book_id].quote_collateral}]")
            bt.logging.info(f"BOOK {book_id} | BASE : {self.accounts[book_id].base_balance.total} [LOAN {self.accounts[book_id].base_loan} | COLLAT {self.accounts[book_id].base_collateral}]")
            if self.direction[book_id] == OrderDirection.BUY:
                # If in the BUY regime, we place orders randomly with leverage selected from the configured range
                # Obtain a random leverage value if there is no open margin position on sell side
                leverage = self.leverage() if self.accounts[book_id].base_loan == 0 else 0.0
                # If an open opposite margin position exists, repay the corresponding loans in order 
                # from oldest to newest by setting LoanSettlementOption.FIFO
                settlement = LoanSettlementOption.NONE if self.accounts[book_id].base_loan == 0 else LoanSettlementOption.FIFO
                # If placing unleveraged order, increase the quantity to better match the average total size of 
                # leveraged orders on the other side.  This avoids accumulating too much inventory in one currency.
                quantity =  round(self.quantity() * (1 + self.leverage()), self.simulation_config.volumeDecimals)
                response.market_order(
                    book_id=book_id, 
                    direction=self.direction[book_id], 
                    quantity=quantity, 
                    stp=random.choice([STP.DECREASE_CANCEL, STP.CANCEL_OLDEST]),
                    leverage=leverage,
                    settlement_option=settlement
                )
                bt.logging.info(f"SUBMITTING BUY MARKET ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}")
            else:
                # If in the SELL regime, we place orders randomly without leverage, but with quantity increased to match the amounts placed on buy side.
                # Obtain a random leverage value if there is no open margin position on sell side
                leverage = self.leverage() if self.accounts[book_id].quote_loan == 0 else 0.0
                # If an open opposite margin position exists, repay the corresponding loans in order 
                # from oldest to newest by setting LoanSettlementOption.FIFO
                settlement = LoanSettlementOption.NONE if self.accounts[book_id].quote_loan == 0 else LoanSettlementOption.FIFO
                # If placing unleveraged order, increase the quantity to better match the average total size of 
                # leveraged orders on the other side.  This avoids accumulating too much inventory in one currency.
                quantity =  round(self.quantity() * (1 + self.leverage()), self.simulation_config.volumeDecimals)
                response.market_order(
                    book_id=book_id, 
                    direction=self.direction[book_id], 
                    quantity=quantity, 
                    stp=random.choice([STP.DECREASE_CANCEL, STP.CANCEL_OLDEST]),
                    leverage=leverage,
                    settlement_option=settlement
                )
                # bt.logging.info(f"SUBMITTING SELL MARKET ORDER FOR {str(round(1+leverage,2))+'x' if leverage > 0 else ''}{quantity}")
        # Return the response with instructions appended
        # The response will be serialized and sent back to the validator for processing
        return response

if __name__ == "__main__":
    """
    Example command for local standalone testing execution using Proxy:
    python RandomTakerAgent.py --port 8888 --agent_id 0 --params min_quantity=0.1 max_quantity=1.0 min_leverage=0.0 max_leverage=1.0 max_fee_rate=0.002
    """
    launch(RandomTakerAgent)