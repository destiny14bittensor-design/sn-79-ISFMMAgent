# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol.models import OrderDirection
from taos.im.protocol.instructions import *
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse

import random

class MaliciousAgent(FinanceSimulationAgent):
    def initialize(self):
        self.min_quantity = self.config.min_quantity
        self.max_quantity = self.config.max_quantity
        self.expiry_period = self.config.expiry_period

    def quantity(self):
        return round(random.uniform(self.min_quantity,self.max_quantity),10)

    def respond(self, state : MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        for book_id, book in state.books.items():
            for order in self.accounts[book_id].orders:
                if state.timestamp > order.timestamp + self.expiry_period:
                    response.cancel_order(book_id=book_id, order_id=order.id)
            if len(book.bids) > 0 and len(book.asks) > 0:
                midquote = (book.bids[0].price+book.asks[0].price)/2
                bidprice = round(random.uniform(book.bids[0].price,book.asks[0].price),8)
                askprice = round(random.uniform(bidprice,book.asks[0].price),8)
            else:
                midquote = 100.0
                bidprice = round(random.uniform(99.95,100.05),8)
                askprice = round(random.uniform(bidprice,100.05),8)
            if bidprice != askprice:
                quantity = self.quantity()
                if self.accounts[book_id].quote_balance.free >= quantity * bidprice:
                    response.add_instruction(PlaceLimitOrderInstruction(agentId=2, direction=OrderDirection.BUY, quantity=quantity, price=bidprice))
                    response.limit_order(direction=OrderDirection.BUY, quantity=-1*quantity, price=bidprice)
                    response.limit_order(direction=OrderDirection.BUY, quantity=quantity, price=-1*bidprice)
                    response.market_order(direction=OrderDirection.BUY, quantity=round(random.uniform(self.min_quantity,self.max_quantity)/10,10))
                else:
                    print(f"Cannot place BUY order for {quantity}@{bidprice} : Insufficient quote balance!")
                if self.accounts[book_id].base_balance.free >= quantity:
                    response.add_instruction(PlaceLimitOrderInstruction(agentId=2, direction=OrderDirection.SELL, quantity=quantity, price=askprice))
                    response.limit_order(direction=OrderDirection.SELL, quantity=-1*quantity, price=askprice)
                    response.limit_order(direction=OrderDirection.SELL, quantity=quantity, price=-1*askprice)
                    response.market_order(direction=OrderDirection.SELL, quantity=round(random.uniform(self.min_quantity,self.max_quantity)/10,10))
                else:
                    print(f"Cannot place SELL order for {quantity}@{askprice} : Insufficient base balance!")
        return response

if __name__ == "__main__":
    launch(MaliciousAgent)