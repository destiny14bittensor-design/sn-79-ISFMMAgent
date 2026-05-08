# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Finance agent instruction classes: limit/market order placement, order
cancellation, position closing, and agent reset for the intelligent markets protocol.
"""
from pydantic import PositiveFloat, NonNegativeInt, PositiveInt, NonNegativeFloat, Field
from typing import Literal, Annotated
from taos.im.protocol.simulator import *
from taos.common.protocol import AgentInstruction, BaseModel
from taos.im.protocol.models import OrderDirection, STP, TimeInForce, OrderCurrency, LoanSettlementOption

UInt32 = Annotated[int, Field(ge=0, le=2**32 - 1)]

class FinanceAgentInstruction(AgentInstruction):
    """
    Base class representing an instruction submitted by an agent in an intelligent markets simulation.

    Attributes:
        agentId (int): The ID of the agent that submitted the instruction.
        delay (NonNegativeInt): The processing delay to be assigned to the instruction. 
            This is set by validators based on the actual response time of the miner, and determines 
            how many simulation steps will elapse after submission before the agent instruction is processed.
        type (Literal["PLACE_ORDER_MARKET", "PLACE_ORDER_LIMIT", "CANCEL_ORDERS", "CLOSE_POSITIONS", "RESET_AGENT"]): 
            String identifier for the type of the submitted instruction in the simulator.
    """
    agentId: UInt32
    delay: NonNegativeInt = 0
    type: Literal["PLACE_ORDER_MARKET", "PLACE_ORDER_LIMIT", "CANCEL_ORDERS", "CLOSE_POSITIONS", "RESET_AGENT"]
    
    def serialize(self) -> dict:
        return {
            "agentId": self.agentId,
            "delay": self.delay,
            "type": self.type,
            "payload": self.payload()
        }
    
    def __str__(self):
        return f"{self.type} ON BOOK {self.bookId} : {self.payload()}"
    
class PlaceOrderInstruction(FinanceAgentInstruction):
    """
    Base class representing an instruction by an agent to place an order.

    Attributes:
        bookId (UInt32): The ID of the book on which the order is to be placed.
        direction (Literal[OrderDirection.BUY, OrderDirection.SELL]): Indicates whether the order is to buy or sell.
        quantity (PositiveFloat): The size of the order to be placed in base currency.
        clientOrderId (UInt32 | None): User-assigned client ID associated with the order.
        stp (Literal[STP.CANCEL_OLDEST, STP.CANCEL_NEWEST, STP.CANCEL_BOTH, STP.DECREASE_CANCEL]): 
            Self-trade prevention strategy to be applied for the order.
        currency (Literal[OrderCurrency.BASE, OrderCurrency.QUOTE]): Currency in which the quantity is specified (BASE or QUOTE).
        leverage (NonNegativeFloat) : The amount of leverage to take for a margin order; the effective order quantity will be `(1+leverage)`
            e.g. an order placed for 1.0 BASE with 0.5 leverage will be placed for a total quantity of 1.5 BASE, where 0.5 is borrowed from the exchange.
        settleFlag (Literal[LoanSettlementOption.NONE, LoanSettlementOption.FIFO] | NonNegativeInt):
            Strategy for settling outstanding margin loans using the proceeds of this order
            LoanSettlementOption.NONE : No loan repayments
            LoanSettlementOption.FIFO : Loans will be repaid, starting from the oldest
            NonNegativeInt : Specify a specific order id for which the associated loan will be repaid
    """
    bookId: UInt32
    direction: Literal[OrderDirection.BUY, OrderDirection.SELL]
    quantity: PositiveFloat
    clientOrderId: UInt32 | None
    stp: Literal[STP.CANCEL_OLDEST, STP.CANCEL_NEWEST, STP.CANCEL_BOTH, STP.DECREASE_CANCEL] = STP.CANCEL_OLDEST
    currency: Literal[OrderCurrency.BASE, OrderCurrency.QUOTE] = OrderCurrency.BASE
    leverage: NonNegativeFloat = 0.0
    settleFlag: Literal[LoanSettlementOption.NONE, LoanSettlementOption.FIFO] | NonNegativeInt = LoanSettlementOption.NONE

    
    def __str__(self):
        return f"{'BUY ' if self.direction == OrderDirection.BUY else 'SELL'} {self.quantity} ON BOOK {self.bookId}"
    
class PlaceMarketOrderInstruction(PlaceOrderInstruction):
    """
    Class representing an instruction by an agent to place a market order.

    Attributes:
        type (Literal['PLACE_ORDER_MARKET']): Fixed to 'PLACE_ORDER_MARKET'.
    """
    type: Literal['PLACE_ORDER_MARKET'] = 'PLACE_ORDER_MARKET'

    def payload(self) -> dict:
        return {
            "direction": self.direction,
            "volume": self.quantity,
            "bookId":self.bookId,
            "clientOrderId":self.clientOrderId,
            "stpFlag":self.stp,
            "currency":self.currency,
            "leverage":self.leverage,
            "settleFlag":self.settleFlag
        }
    
    def __str__(self):
        return f"{'BUY ' if self.direction == OrderDirection.BUY else 'SELL'} {f'{1+self.leverage:.2f}x' if self.leverage > 0 else ''}{self.quantity}{'' if self.currency==OrderCurrency.BASE else 'QUOTE'}@MARKET ON BOOK {self.bookId}"
        
class PlaceLimitOrderInstruction(PlaceOrderInstruction):
    """
    Class representing an instruction by an agent to place a limit order.

    Attributes:
        type (Literal['PLACE_ORDER_LIMIT']): Fixed to 'PLACE_ORDER_LIMIT'.
        price (PositiveFloat): The price level at which the order is to be placed.
        postOnly (bool): Boolean flag specifying if the order should be placed with Post-Only enforcement.
        timeInForce (Literal[TimeInForce.GTC, TimeInForce.GTT, TimeInForce.IOC, TimeInForce.FOK]): 
            Time-In-Force option to be applied for the order.
        expiryPeriod (PositiveInt | None): The period in simulation time after which the order should 
            be cancelled (valid only with `timeInForce = TimeInForce.GTT`).
    """
    type: Literal['PLACE_ORDER_LIMIT'] = 'PLACE_ORDER_LIMIT'
    price: PositiveFloat
    postOnly: bool = False
    timeInForce: Literal[TimeInForce.GTC, TimeInForce.GTT, TimeInForce.IOC, TimeInForce.FOK] = TimeInForce.GTC
    expiryPeriod: PositiveInt | None = None

    def payload(self) -> dict:
        return {
            "direction": self.direction,
            "volume": self.quantity,
            "price": self.price,
            "bookId": self.bookId,
            "clientOrderId":self.clientOrderId,
            "postOnly" : self.postOnly,
            "timeInForce" : self.timeInForce,
            "expiryPeriod" : self.expiryPeriod,
            "stpFlag" : self.stp,
            "leverage":self.leverage,
            "settleFlag":self.settleFlag
        }
    
    def __str__(self):
        return f"{'BUY ' if self.direction == OrderDirection.BUY else 'SELL'} {f'{1+self.leverage:.2f}x' if self.leverage > 0 else ''}{self.quantity}@{self.price} ON BOOK {self.bookId}"
    
class CancelOrderInstruction(BaseModel):
    """
    Class representing an instruction by an agent to cancel an open limit order.

    Attributes:
        orderId (UInt32): The simulator-assigned ID of the order to be cancelled.
        volume (PositiveFloat | None): The quantity of the order that should be cancelled 
            (`None` to cancel the entire remaining order size).
    """
    orderId: UInt32
    volume: PositiveFloat | None

    def serialize(self) -> dict:
        return {
            "orderId" : self.orderId,
            "volume" : self.volume
        }
    
    def __str__(self):
        return f"CANCEL ORDER #{self.orderId}{' FOR ' + str(self.volume) if self.volume else ''}"
        
class CancelOrdersInstruction(FinanceAgentInstruction):
    """
    Class representing an instruction by an agent to cancel a list of open limit orders.

    Attributes:
        type (Literal['CANCEL_ORDERS']): Fixed to 'CANCEL_ORDERS'.
        bookId (UInt32): The ID of the book on which cancellations are to be performed.
        cancellations (list[CancelOrderInstruction]): A list of CancelOrderInstruction objects.
    """
    type: Literal['CANCEL_ORDERS'] = 'CANCEL_ORDERS'
    bookId: UInt32
    cancellations: list[CancelOrderInstruction]

    def payload(self) -> dict:
        return {
            "cancellations": [cancellation.serialize() for cancellation in self.cancellations],
            "bookId": self.bookId
        }
    
    def __str__(self):
        return "\n".join([f"{c} ON BOOK {self.bookId}" for c in self.cancellations])
    
class ClosePositionInstruction(BaseModel):
    """
    Class representing an instruction by an agent to close a margin position.

    Attributes:
        orderId (UInt32): The simulator-assigned ID of the order for which position is to be closed.
        volume (PositiveFloat | None): The quantity to be closed
            (`None` to close entire remaining position).
    """
    orderId: UInt32
    volume: PositiveFloat | None

    def serialize(self) -> dict:
        return {
            "orderId" : self.orderId,
            "volume" : self.volume
        }
    
    def __str__(self):
        return f"CLOSE POSITION FOR ORDER #{self.orderId}{' FOR ' + str(self.volume) if self.volume else ''}"
        
class ClosePositionsInstruction(FinanceAgentInstruction):
    """
    Class representing an instruction by an agent to close margin positions associated with a list of orders.

    Attributes:
        type (Literal['CLOSE_POSITIONS']): Fixed to 'CLOSE_POSITIONS'.
        bookId (UInt32): The ID of the book on which closures are to be performed.
        closes (list[ClosePositionInstruction]): A list of ClosePositionInstruction objects.
    """
    type: Literal['CLOSE_POSITIONS'] = 'CLOSE_POSITIONS'
    bookId: UInt32
    closes: list[ClosePositionInstruction]

    def payload(self) -> dict:
        return {
            "closes": [close_position.serialize() for close_position in self.closes],
            "bookId": self.bookId
        }
    
    def __str__(self):
        return "\n".join([f"{c} ON BOOK {self.bookId}" for c in self.closes])
        
class ResetAgentsInstruction(FinanceAgentInstruction):
    """
    Class representing an instruction to reset an agent's accounts.
    This instruction can only be submitted by validators to handle deregistration of a miner.

    Attributes:
        type (Literal['RESET_AGENT']): Fixed to 'RESET_AGENT'.
        agentIds (list[UInt32]): List of IDs of the agents for which reset should be applied.
    """
    type: Literal['RESET_AGENT'] = 'RESET_AGENT'
    agentIds: list[UInt32]

    def payload(self) -> dict:
        return {
            "agentIds": self.agentIds
        }
    
    def __str__(self):
        return f"RESET AGENTS {','.join(['#' + str(agentId) for agentId in self.agentIds])}"