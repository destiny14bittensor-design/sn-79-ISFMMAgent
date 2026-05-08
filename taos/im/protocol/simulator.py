# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Simulator response models: serialisable agent response and response-batch
types consumed by the C++ simulator via the validator protocol.
"""
from taos.common.protocol import BaseModel
from typing import Any

class SimulatorAgentResponse(BaseModel):
    """
    Represents a response from an agent.

    Attributes:
        agentId (int): Identifier for the agent sending the response.
        delay (int): Delay to be applied in processing the response.
        type (str): Type of the response.
        payload (dict[str, Any] | None): Additional data related to the response.
    """
    agentId: int
    delay: int
    type: str
    payload: dict[str, Any] | None  

    def serialize(self) -> dict:
        """
        Serializes the response into a dictionary format.
        """
        return {
            "agentId": self.agentId,
            "delay": self.delay,
            "type": self.type,
            "payload": self.payload,
        }

class SimulatorResponseBatch(BaseModel):
    """
    Represents a batch of responses from agents.

    Attributes:
        responses (list[SimulatorAgentResponse]): List of agent responses.
    """
    responses: list[SimulatorAgentResponse]

    def __init__(self, responses: list[SimulatorAgentResponse]):
        """
        Initialise the response batch.

        Args:
            responses (list[SimulatorAgentResponse]): Agent responses to include in the batch.
        """
        instructions = []
        for response in responses:
            if response:
                instructions.extend(response.serialize())
        super().__init__(responses=instructions)

    def serialize(self) -> dict:
        """
        Serialise the batch of responses into a dictionary format.

        Returns:
            dict: Dictionary with a 'responses' key mapping to a list of serialised responses.
        """
        return {
            "responses": [response.serialize() for response in self.responses]
        }