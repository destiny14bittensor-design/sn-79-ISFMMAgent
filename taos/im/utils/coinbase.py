# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Coinbase WebSocket client wrapper: patches open_async and _is_websocket_open
for compatibility with the current websockets library version.
"""
import websockets
import asyncio
import ssl

import websockets

from coinbase.api_base import get_logger
from coinbase.constants import USER_AGENT

logger = get_logger("coinbase.WSClient")
from coinbase.websocket.websocket_base import WSBase, WSClientException


class CoinbaseClient(WSBase):
    """Override the base Coinbase WebSocket client for compatibility with the latest websockets library."""
    async def open_async(self) -> None:
        """
        Open the websocket client connection asynchronously.  
        This is overridden here to provide compatibility with the latest websockets library.
        """
        self._ensure_websocket_not_open()

        headers = self._set_headers()

        logger.debug("Connecting to %s", self.base_url)
        try:
            self.websocket = await websockets.connect(
                self.base_url,
                open_timeout=self.timeout,
                max_size=self.max_size,
                user_agent_header=USER_AGENT,
                additional_headers=headers,
                ssl=ssl.SSLContext() if self.base_url.startswith("wss://") else None,
            )
            logger.debug("Successfully connected to %s", self.base_url)

            if self.on_open:
                self.on_open()

            # Start the message handler coroutine after establishing connection
            if not self._retrying:
                self._task = asyncio.create_task(self._message_handler())

        except asyncio.TimeoutError as toe:
            self.websocket = None
            logger.error("Connection attempt timed out: %s", toe)
            raise WSClientException("Connection attempt timed out") from toe
        except (websockets.exceptions.WebSocketException, OSError) as wse:
            self.websocket = None
            logger.error("Failed to establish WebSocket connection: %s", wse)
            raise WSClientException("Failed to establish WebSocket connection") from wse
        
    def _is_websocket_open(self) -> bool:
        """
        Checks and returns boolean indicating if the websocket is open.
        Overridden here as necessary for proper function of the library.
        """
        return self.websocket and self.websocket.state == websockets.protocol.State.OPEN