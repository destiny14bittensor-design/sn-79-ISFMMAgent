# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""
Intelligent markets miner neuron: serves FinanceAgentResponse synapses in
response to MarketSimulationStateUpdate queries from the validator.
"""

# Cap glibc malloc arenas to bound RSS growth. Each training-cycle thread
# otherwise gets its own arena (default 8×CPU) and arenas are never returned
# to the OS, so per-cycle thread creation leaks ~1.2 GB of anonymous CPU
# memory per round. Re-exec once with MALLOC_ARENA_MAX set so glibc sees it
# before allocating any arenas. Only triggers on direct script execution.
if __name__ == "__main__":
    import os, sys
    if os.environ.get("MALLOC_ARENA_MAX") is None:
        os.environ["MALLOC_ARENA_MAX"] = "2"
        os.execvp(sys.executable, [sys.executable] + sys.argv)

if __name__ != "__mp_main__":
    import time
    import asyncio
    import typing
    import traceback
    import concurrent.futures
    import bittensor as bt

    from taos.common.neurons.miner import BaseMinerNeuron
    from taos.im.protocol import MarketSimulationStateUpdate
    from taos.im.protocol.gentrx import GenTRXAssignment

    class Miner(BaseMinerNeuron):

        def __init__(self):
            super().__init__()
            # GenTRX: wire chain access onto the agent so that
            # _get_aggregator_store_for_assignment can do on-chain bucket
            # discovery.  The base agent class receives no subtensor/metagraph
            # at construction time, so we set them here after super().__init__()
            # has established the chain connection.
            self.agent.subtensor = self.subtensor
            self.agent.metagraph = self.metagraph
            self.agent.config.netuid = self.config.netuid

            # Re-run model bootstrap now that chain is available.
            _gtx = getattr(self.agent, "_gtx", None)
            if _gtx is not None and _gtx.model is None:
                self.agent._ensure_model_version()

            # GenTRX: attach assignment handler so validators can push training
            # assignments via dendrite.  Ignored silently by non-GenTRX agents
            # (forward checks for _pending_assignment attribute before setting).
            self.axon.attach(
                forward_fn=self.forward_gentrx_assignment,
                blacklist_fn=self.blacklist_gentrx_assignment,
                priority_fn=self.priority_gentrx_assignment,
            )

            # GenTRX: commit S3 bucket credentials on-chain so validators can
            # discover where to fetch our gradients. Hard fail if env vars are
            # set but the chain commitment fails — running without commitment
            # means the validator can't find us = no point continuing.
            self._commit_gentrx_bucket()

            # Single-worker thread pool: offloads CPU-bound handle() from the
            # asyncio event loop so Nonce verification is never blocked, while
            # serialising calls to keep ISFMMAgent thread-safe (the agent has
            # extensive shared mutable state and is not designed for concurrent
            # access across multiple simultaneous handle() invocations).
            self._handle_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="handle"
            )

        def _commit_gentrx_bucket(self) -> None:
            """Commit GenTRX S3 bucket credentials on-chain.

            Skips silently if GENTRX_AGENT_S3_BUCKET is not set (miner not
            participating in GenTRX). Hard-fails if commitment fails.
            """
            try:
                from GenTRX.src.chain import BucketInfo, GenTRXChain
            except ImportError:
                bt.logging.debug("GenTRX not installed — skipping bucket commitment")
                return

            bucket_info = BucketInfo.from_env()
            if bucket_info is None:
                bt.logging.info(
                    "GenTRX env vars not set — skipping bucket commitment "
                    "(miner not participating in GenTRX)"
                )
                return

            try:
                chain = GenTRXChain(self.subtensor, self.config.netuid, self.metagraph)
                chain.commit_bucket(self.wallet, bucket_info)
                bt.logging.info(
                    f"GenTRX bucket committed on-chain: account={bucket_info.account_id}"
                )
            except Exception as exc:
                # Don't crash the miner — chain commit failure (rate limit,
                # invalid transaction, transient RPC) is recoverable. The
                # validator will discover the bucket on a later resync once
                # the commit eventually lands. Log and continue.
                bt.logging.warning(
                    f"GenTRX bucket commitment failed (will retry on next start): {exc}"
                )

        async def forward_gentrx_assignment(
            self, synapse: GenTRXAssignment
        ) -> GenTRXAssignment:
            """Receive a GenTRX training assignment from the validator."""
            try:
                bt.logging.info(
                    f"[GTX] assignment received: round={synapse.round} "
                    f"model_version={synapse.model_version} "
                    f"books={synapse.books} "
                    f"data={len(synapse.data)} files "
                    f"ts={synapse.ts_start}..{synapse.ts_end} "
                    f"validator_uid={synapse.validator_uid} "
                    f"source={synapse.data_source}"
                )
                gtx = getattr(self.agent, "_gtx", None)
                if gtx is None:
                    bt.logging.debug("[GTX] agent has no _gtx — not a GenTRX agent, ignoring assignment")
                    return synapse
                gtx.pending_assignments.append({
                    "round":         synapse.round,
                    "model_version": synapse.model_version,
                    "books":         synapse.books,
                    "ts_start":      synapse.ts_start,
                    "ts_end":        synapse.ts_end,
                    "data":          synapse.data,
                    "data_source":   synapse.data_source,
                    "data_endpoint": synapse.data_endpoint,
                    "data_bucket":   synapse.data_bucket,
                    "data_access_key": synapse.data_access_key,
                    "data_secret_key": synapse.data_secret_key,
                    "validator_uid": synapse.validator_uid,
                })
                bt.logging.info(
                    f"[GTX] assignment queued: round={synapse.round} "
                    f"pending={len(gtx.pending_assignments)}"
                )
            except Exception:
                bt.logging.error(f"[GTX] forward_gentrx_assignment failed:\n{traceback.format_exc()}")
                raise
            return synapse

        def blacklist_gentrx_assignment(
            self, synapse: GenTRXAssignment
        ) -> typing.Tuple[bool, str]:
            return self.blacklist(synapse)

        def priority_gentrx_assignment(self, synapse: GenTRXAssignment) -> float:
            return self.priority(synapse)

        async def forward(
            self, synapse: MarketSimulationStateUpdate
        ) -> MarketSimulationStateUpdate:
            """
            Processes incoming market simulation state synapse by forwarding to the associated agent class for handling.

            Args:
                synapse (taos.im.protocol.MarketSimulationStateUpdate): The synapse object containing the latest simulation state update.

            Returns:
                taos.im.protocol.MarketSimulationStateUpdate: The synapse object with the 'response' field updated with any instructions generated by the agent.
            """
            start = time.time()
            synapse.decompress(lazy=self.config.agent.params.lazy_load)
            bt.logging.debug(f"Decompressed ({time.time() - start}s)")
            loop = asyncio.get_running_loop()
            synapse.response = await loop.run_in_executor(
                self._handle_executor, self.agent.handle, synapse
            )
            start = time.time()
            compressed = synapse.clear_inputs().compress()
            bt.logging.debug(f"Compressed ({time.time() - start}s)")
            return compressed
        
        def blacklist_forward(
            self, synapse: MarketSimulationStateUpdate
        ) -> typing.Tuple[bool, str]:
            """
            Apply default blacklisting to all received market simulation state synapses.
            
            Args:
                synapse (taos.im.protocol.MarketSimulationStateUpdate): The synapse object containing the latest simulation state update.

            Returns:
                (bool, str): Tuple containing [1] boolean indicating if the request was blacklisted [2] string containing the message indicating reason for blacklisting.
            """
            return self.blacklist(synapse)
        
        def priority_forward(self, synapse: MarketSimulationStateUpdate) -> float:
            """
            Apply default prioritization to all received simulation state synapses.
            
            Args:
                synapse (taos.im.protocol.MarketSimulationStateUpdate): The synapse object containing the latest simulation state update.

            Returns:
                float: A priority score calculated using the standard priority function.
            """
            return self.priority(synapse)

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            time.sleep(5)
