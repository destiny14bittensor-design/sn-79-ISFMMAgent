# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Standalone query service using POSIX IPC for communication.
"""

import time
import asyncio
import bittensor as bt
import bittensor.utils.networking as _bt_net
import posix_ipc
import mmap
import struct
import gc
import pickle
import os
import argparse
import traceback
from typing import Dict, Any
from collections import defaultdict
from taos.im.protocol import STP
from taos.im.protocol import MarketSimulationStateUpdate
from taos.im.validator.forward import DendriteManager

class QueryService:
    def __init__(self, config):
        """
        Initialize the standalone validator-side query service.

        This sets up:
        - Wallet and dendrite client for querying miners
        - Service configuration
        - IPC resource placeholders
        - Notification pipe
        - Internal running state

        Args:
            config (bt.Config): The validator configuration object.

        Returns:
            None
        """
        self.config = config
        self.wallet = bt.Wallet(
            path=self.config.wallet.path,
            name=self.config.wallet.name,
            hotkey=self.config.wallet.hotkey
        )
        self.dendrite = bt.Dendrite(wallet=self.wallet)
        # Cache the external IP so subsequent bt.Dendrite() calls in query_miners
        # don't make synchronous HTTP requests to AWS/ipinfo/ifconfig.me.
        _cached_ip = self.dendrite.external_ip
        _bt_net.get_external_ip = lambda: _cached_ip
        self.running = True
        self.request_queue = None
        self.response_queue = None
        self.request_shm = None
        self.response_shm = None
        self.notify_fd = config.notify_fd if hasattr(config, 'notify_fd') else None

    def setup_ipc(self):
        """
        Sets up POSIX IPC message queues and shared memory buffers for
        communication between the validator and the standalone query process.

        Creates:
        - Request message queue
        - Response message queue
        - Shared memory segments for request + response payloads
        - Memory maps for reading/writing SHM

        Raises:
            posix_ipc.Error: If IPC creation fails.

        Returns:
            None
        """
        queue_name = f"/validator_query_{self.config.wallet.hotkey}"

        self.request_queue = posix_ipc.MessageQueue(
            f"{queue_name}_req",
            flags=posix_ipc.O_CREAT,
            max_messages=10,
            max_message_size=1024
        )

        self.response_queue = posix_ipc.MessageQueue(
            f"{queue_name}_res",
            flags=posix_ipc.O_CREAT,
            max_messages=10,
            max_message_size=1024
        )

        self.request_shm = posix_ipc.SharedMemory(
            f"{queue_name}_req_shm",
            flags=posix_ipc.O_CREAT,
            size=500 * 1024 * 1024
        )

        self.response_shm = posix_ipc.SharedMemory(
            f"{queue_name}_res_shm",
            flags=posix_ipc.O_CREAT,
            size=500 * 1024 * 1024
        )

        self.request_mem = mmap.mmap(self.request_shm.fd, self.request_shm.size)
        self.response_mem = mmap.mmap(self.response_shm.fd, self.response_shm.size)

        bt.logging.info(f"IPC setup complete: {queue_name}")

    async def initialize(self):
        """
        Initializes the query service runtime components.

        This includes:
        - Setting up POSIX IPC
        - Ensuring a valid dendrite session is active
        - Sending ready signal via pipe

        Returns:
            None
        """
        self.setup_ipc()
        DendriteManager.configure_session(self)
        if self.notify_fd is not None:
            try:
                os.write(self.notify_fd, b'R')
                bt.logging.info("Query service sent ready signal")
            except Exception as e:
                bt.logging.error(f"Query service failed to send ready signal: {e}\n{traceback.format_exc()}")
        bt.logging.info("Query service initialized")

    def validate_responses(self, synapses: dict, request_data: dict, deregistered_uids: set) -> dict:
        """
        Validates miner responses received through dendrite.

        The validation enforces:
        - Matching agent_id
        - Instruction limits per book
        - Trade volume caps
        - Decompression integrity
        - Instruction structure and field correctness

        Aggregates:
        - Response count
        - Instruction totals
        - Success / timeout / failure counts

        Args:
            synapses (dict[int, MarketSimulationStateUpdate]):
                Raw synapse responses from miners.
            request_data (dict): Original request payload sent to miners.
            deregistered_uids (set[int]): Miners excluded from validation.

        Returns:
            tuple:
                (
                    total_valid_responses (int),
                    total_instructions (int),
                    success_count (int),
                    timeout_count (int),
                    failure_count (int)
                )
        """
        gc.disable()
        try:
            total_responses = 0
            total_instructions = 0
            success = 0
            timeouts = 0
            failures = 0

            miner_wealth = request_data.get('miner_wealth', 1000000)
            volume_decimals = request_data.get('volume_decimals', 2)
            book_count = request_data.get('book_count', len(request_data['books']))
            capital_turnover_cap = request_data.get('capital_turnover_cap', 10.0)
            max_instructions_per_book = request_data.get('max_instructions_per_book', 100)

            volume_cap = round(capital_turnover_cap * miner_wealth, volume_decimals)
            volume_sums = request_data.get('volume_sums', {})

            all_miner_volumes = {}
            for uid in synapses.keys():
                if uid not in deregistered_uids:
                    all_miner_volumes[uid] = {
                        book_id: volume_sums.get(uid, {}).get(book_id, 0.0)
                        for book_id in range(book_count)
                    }

            for uid, synapse in synapses.items():
                if uid in deregistered_uids:
                    continue
                if synapse.is_timeout:
                    timeouts += 1
                    continue
                elif synapse.is_failure:
                    failures += 1
                    continue
                elif not synapse.is_success:
                    failures += 1
                    bt.logging.warning(f"UID {uid} invalid state: {synapse.dendrite.status_message}")
                    continue
                
                success += 1
                
                if synapse.compressed:
                    synapse.decompress()
                    if synapse.compressed:
                        bt.logging.warning(f"Failed to decompress response for {uid}!")
                        continue
                
                if not synapse.response:
                    bt.logging.debug(f"UID {uid} failed to respond: {synapse.dendrite.status_message}")
                    continue
                
                if synapse.response.agent_id != uid:
                    bt.logging.warning(f"Invalid response submitted by agent {uid} (Mismatched Agent Ids)")
                    continue

                miner_volumes = all_miner_volumes[uid]
                
                valid_instructions = []
                instructions_per_book = defaultdict(int)
                invalid_agent_id = False
                volume_cap_logged = False
                
                for instruction in synapse.response.instructions:
                    try:
                        if instruction.agentId != uid or instruction.type == 'RESET_AGENT':
                            bt.logging.warning(f"Invalid instruction submitted by agent {uid} (Mismatched Agent Ids)")
                            invalid_agent_id = True
                            break
                        
                        if instruction.bookId >= book_count:
                            bt.logging.warning(f"Invalid instruction submitted by agent {uid} (Invalid Book Id {instruction.bookId})")
                            continue

                        if miner_volumes[instruction.bookId] >= volume_cap and instruction.type != "CANCEL_ORDERS":
                            if not volume_cap_logged:
                                bt.logging.info(f"Agent {uid} hit volume cap on one or more books")
                                volume_cap_logged = True
                            continue

                        if instruction.type in ['PLACE_ORDER_MARKET', 'PLACE_ORDER_LIMIT']:
                            stp_value = instruction.stp
                            if hasattr(stp_value, 'value'):
                                stp_value = stp_value.value
                            if stp_value == 'NO_STP' or stp_value == 0:
                                instruction.stp = STP.CANCEL_OLDEST

                        instructions_per_book[instruction.bookId] += 1

                        if instructions_per_book[instruction.bookId] <= max_instructions_per_book:
                            valid_instructions.append(instruction)
                            
                    except Exception as ex:
                        bt.logging.warning(f"Error processing instruction by agent {uid}: {ex}\n{instruction}\n{traceback.format_exc()}")
                
                if invalid_agent_id:
                    valid_instructions = []
                
                total_submitted = sum(instructions_per_book.values())
                
                if len(valid_instructions) < total_submitted:
                    bt.logging.warning(
                        f"Agent {uid} sent {total_submitted} instructions "
                        f"(Avg. {total_submitted / len(instructions_per_book):.2f} / book), "
                        f"with more than {max_instructions_per_book} instructions on some books - "
                        f"excess instructions dropped. Final count: {len(valid_instructions)}"
                    )
                
                synapse.response.instructions = valid_instructions
                if valid_instructions:
                    total_responses += 1
                    total_instructions += len(valid_instructions)
            
            return total_responses, total_instructions, success, timeouts, failures
        finally:
            gc.enable()

    async def query_miners(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Issues parallel dendrite requests to all miners and collects results.

        Performs:
        - Reconstruction of metagraph axons
        - Compression of books and synapses
        - Parallel async dendrite calls with global timeout
        - Graceful fallback for timed-out or failed miners
        - Response serialization for IPC transport
        - Delegation to response validation

        Args:
            request_data (dict): A fully prepared simulation state update
                containing books, accounts, notices, metagraph data, settings,
                and compression metadata.

        Returns:
            dict: Structured result object:
                {
                    'success': bool,
                    'responses': dict,
                    'error': str (optional),
                    'traceback': str (optional)
                }
        """
        gc_was_enabled = gc.isenabled()
        old_dendrite = None
        try:
            gc.disable()
            
            old_dendrite = self.dendrite
            self.dendrite = bt.Dendrite(wallet=self.wallet)
            
            class MinimalMetagraph:
                def __init__(self, axons, uids):
                    self.axons = axons
                    self.uids = uids

            axon_list = []
            uid_list = []  # Track actual UIDs, not just sequential indices
            version_split = bt.__version__.split(".")
            _version_info = tuple(int(part) for part in version_split)
            _version_int_base = 1000
            version_as_int: int = sum(
                e * (_version_int_base**i) for i, e in enumerate(reversed(_version_info))
            )

            for uid, axon_data in enumerate(request_data['metagraph_axons']):
                axon = bt.AxonInfo(
                    version=version_as_int,
                    hotkey=axon_data['hotkey'],
                    coldkey=axon_data['coldkey'],
                    ip=axon_data['ip'],
                    port=axon_data['port'],
                    ip_type=axon_data['ip_type'],
                    protocol=axon_data['protocol'],
                    placeholder1=0,
                    placeholder2=0,
                )
                if axon_data['ip'] != "0.0.0.0":
                    axon_list.append(axon)
                    uid_list.append(uid)

            self.metagraph = MinimalMetagraph(axon_list, uid_list)
            deregistered_uids = set(request_data['deregistered_uids'])

            if not uid_list:
                bt.logging.warning(
                    "No miners with reachable axons in metagraph — "
                    "skipping forward (will retry next tick)"
                )
                return {
                    'success': True,
                    'responses': {},
                    'validation_stats': {
                        "total_responses": 0,
                        "total_instructions": 0,
                        "success": 0,
                        "timeouts": 0,
                        "failures": 0,
                    },
                }

            bt.logging.info(
                f"Querying {len(self.metagraph.axons)} miners "
                f"(UIDs: {min(uid_list)}-{max(uid_list)})"
            )

            from taos.im.validator.forward import DendriteManager
            DendriteManager.configure_session(self)

            from taos.im.utils.compress import compress, batch_compress
            import multiprocessing

            compress_start = time.time()
            compressed_books = compress(
                request_data['books'],
                level=self.config.compression.level,
                engine=self.config.compression.engine,
                version=request_data['version'],
            )
            bt.logging.info(f"Compressed books ({time.time()-compress_start:.4f}s).")

            def create_axon_synapse(uid):
                synapse = MarketSimulationStateUpdate.parse_dict(request_data)
                # Benchmark miners (UIDs >= metagraph.n) are not simulation agents,
                # so the simulator never sends account/notice data for them.
                accounts = synapse.accounts or {}
                notices = synapse.notices or {}
                object.__setattr__(synapse, "accounts", {uid: accounts[uid]} if uid in accounts else {})
                object.__setattr__(synapse, "notices", {uid: notices[uid]} if uid in notices else {uid: []})
                object.__setattr__(synapse, "config", request_data['config'])
                synapse.version = request_data['version']
                return synapse

            create_start = time.time()
            axon_synapses = {uid: create_axon_synapse(uid) for uid in uid_list}
            bt.logging.info(f"Created axon synapses ({time.time()-create_start:.4f}s)")

            synapse_start = time.time()
            if self.config.compression.parallel_workers == 0:
                def compress_axon_synapse(synapse):
                    return synapse.compress(
                        level=self.config.compression.level,
                        engine=self.config.compression.engine,
                        compressed_books=compressed_books
                    )
                axon_synapses = {uid: compress_axon_synapse(axon_synapses[uid]) for uid in uid_list}
            else:
                num_processes = self.config.compression.parallel_workers if self.config.compression.parallel_workers > 0 else multiprocessing.cpu_count() // 2
                num_axons = len(uid_list)
                batch_size = max(1, int(num_axons / num_processes))
                batches = [uid_list[i:i+batch_size] for i in range(0, num_axons, batch_size)]
                axon_synapses = batch_compress(
                    axon_synapses,
                    compressed_books,
                    batches,
                    level=self.config.compression.level,
                    engine=self.config.compression.engine,
                    version=request_data['version']
                )
            bt.logging.info(f"Compressed synapses ({time.time()-synapse_start:.4f}s).")

            query_start = time.time()
            synapse_responses = {}

            async def query_uid(index, uid):
                """Query a specific UID at the given axon index."""
                try:
                    response = await self.dendrite(
                        axons=self.metagraph.axons[index],
                        synapse=axon_synapses[uid],
                        timeout=self.config.neuron.timeout,
                        deserialize=False
                    )
                    return uid, response
                except asyncio.CancelledError:
                    axon_synapses[uid] = self.dendrite.preprocess_synapse_for_request(
                        self.metagraph.axons[index],
                        axon_synapses[uid],
                        self.config.neuron.timeout
                    )
                    axon_synapses[uid].dendrite.status_code = 408
                    return uid, axon_synapses[uid]
                except Exception as e:
                    bt.logging.debug(f"Error querying UID {uid}: {e}\n{traceback.format_exc()}")
                    axon_synapses[uid] = self.dendrite.preprocess_synapse_for_request(
                        self.metagraph.axons[index],
                        axon_synapses[uid],
                        self.config.neuron.timeout
                    )
                    axon_synapses[uid].dendrite.status_code = 500
                    return uid, axon_synapses[uid]

            query_tasks = []
            for index, uid in enumerate(uid_list):
                if uid not in deregistered_uids:
                    query_tasks.append(asyncio.create_task(query_uid(index, uid)))

            bt.logging.info(
                f"Created {len(query_tasks)} query tasks, "
                f"starting wait with {self.config.neuron.global_query_timeout}s timeout"
            )

            done, pending = await asyncio.wait(
                query_tasks,
                timeout=self.config.neuron.global_query_timeout,
                return_when=asyncio.ALL_COMPLETED
            )

            elapsed = time.time() - query_start
            if elapsed > self.config.neuron.global_query_timeout:
                bt.logging.warning(
                    f"Query overshot timeout: {elapsed:.4f}s > {self.config.neuron.global_query_timeout}s"
                )
                for task in pending:
                    task.cancel()
                # Drain the cancellations so each task's CancelledError
                # handler runs (and finishes touching axon_synapses) before
                # the outer scope unbinds the closure cell at end-of-function.
                # Without this await, those handlers hit a NameError on
                # axon_synapses[uid] after `del axon_synapses` fires below.
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                pending = set()

            bt.logging.info(f"Wait completed: {len(done)} done, {len(pending)} pending in {elapsed:.4f}s")

            collect_start = time.time()
            completed_count = 0
            for task in done:
                try:
                    uid, response = task.result()
                    synapse_responses[uid] = response
                    completed_count += 1
                except Exception as e:
                    bt.logging.debug(f"Task failed: {e}\n{traceback.format_exc()}")

            if pending:
                bt.logging.warning(f"Cancelling {len(pending)} pending tasks")
                for task in pending:
                    task.cancel()
                # Drain so cancelled tasks' handlers complete before the
                # `del axon_synapses` below unbinds their closure cell.
                await asyncio.gather(*pending, return_exceptions=True)

            missing_count = 0
            for index, uid in enumerate(uid_list):
                if uid not in deregistered_uids and uid not in synapse_responses:
                    axon_synapses[uid] = self.dendrite.preprocess_synapse_for_request(
                        self.metagraph.axons[index],
                        axon_synapses[uid],
                        self.config.neuron.timeout
                    )
                    axon_synapses[uid].dendrite.status_code = 408
                    synapse_responses[uid] = axon_synapses[uid]
                    missing_count += 1

            if missing_count > 0:
                bt.logging.info(f"Filled in {missing_count} missing responses as timeouts")

            bt.logging.info(f"Collected {completed_count} Responses ({time.time()-collect_start:.4f}s)") 

            bt.logging.info(
                f"Dendrite call completed ({time.time()-query_start:.4f}s | "
                f"Timeout {self.config.neuron.timeout}s / {self.config.neuron.global_query_timeout}s). "
                f"Total responses collected: {len(synapse_responses)}"
            )

            validate_start = time.time()
            total_responses, total_instructions, success, timeouts, failures = self.validate_responses(
                synapse_responses,
                request_data,
                deregistered_uids
            )
            bt.logging.info(f"Validated Responses ({time.time()-validate_start:.4f}s).")

            del compressed_books
            del axon_synapses
            del query_tasks
            del done
            del pending
            del self.metagraph
            
            if old_dendrite:
                del old_dendrite

            return {
                'success': True,
                'responses': synapse_responses,
                'validation_stats': {
                    "total_responses": total_responses,
                    "total_instructions": total_instructions,
                    "success": success,
                    "timeouts": timeouts,
                    "failures": failures
                }
            }

        except Exception as e:
            bt.logging.error(f"Error in query_miners: {e}\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        finally:
            if gc_was_enabled:
                gc.enable()


    async def deliver_gentrx_miners(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver GenTRX assignments to miners via parallel raw HTTP posts.

        Mirrors query_miners pattern: all expensive CPU work (header signing,
        JSON serialization) is done synchronously before any tasks are created
        so that tasks contain only async I/O and never block the event loop.

        Args:
            request_data (dict): Contains 'round' and 'deliveries' (list of
                {uid, axon_data, assignment} dicts).

        Returns:
            dict: {'success': bool, 'ok': int, 'fail': int}
        """
        import json as _json
        import aiohttp as _aio
        from taos.im.protocol.gentrx import GenTRXAssignment

        deliveries = request_data.get('deliveries', [])
        round_id = request_data.get('round', '?')

        if not deliveries:
            return {'success': True, 'ok': 0, 'fail': 0}

        gc_was_enabled = gc.isenabled()
        old_dendrite = self.dendrite
        send_ok = 0
        send_fail = 0

        try:
            gc.disable()

            self.dendrite = bt.Dendrite(wallet=self.wallet)
            DendriteManager.configure_session(self)

            version_as_int: int = sum(
                int(p) * (1000 ** i)
                for i, p in enumerate(reversed(bt.__version__.split(".")))
            )

            # Phase 1 (sync, before task creation): build body + headers.
            # The axon's verify fn uses message=nonce.dendrite_key.MINER_KEY.uuid.body_hash
            # so each miner needs its own signature.  We call preprocess once to get
            # a valid nonce/uuid/body_hash/base-headers, then for each miner do only
            # keypair.sign(per_miner_message) — no pydantic model construction, no
            # to_headers(), no model_dump() — dropping per-miner cost to raw crypto.
            prep_start = time.time()
            first_d = deliveries[0]
            first_adat = first_d['axon_data']
            first_axon = bt.AxonInfo(
                version=version_as_int,
                hotkey=first_adat['hotkey'], coldkey=first_adat['coldkey'],
                ip=first_adat['ip'], port=first_adat['port'],
                ip_type=first_adat['ip_type'], protocol=first_adat['protocol'],
                placeholder1=0, placeholder2=0,
            )
            base_synapse = GenTRXAssignment(**first_d['assignment'])
            base_synapse = self.dendrite.preprocess_synapse_for_request(
                first_axon, base_synapse, 5.0
            )
            base_headers = base_synapse.to_headers()
            base_headers['Content-Type'] = 'application/json'
            base_body = _json.dumps(base_synapse.model_dump()).encode('utf-8')
            synapse_name = base_synapse.name

            # Extract signing inputs from the preprocessed synapse — these are
            # constant across all miners for this round.
            dendrite_nonce  = base_synapse.dendrite.nonce
            dendrite_uuid   = base_synapse.dendrite.uuid
            dendrite_hotkey = base_synapse.dendrite.hotkey
            body_hash       = base_synapse.body_hash  # property, matches what preprocess signs with
            bt.logging.info(
                f"[GTX] deliver round={round_id} base prep done in "
                f"{time.time()-prep_start:.3f}s — signing {len(deliveries)} miners"
            )

            t_sign_start = time.time()
            prepared = []
            for d in deliveries:
                uid = d['uid']
                adat = d['axon_data']
                miner_hotkey = adat['hotkey']
                # Sign only — skips pydantic overhead that made per-miner cost ~86ms
                message = (
                    f"{dendrite_nonce}.{dendrite_hotkey}."
                    f"{miner_hotkey}.{dendrite_uuid}.{body_hash}"
                )
                sig = f"0x{self.wallet.hotkey.sign(message).hex()}"
                headers = dict(base_headers)
                headers['bt_header_axon_ip']           = adat['ip']
                headers['bt_header_axon_port']         = str(adat['port'])
                headers['bt_header_axon_hotkey']       = miner_hotkey
                headers['bt_header_dendrite_signature'] = sig
                url = f"http://{adat['ip']}:{adat['port']}/{synapse_name}"
                prepared.append((uid, adat['ip'], adat['port'], url, headers, base_body))
            bt.logging.info(
                f"[GTX] deliver round={round_id} signed {len(prepared)} miners in "
                f"{time.time()-t_sign_start:.3f}s "
                f"(total prep {time.time()-prep_start:.3f}s)"
            )

            # Phase 2 (async): fire all requests concurrently.  Tasks contain
            # only a single session.post() yield — no bittensor overhead inside.
            session = await self.dendrite.session
            per_req_timeout = _aio.ClientTimeout(total=5, sock_connect=1.0)

            async def _fire_one(uid, ip, port, url, headers, body):
                nonlocal send_ok, send_fail
                t0 = time.time()
                try:
                    err_body = ""
                    async with session.post(
                        url, headers=headers, data=body,
                        timeout=per_req_timeout,
                    ) as resp:
                        status = resp.status
                        if status != 200:
                            try:
                                err_body = (await resp.text())[:300]
                            except Exception:
                                pass
                    elapsed = time.time() - t0
                    if status == 200:
                        bt.logging.debug(
                            f"[GTX] deliver round={round_id} uid={uid} "
                            f"{ip}:{port} status={status} t={elapsed:.2f}s"
                        )
                        send_ok += 1
                    else:
                        bt.logging.info(
                            f"[GTX] deliver round={round_id} uid={uid} "
                            f"{ip}:{port} status={status} t={elapsed:.2f}s"
                            + (f" err={err_body!r}" if err_body else "")
                        )
                        send_fail += 1
                except asyncio.CancelledError:
                    bt.logging.warning(
                        f"[GTX] deliver round={round_id} uid={uid} "
                        f"cancelled after {time.time()-t0:.2f}s"
                    )
                    send_fail += 1
                    raise
                except Exception as exc:
                    bt.logging.info(
                        f"[GTX] deliver round={round_id} uid={uid} "
                        f"{ip}:{port} {type(exc).__name__}: {exc} t={time.time()-t0:.2f}s"
                    )
                    send_fail += 1

            tasks = [
                asyncio.create_task(_fire_one(uid, ip, port, url, headers, body))
                for uid, ip, port, url, headers, body in prepared
            ]

            t_start = time.time()
            _, pending = await asyncio.wait(tasks, timeout=30)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            t_total = time.time() - t_start
            bt.logging.info(
                f"[GTX] deliver round={round_id} n={len(deliveries)} "
                f"ok={send_ok} fail={send_fail} t={t_total:.2f}s"
            )
            return {'success': True, 'ok': send_ok, 'fail': send_fail}

        except Exception as e:
            bt.logging.error(f"[GTX] deliver_gentrx_miners error: {e}\n{traceback.format_exc()}")
            return {'success': False, 'error': str(e), 'ok': send_ok, 'fail': send_fail}
        finally:
            if gc_was_enabled:
                gc.enable()
            if old_dendrite:
                del old_dendrite

    async def run(self):
        """
        Main event loop for the standalone query service.

        Responsibilities:
        - Wait for commands from the validator via IPC
        - Read inbound requests from shared memory
        - Execute miner queries and GenTRX deliveries
        - Write results to response shared memory
        - Send acknowledgement signaling readiness
        - Handle shutdown command gracefully

        Returns:
            None
        """
        await self.initialize()

        bt.logging.info("Query service ready, waiting for requests...")
        while True:
            try:
                self.request_queue.receive(timeout=0.0)
                bt.logging.warning("Drained stale message from query request queue")
            except posix_ipc.BusyError:
                break

        while self.running:
            try:
                message, _ = self.request_queue.receive(timeout=1.0)
                receive_time = time.time()
                bt.logging.info(f"Received message at {receive_time}")
                command = message.decode('utf-8')

                if command == 'query':
                    read_start = time.time()
                    bt.logging.info(f"Starting read, {read_start - receive_time:.4f}s after receive")                    
                    self.request_mem.seek(0)
                    seek_time = time.time()
                    bt.logging.info(f"Seek completed in {seek_time - read_start:.4f}s")                    
                    size_bytes = self.request_mem.read(8)
                    size_read_time = time.time()
                    bt.logging.info(f"Read size in {size_read_time - seek_time:.4f}s")                    
                    data_size = struct.unpack('Q', size_bytes)[0]
                    request_bytes = self.request_mem.read(data_size)
                    data_read_time = time.time()
                    bt.logging.info(f"Read {data_size} bytes in {data_read_time - size_read_time:.4f}s")
                    
                    request_data = pickle.loads(request_bytes)
                    bt.logging.info(f"Read Query request data ({time.time()-read_start:.4f}s).")

                    result = await self.query_miners(request_data)
                    del request_data

                    write_start = time.time()
                    result_bytes = pickle.dumps(result, protocol=5)
                    del result
                    self.response_mem.seek(0)
                    self.response_mem.write(struct.pack('Q', len(result_bytes)))
                    self.response_mem.write(result_bytes)
                    self.response_mem.flush()
                    del result_bytes
                    
                    bt.logging.info(f"Wrote Query response data ({time.time()-write_start:.4f}s).")
                    
                    if self.notify_fd is not None:
                        try:
                            os.write(self.notify_fd, b'1')
                            bt.logging.info("Sent query completion notification")
                        except Exception as e:
                            bt.logging.error(f"Failed to send notification: {e}\n{traceback.format_exc()}")
                    else:
                        bt.logging.error("Cannot send notification - notify_fd is None!")
                    
                    gc_start = time.time()
                    gc.collect(generation=2)
                    bt.logging.info(f"Query GC completed in {time.time()-gc_start:.4f}s")
                elif command == 'deliver_gentrx':
                    read_start = time.time()
                    self.request_mem.seek(0)
                    size_bytes = self.request_mem.read(8)
                    data_size = struct.unpack('Q', size_bytes)[0]
                    request_bytes = self.request_mem.read(data_size)
                    request_data = pickle.loads(request_bytes)
                    bt.logging.info(
                        f"[GTX] deliver_gentrx: read {data_size} bytes "
                        f"({time.time()-read_start:.4f}s)"
                    )

                    result = await self.deliver_gentrx_miners(request_data)
                    del request_data

                    write_start = time.time()
                    result_bytes = pickle.dumps(result, protocol=5)
                    del result
                    self.response_mem.seek(0)
                    self.response_mem.write(struct.pack('Q', len(result_bytes)))
                    self.response_mem.write(result_bytes)
                    self.response_mem.flush()
                    del result_bytes
                    bt.logging.info(
                        f"[GTX] deliver_gentrx: wrote response ({time.time()-write_start:.4f}s)"
                    )

                    if self.notify_fd is not None:
                        try:
                            os.write(self.notify_fd, b'G')
                            bt.logging.info("[GTX] Sent deliver_gentrx completion notification")
                        except Exception as e:
                            bt.logging.error(
                                f"[GTX] Failed to send delivery notification: {e}\n"
                                f"{traceback.format_exc()}"
                            )
                    else:
                        bt.logging.error("[GTX] Cannot send notification - notify_fd is None!")

                    gc.collect(generation=2)
                elif command == 'shutdown':
                    bt.logging.info("Shutdown command received")
                    self.running = False

            except posix_ipc.BusyError:
                await asyncio.sleep(0.01)
            except Exception as e:
                bt.logging.error(f"Error in main loop: {e}\n{traceback.format_exc()}")
                bt.logging.error(traceback.format_exc())

        self.cleanup()

    def cleanup(self):
        """
        Cleans up all POSIX IPC resources used by the query service.

        Actions:
        - Close mmap buffers
        - Close and unlink shared memory segments
        - Close and unlink message queues

        Safe to call multiple times.

        Returns:
            None
        """
        try:
            if self.request_mem:
                self.request_mem.close()
            if self.response_mem:
                self.response_mem.close()
            if self.request_shm:
                self.request_shm.close_fd()
                self.request_shm.unlink()
            if self.response_shm:
                self.response_shm.close_fd()
                self.response_shm.unlink()
            if self.request_queue:
                self.request_queue.close()
                self.request_queue.unlink()
            if self.response_queue:
                self.response_queue.close()
                self.response_queue.unlink()
        except Exception as e:
            bt.logging.error(f"Error cleaning up IPC: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.Axon.add_args(parser)
    bt.logging.set_info()

    parser.add_argument('--netuid', type=int, default=1)
    parser.add_argument('--logging.level', type=str, default="info")
    parser.add_argument('--neuron.timeout', type=float, default=3.0)
    parser.add_argument('--neuron.global_query_timeout', type=float, default=4.0)
    parser.add_argument('--compression.level', type=int, default=1)
    parser.add_argument('--compression.engine', type=str, default='zlib')
    parser.add_argument('--compression.parallel_workers', type=int, default=0)
    parser.add_argument('--cpu-cores', type=str, default=None)    
    parser.add_argument('--notify-fd', type=int, default=None)
    

    config = bt.Config(parser)
    bt.logging(config=config)

    if config.cpu_cores:
        cores = [int(c) for c in config.cpu_cores.split(',')]
        os.sched_setaffinity(0, set(cores))
        bt.logging.info(f"Query service assigned to cores: {cores}")

    service = QueryService(config)

    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        bt.logging.info("Query service interrupted")
    except Exception as e:
        bt.logging.error(f"Query service crashed: {e}\n{traceback.format_exc()}")
        bt.logging.error(traceback.format_exc())