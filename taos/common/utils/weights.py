# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""
Weight conversion utilities: normalise and clamp reward weight tensors, convert
between float representations and the u16 on-chain encoding.
"""

import logging
import typing
from typing import Union, Optional

import numpy as np

from numpy.typing import NDArray

from bittensor.utils.btlogging import logging
from bittensor.utils.registration import legacy_torch_api_compat, torch, use_torch

if typing.TYPE_CHECKING:
    from bittensor.core.metagraph import Metagraph
    from bittensor.core.subtensor import Subtensor


U32_MAX = 4294967295
U16_MAX = 65535

@legacy_torch_api_compat
def normalize_max_weight(
    x: Union[NDArray[np.float32], "torch.FloatTensor"], limit: float = 0.1
) -> Union[NDArray[np.float32], "torch.FloatTensor"]:
    """
    Normalise a weight tensor so that sum(x) = 1 and no single entry exceeds `limit`.

    Args:
        x (np.float32 or torch.FloatTensor): Weight tensor to normalise.
        limit (float): Maximum allowed weight for any single entry after normalisation.
            Defaults to 0.1.

    Returns:
        np.float32 or torch.FloatTensor: Normalised weight tensor of the same type as `x`.
    """
    epsilon = 1e-7  # For numerical stability after normalization

    weights = x.copy()
    values = np.sort(weights)

    if x.sum() == 0 or x.shape[0] * limit <= 1:
        return np.ones_like(x) / x.shape[0]
    else:
        estimation = values / values.sum()

        if estimation.max() <= limit:
            return weights / weights.sum()

        # Find the cumulative sum and sorted tensor
        cumsum = np.cumsum(estimation, 0)

        # Determine the index of cutoff
        estimation_sum = np.array(
            [(len(values) - i - 1) * estimation[i] for i in range(len(values))]
        )
        n_values = (estimation / (estimation_sum + cumsum + epsilon) < limit).sum()

        # Determine the cutoff based on the index
        cutoff_scale = (limit * cumsum[n_values - 1] - epsilon) / (
            1 - (limit * (len(estimation) - n_values))
        )
        cutoff = cutoff_scale * values.sum()

        # Applying the cutoff
        weights[weights > cutoff] = cutoff

        y = weights / weights.sum()

        return y

def convert_weights_and_uids_for_emit(
    uids: Union[NDArray[np.int64], "torch.LongTensor"],
    weights: Union[NDArray[np.float32], "torch.FloatTensor"],
) -> tuple[list[int], list[int]]:
    """Converts weights into integer u32 representation that sum to MAX_INT_WEIGHT.

    Args:
        uids (np.int64):Tensor of uids as destinations for passed weights.
        weights (np.float32):Tensor of weights.

    Returns:
        weight_uids (list[int]): Uids as a list.
        weight_vals (list[int]): Weights as a list.
    """
    # Checks.
    weights = weights.tolist()
    uids = uids.tolist()
    if min(weights) < 0:
        raise ValueError(f"Passed weight is negative cannot exist on chain {weights}")
    if min(uids) < 0:
        raise ValueError(f"Passed uid is negative cannot exist on chain {uids}")
    if len(uids) != len(weights):
        raise ValueError(
            f"Passed weights and uids must have the same length, got {len(uids)} and {len(weights)}"
        )
    if sum(weights) == 0:
        return [], []  # Nothing to set on chain.
    else:
        max_weight = float(max(weights))
        weights = [
            float(value) / max_weight for value in weights
        ]  # max-upscale values (max_weight = 1).

    weight_vals = []
    weight_uids = []
    for i, (weight_i, uid_i) in enumerate(list(zip(weights, uids))):
        uint16_val = round(
            float(weight_i) * int(U16_MAX)
        )  # convert to int representation.

        # Filter zeros
        if uint16_val != 0:  # Filter zeros
            weight_vals.append(uint16_val)
            weight_uids.append(uid_i)

    return weight_uids, weight_vals


# The community uses / bittensor does not
def process_weights_for_netuid(
    uids: Union[NDArray[np.int64], "torch.Tensor"],
    weights: Union[NDArray[np.float32], "torch.Tensor"],
    netuid: int,
    subtensor: "Subtensor",
    metagraph: Optional["Metagraph"] = None,
    exclude_quantile: int = 0,
) -> Union[
    tuple["torch.Tensor", "torch.FloatTensor"],
    tuple[NDArray[np.int64], NDArray[np.float32]],
]:
    """
    Processes weight tensors for a given subnet id using the provided weight and UID arrays, applying constraints and normalization based on the subtensor and metagraph data. This function can handle both NumPy arrays and PyTorch tensors.

    Args:
        uids (Union[NDArray[np.int64], "torch.Tensor"]): Array of unique identifiers of the neurons.
        weights (Union[NDArray[np.float32], "torch.Tensor"]): Array of weights associated with the user IDs.
        netuid (int): The network uid to process weights for.
        subtensor (Subtensor): Subtensor instance to access blockchain data.
        metagraph (Optional[Metagraph]): Metagraph instance for additional network data. If None, it is fetched from the subtensor using the netuid.
        exclude_quantile (int): Quantile threshold for excluding lower weights. Defaults to ``0``.

    Returns:
        Union[tuple["torch.Tensor", "torch.FloatTensor"], tuple[NDArray[np.int64], NDArray[np.float32]]]: tuple containing the array of user IDs and the corresponding normalized weights. The data type of the return matches the type of the input weights (NumPy or PyTorch).
    """

    logging.debug("process_weights_for_netuid()")
    logging.debug(f"weights={weights}")
    logging.debug(f"netuid={netuid}")
    logging.debug(f"subtensor={subtensor}")
    logging.debug(f"metagraph={metagraph}")

    # Get latest metagraph from chain if metagraph is None.
    if metagraph is None:
        metagraph = subtensor.metagraph(netuid)

    # Cast weights to floats.
    if use_torch():
        if not isinstance(weights, torch.FloatTensor):
            weights = weights.type(torch.float32)
    else:
        if not isinstance(weights, np.float32):
            weights = weights.astype(np.float32)

    # Network configuration parameters from an subtensor.
    # These parameters determine the range of acceptable weights for each neuron.
    quantile = exclude_quantile / U16_MAX
    min_allowed_weights = subtensor.min_allowed_weights(netuid=netuid)
    max_weight_limit = subtensor.max_weight_limit(netuid=netuid)
    logging.debug(f"quantile={quantile}")
    logging.debug(f"min_allowed_weights={min_allowed_weights}")
    logging.debug(f"max_weight_limit={max_weight_limit}")

    # Find all non zero weights.
    non_zero_weight_idx = (
        torch.argwhere(weights > 0).squeeze(dim=1)
        if use_torch()
        else np.argwhere(weights > 0).squeeze(axis=1)
    )
    non_zero_weight_uids = uids[non_zero_weight_idx]
    non_zero_weights = weights[non_zero_weight_idx]
    nzw_size = non_zero_weights.numel() if use_torch() else non_zero_weights.size
    if nzw_size == 0 or metagraph.n < min_allowed_weights:
        logging.warning("No non-zero weights returning all ones.")
        final_weights = (
            torch.ones((metagraph.n)).to(metagraph.n) / metagraph.n
            if use_torch()
            else np.ones((metagraph.n), dtype=np.int64) / metagraph.n
        )
        logging.debug(f"final_weights={final_weights}")
        final_weights_count = (
            torch.tensor(list(range(len(final_weights))))
            if use_torch()
            else np.arange(len(final_weights))
        )
        return (
            (final_weights_count, final_weights)
            if use_torch()
            else (final_weights_count, final_weights)
        )

    elif nzw_size < min_allowed_weights:
        logging.warning(
            "No non-zero weights less then min allowed weight, returning all ones."
        )
        # ( const ): Should this be np.zeros( ( metagraph.n ) ) to reset everyone to build up weight?
        weights = (
            torch.ones((metagraph.n)).to(metagraph.n) * 1e-5
            if use_torch()
            else np.ones((metagraph.n), dtype=np.int64) * 1e-5
        )  # creating minimum even non-zero weights
        weights[non_zero_weight_idx] += non_zero_weights
        logging.debug(f"final_weights={weights}")
        normalized_weights = normalize_max_weight(x=weights, limit=max_weight_limit)
        nw_arange = (
            torch.tensor(list(range(len(normalized_weights))))
            if use_torch()
            else np.arange(len(normalized_weights))
        )
        return nw_arange, normalized_weights

    logging.debug(f"non_zero_weights={non_zero_weights}")

    # Compute the exclude quantile and find the weights in the lowest quantile
    max_exclude = max(0, len(non_zero_weights) - min_allowed_weights) / len(
        non_zero_weights
    )
    exclude_quantile = min([quantile, max_exclude])
    lowest_quantile = (
        non_zero_weights.quantile(exclude_quantile)
        if use_torch()
        else np.quantile(non_zero_weights, exclude_quantile)
    )
    logging.debug(f"max_exclude={max_exclude}")
    logging.debug(f"exclude_quantile={exclude_quantile}")
    logging.debug(f"lowest_quantile={lowest_quantile}")

    # Exclude all weights below the allowed quantile.
    non_zero_weight_uids = non_zero_weight_uids[lowest_quantile <= non_zero_weights]
    non_zero_weights = non_zero_weights[lowest_quantile <= non_zero_weights]
    logging.debug(f"non_zero_weight_uids={non_zero_weight_uids}")
    logging.debug(f"non_zero_weights={non_zero_weights}")

    # Normalize weights and return.
    normalized_weights = normalize_max_weight(
        x=non_zero_weights, limit=max_weight_limit
    )
    logging.debug(f"final_weights={normalized_weights}")

    return non_zero_weight_uids, normalized_weights