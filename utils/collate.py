from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate


def future_latents_collate(batch: list[Any]) -> Any:
    """
    Custom collate function for flow matching training.

    Returns a dictionary with:
    - Batched tensors where possible (video, metadata tensors, etc.)
    - Recursively collated nested structures

    Output format:
    {
        'video': tensor[B, T, C, H, W],
        'metadata': {
            'image_ids': tensor[B, T],
            'original_height': tensor[B, T],
            ...
        }
    }
    """
    elem = batch[0]

    # Handle tensors
    if isinstance(elem, torch.Tensor):
        return default_collate(batch)

    # Handle dictionaries
    if isinstance(elem, Mapping):
        collated = {}
        for key in elem:
            values = [sample[key] for sample in batch]
            # Recursively collate everything
            collated[key] = future_latents_collate(values)

        return collated

    # Handle sequences (lists, tuples, but not strings/bytes)
    if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
        # Check if all elements have the same length
        lengths = {len(item) for item in batch}
        if len(lengths) > 1:
            # Variable length - cannot batch
            return list(batch)

        try:
            return default_collate(batch)
        except (RuntimeError, TypeError):
            # If default collate fails, return as list
            return list(batch)

    # For primitive types, try default collate
    try:
        return default_collate(batch)
    except (RuntimeError, TypeError):
        # Fallback to list
        return list(batch)
