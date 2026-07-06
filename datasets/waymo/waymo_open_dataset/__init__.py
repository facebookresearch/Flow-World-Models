"""Helpers for importing the vendored Waymo proto definitions.

These proto files were generated with an older ``protoc`` version that still
relies on runtime behaviors removed from ``protobuf`` 4.21+.  When a user has a
newer ``protobuf`` runtime installed, importing the generated modules raises
``TypeError: Descriptors cannot be created directly`` before the converter even
starts.  The official workaround (per the protobuf release notes) is to force
the pure-Python implementation by setting ``PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python``.

To keep the repository self-contained, we automatically flip that environment
variable when we detect an incompatible protobuf runtime and the user has not
explicitly opted into another implementation.
"""

from __future__ import annotations

import logging
import os
from importlib import metadata

LOGGER = logging.getLogger(__name__)


def _needs_python_proto_impl() -> bool:
    """Return True if the environment should force the Python proto runtime."""
    # Respect any user-provided override so advanced users can experiment with
    # different implementations.
    if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"):
        return False

    try:
        version_str = metadata.version("protobuf")
    except metadata.PackageNotFoundError:
        # Without protobuf installed there is nothing to patch.
        return False

    # ``protobuf`` started enforcing the regenerated descriptor format in 4.21.
    # Using the Python implementation keeps our vendored descriptors working.
    major = int(version_str.split(".", 1)[0])
    return major >= 4


if _needs_python_proto_impl():
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    LOGGER.info(
        "Detected protobuf>=4; forcing PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "
        "to keep the vendored Waymo proto definitions compatible. This only affects "
        "the current Python process."
    )
