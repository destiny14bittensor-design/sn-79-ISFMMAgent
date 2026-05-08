"""[GTX]-prefixed logging shim for bt-hosted processes.

Exports `gtx_log` with the same surface as `bt.logging`
(`.info` / `.warning` / `.error` / `.debug` / `.success`). Records are
routed through `bt.logging` with `prefix="[GTX]"` so they pick up bt's
formatter + level config — grep-friendly for operators, no f-string
prefix noise at call sites.

When bittensor isn't importable (pytest runs without bt, the standalone
gradient server in proxy-test mode), we fall back to a plain Python
logger under the `GenTRX` namespace and render the prefix inline. Same
API either way.

Why a shim instead of `logging.getLogger("GenTRX")` everywhere: validator
and miner processes already load bt and configure `bt.logging` as their
primary sink. Using a second logger tree meant either duplicating every
GenTRX record into two streams (the old `_BtLoggingHandler` forwarder) or
silently dropping messages when bt reconfigured the root logger. The
shim makes bt the single sink in bt processes.
"""
from __future__ import annotations

import logging

try:  # bt is only present in validator/miner/proxy processes
    import bittensor as bt
    _bt = bt.logging
except Exception:  # ImportError, or bittensor's own VersionError at import
    _bt = None

_py = logging.getLogger("GenTRX")


class _GTXLog:
    """Thin wrapper that prepends `[GTX]` via bt.logging's native prefix arg."""

    _PREFIX = "[GTX]"

    def _emit(self, level: str, msg: object, *args, **kwargs) -> None:
        # Pre-format %-args ourselves — bt.logging's signature is
        # info(msg='', prefix='', suffix='', *args), so positional args after
        # `msg` would bind to `prefix`. Supporting `gtx_log.info("x=%d", n)`
        # requires formatting before handing off.
        if args and isinstance(msg, str):
            try:
                msg = msg % args
            except Exception:
                msg = f"{msg} {args}"
        # Bake the prefix into the message itself (rather than bt's `prefix`
        # parameter) so downstream handler filters can match on
        # record.getMessage(). bt's Formatter renders `prefix` at format time,
        # which runs AFTER handler filters — filters never see the prefix
        # if we use bt's native arg.
        msg = f"{self._PREFIX} {msg}"
        if _bt is not None:
            getattr(_bt, level)(msg, **kwargs)
            return
        py_level = "info" if level == "success" else level
        getattr(_py, py_level)(msg, **kwargs)

    def info(self, msg: object, *a, **kw) -> None:
        self._emit("info", msg, *a, **kw)

    def warning(self, msg: object, *a, **kw) -> None:
        self._emit("warning", msg, *a, **kw)

    def error(self, msg: object, *a, **kw) -> None:
        self._emit("error", msg, *a, **kw)

    def debug(self, msg: object, *a, **kw) -> None:
        self._emit("debug", msg, *a, **kw)

    def success(self, msg: object, *a, **kw) -> None:
        self._emit("success", msg, *a, **kw)


gtx_log = _GTXLog()
