from __future__ import annotations

import json
import logging
import sys
import time

_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "kv", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        base = f"{ts} {record.levelname.lower():7s} {record.name}: {record.getMessage()}"
        extra = getattr(record, "kv", None)
        if extra:
            base += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(level: str = "WARNING", as_json: bool = False) -> None:
    global _configured
    root = logging.getLogger("kernelpilot")
    root.setLevel(level.upper())
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if as_json else KVFormatter())
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"kernelpilot.{name}")


def kv(**fields: object) -> dict[str, dict[str, object]]:
    """logger.info("compiled", extra=kv(candidate=cid, ms=12.3))"""
    return {"kv": fields}
