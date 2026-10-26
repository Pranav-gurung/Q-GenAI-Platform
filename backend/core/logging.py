"""Structured logging.

`console` format is for humans:   2026-10-26T10:00:00Z INFO  INS-0A1B | CV_PROCESSING | STARTED
`json` format is for log shippers: one JSON object per line with the same fields.

Pipeline code logs through `log_pipeline()` so every pipeline line carries
inspection_id / pipeline_stage / status / duration_ms / error_code.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Fields that may be attached via `extra=` and are emitted as structured fields.
STRUCTURED_FIELDS = ("inspection_id", "pipeline_stage", "status", "duration_ms", "error_code", "pipeline_module",
                     "request_id", "method", "path", "status_code")

PIPELINE_LOGGER = "qgenai.pipeline"


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = request_id_var.get()
        return True


def _ts(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {"timestamp": _ts(record), "level": record.levelname,
                                   "logger": record.name, "message": record.getMessage()}
        for key in STRUCTURED_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        head = f"{_ts(record)} {record.levelname:<5} "
        if getattr(record, "inspection_id", None):
            parts = [record.inspection_id, getattr(record, "pipeline_stage", None) or "-",
                     getattr(record, "status", None) or "-"]
            line = head + " | ".join(parts)
            extras = [f"{k}={getattr(record, k)}" for k in ("pipeline_module", "duration_ms", "error_code")
                      if getattr(record, k, None) is not None]
            msg = record.getMessage()
            if msg:
                extras.append(msg)
            if extras:
                line += " | " + " ".join(str(e) for e in extras)
        else:
            line = head + f"{record.name}: {record.getMessage()}"
            rid = getattr(record, "request_id", None)
            if rid:
                line += f" [request_id={rid}]"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    handler.addFilter(_RequestIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # uvicorn's access log duplicates our request log
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def log_pipeline(inspection_id: str, stage: str, status: str, *, level: int = logging.INFO,
                 duration_ms: float | None = None, error_code: str | None = None,
                 module: str | None = None, message: str = "") -> None:
    logging.getLogger(PIPELINE_LOGGER).log(level, message, extra={
        "inspection_id": inspection_id, "pipeline_stage": stage, "status": status,
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
        "error_code": error_code, "pipeline_module": module,
    })
