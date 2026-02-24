"""
Structured Logging for Grafana Loki

All logs are JSON with consistent, queryable fields.
Query logs by: module, action, claim_id, etc.

GRAFANA LOKI QUERIES
====================
# All errors
{project="spin-cycle"} | json | level="ERROR"

# Research failures
{project="spin-cycle"} | json | module="research" action="agent_failed"

# Track a single claim end-to-end
{project="spin-cycle"} | json | claim_id="<uuid>"

# All workflow starts
{project="spin-cycle"} | json | action=~".*_start"

# Judge verdicts
{project="spin-cycle"} | json | module="judge" action="done"

# Tool invocations
{project="spin-cycle"} | json | module="tools"

# Monitor LLM latency
{project="spin-cycle"} | json | action="llm_response"

USAGE
=====
from src.utils.logging import log, get_logger, configure_logging

# In activities (pass activity.logger for Temporal context):
log.info(activity.logger, "decompose", "start", "Decomposing claim",
         claim_id=claim_id, claim=claim_text[:80])

log.error(activity.logger, "judge", "parse_failed", "Failed to parse verdict",
          error=str(e), raw=raw[:200])

# In workflows (pass workflow.logger):
log.info(workflow.logger, "workflow", "started", "Verification started",
         claim_id=claim_id, sub_claims=3)

# In infrastructure code (no activity/workflow context):
logger = get_logger()
log.info(logger, "worker", "ready", "Worker listening", task_queue="spin-cycle-verify")

ACTION NAMING
=============
Consistent suffixes for queryable actions:
  *_start     — beginning of an operation
  *_done      — successful completion
  *_failed    — error/failure
  *_skipped   — intentionally skipped
  *_fallback  — falling back to alternative path
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """JSON formatter for Grafana Loki. Strips Temporal context dicts."""

    TEMPORAL_CONTEXT = re.compile(r"\s*\(\{'.+\}\)\s*$")

    def __init__(self, pretty: bool = False):
        super().__init__()
        self.pretty = pretty

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        msg = self.TEMPORAL_CONTEXT.sub("", msg)

        # Structured log (emitted via StructuredLogger)
        if hasattr(record, "_structured") and record._structured:
            data = {
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "level": record.levelname,
                "module": record._module,
                "action": record._action,
                "msg": msg,
            }
            for key, value in record._extra.items():
                if value is not None:
                    data[key] = value

            if self.pretty:
                return self._pretty(data)
            return json.dumps(data, default=str, separators=(",", ":"))

        # Legacy/third-party log — wrap in JSON so Promtail can still parse it
        if self.pretty:
            return msg
        return json.dumps(
            {
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "level": record.levelname,
                "module": "legacy",
                "action": "log",
                "msg": msg,
            },
            default=str,
            separators=(",", ":"),
        )

    def _pretty(self, data: dict) -> str:
        """Human-readable format for development."""
        ts = data["ts"][11:23]  # Extract HH:MM:SS.mmm from ISO timestamp
        lvl = data["level"][0]  # I/W/E/D
        mod = data["module"].upper()[:10].ljust(10)
        act = data["action"]
        msg = data["msg"]

        skip = {"ts", "level", "module", "action", "msg"}
        ctx = " ".join(f"{k}={v}" for k, v in data.items() if k not in skip)

        return f"{ts} {lvl} [{mod}] {act}: {msg}" + (f" | {ctx}" if ctx else "")


class StructuredLogger:
    """
    Centralized structured logging.

    All methods accept a logger (activity.logger, workflow.logger, or a
    stdlib logging.Logger), module name, action name, message, and
    arbitrary context fields.
    """

    def _log(
        self,
        logger: logging.Logger,
        level: int,
        module: str,
        action: str,
        msg: str,
        **kwargs,
    ) -> None:
        """Emit a structured log."""
        extra = {
            "_structured": True,
            "_module": module,
            "_action": action,
            "_extra": {k: v for k, v in kwargs.items() if v is not None},
        }
        logger.log(level, msg, extra=extra)

    def info(
        self,
        logger: logging.Logger,
        module: str,
        action: str,
        msg: str,
        **kwargs,
    ) -> None:
        """Log INFO level."""
        self._log(logger, logging.INFO, module, action, msg, **kwargs)

    def warning(
        self,
        logger: logging.Logger,
        module: str,
        action: str,
        msg: str,
        **kwargs,
    ) -> None:
        """Log WARNING level."""
        self._log(logger, logging.WARNING, module, action, msg, **kwargs)

    def error(
        self,
        logger: logging.Logger,
        module: str,
        action: str,
        msg: str,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log ERROR level."""
        self._log(
            logger, logging.ERROR, module, action, msg,
            error=error, error_type=error_type, **kwargs,
        )

    def debug(
        self,
        logger: logging.Logger,
        module: str,
        action: str,
        msg: str,
        **kwargs,
    ) -> None:
        """Log DEBUG level."""
        self._log(logger, logging.DEBUG, module, action, msg, **kwargs)


# Singleton instance — import this everywhere
log = StructuredLogger()

# Fallback logger for infrastructure code that doesn't have
# access to activity.logger or workflow.logger
_fallback_logger = None


def get_logger() -> logging.Logger:
    """Get a fallback logger for infrastructure code."""
    global _fallback_logger
    if _fallback_logger is None:
        _fallback_logger = logging.getLogger("spin-cycle.infra")
    return _fallback_logger


def configure_logging() -> None:
    """Configure root logger with structured formatter. Call once at startup.

    Reads from environment:
      LOG_FORMAT: "json" (default, for Loki) or "pretty" (for development)
      LOG_LEVEL: "INFO" (default), "DEBUG", "WARNING", "ERROR"
    """
    pretty = os.environ.get("LOG_FORMAT", "json") == "pretty"
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter(pretty=pretty))

    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True,
    )

    # Temporal loggers — keep activity/workflow at INFO, silence the chatty ones
    logging.getLogger("temporalio.activity").setLevel(logging.INFO)
    logging.getLogger("temporalio.workflow").setLevel(logging.INFO)
    logging.getLogger("temporalio.worker").setLevel(logging.WARNING)
    logging.getLogger("temporalio.client").setLevel(logging.WARNING)
    logging.getLogger("temporalio.service").setLevel(logging.WARNING)
    logging.getLogger("temporalio.bridge").setLevel(logging.WARNING)

    # LangChain / LangGraph — extremely chatty at DEBUG
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langchain_core").setLevel(logging.WARNING)
    logging.getLogger("langchain_openai").setLevel(logging.WARNING)
    logging.getLogger("langgraph").setLevel(logging.WARNING)

    # HTTP clients
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # SQLAlchemy
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    # Other noisy libs
    logging.getLogger("asyncio").setLevel(logging.WARNING)
