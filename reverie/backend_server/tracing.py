"""
OpenTelemetry-style tracing infrastructure for generative agents.
Provides structured tracing with trace IDs and spans for agent decision flows.
Uses print-based logging for integration with existing observability stack.
No external dependencies required.
"""

import uuid
import functools
import sys
import threading
from contextvars import ContextVar
from typing import Optional, Callable, Any, Dict, List
from datetime import datetime
from enum import Enum

trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
span_stack_var: ContextVar[List[str]] = ContextVar("span_stack", default=None)


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARN"
    ERROR = "ERROR"


TRACING_ENABLED = True
TRACE_LOCK = threading.Lock()


def _get_span_stack() -> List[str]:
    stack = span_stack_var.get()
    if stack is None:
        stack = []
        span_stack_var.set(stack)
    return stack


def generate_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def generate_span_id() -> str:
    return uuid.uuid4().hex[:8]


def get_current_trace_id() -> Optional[str]:
    return trace_id_var.get()


def get_current_span_id() -> Optional[str]:
    stack = _get_span_stack()
    return stack[-1] if stack else None


def get_trace_depth() -> int:
    return len(_get_span_stack())


class TraceContext:
    __slots__ = ('trace_id', 'span_id', '_token_trace', '_token_span', '_prev_stack')

    def __init__(self, trace_id: Optional[str] = None, span_id: Optional[str] = None):
        self.trace_id = trace_id or generate_trace_id()
        self.span_id = span_id or generate_span_id()
        self._token_trace = None
        self._token_span = None
        self._prev_stack = None

    def __enter__(self):
        self._token_trace = trace_id_var.set(self.trace_id)
        stack = _get_span_stack()
        self._prev_stack = list(stack)
        stack.append(self.span_id)
        span_stack_var.set(stack)
        return self

    def __exit__(self, *args):
        if self._token_trace:
            trace_id_var.reset(self._token_trace)
        if self._token_span:
            span_stack_var.reset(self._token_span)


def _print_trace(
    level: LogLevel,
    message: str,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    depth: int = 0,
    persona_name: Optional[str] = None,
    module: Optional[str] = None,
    event: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    file=sys.stdout
):
    if not TRACING_ENABLED:
        return

    tid = trace_id or get_current_trace_id()
    sid = span_id or get_current_span_id()
    depth = depth or get_trace_depth()

    indent = "  " * depth

    parts = [f"[{level.value}]"]
    if module:
        parts[0] = f"[{level.value}][{module}]"
    if tid:
        parts.append(f"[trace:{tid}]")
    if sid:
        parts.append(f"[span:{sid}]")
    if persona_name:
        parts.append(f"[{persona_name}]")
    if depth > 0:
        parts.append(f"[depth:{depth}]")

    if event:
        parts.append(f"event={event}")
    if attributes:
        for k, v in attributes.items():
            if k not in ('trace.id', 'span.id', 'depth', 'persona_name', 'module', 'event'):
                parts.append(f"{k}={v}")
    if duration_ms is not None:
        parts.append(f"duration_ms={duration_ms:.2f}")
    if error:
        parts.append(f"error={error}")

    parts.append(f"{indent}{message}")

    with TRACE_LOCK:
        print(" ".join(parts), file=file)


def trace_span(
    name: str,
    module: Optional[str] = None,
    persona_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_duration: bool = True
):
    return _TraceSpan(name, module, persona_name, attributes, record_duration)


class _TraceSpan:
    def __init__(
        self,
        name: str,
        module: Optional[str],
        persona_name: Optional[str],
        attributes: Optional[Dict[str, Any]],
        record_duration: bool
    ):
        self.name = name
        self.module = module
        self.persona_name = persona_name
        self.attributes = attributes or {}
        self.record_duration = record_duration
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.ctx: Optional[TraceContext] = None

    def __enter__(self):
        self.start_time = datetime.now()
        trace_id = get_current_trace_id() or generate_trace_id()
        span_id = generate_span_id()
        self.ctx = TraceContext(trace_id, span_id)
        self.ctx.__enter__()

        _print_trace(
            LogLevel.INFO,
            f"→ {self.name}",
            trace_id=trace_id,
            span_id=span_id,
            depth=get_trace_depth(),
            persona_name=self.persona_name,
            module=self.module,
            event="span_start",
            attributes=self.attributes
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration_ms = None
        if self.start_time and self.record_duration:
            delta = self.end_time - self.start_time
            duration_ms = delta.total_seconds() * 1000

        error_str = None
        if exc_type:
            error_str = f"{exc_type.__name__}: {exc_val}"

        trace_id = get_current_trace_id()
        span_id = get_current_span_id()

        if error_str:
            _print_trace(
                LogLevel.ERROR,
                f"← {self.name}",
                trace_id=trace_id,
                span_id=span_id,
                depth=get_trace_depth(),
                persona_name=self.persona_name,
                module=self.module,
                event="span_end",
                duration_ms=duration_ms,
                error=error_str
            )
        else:
            _print_trace(
                LogLevel.INFO,
                f"← {self.name}",
                trace_id=trace_id,
                span_id=span_id,
                depth=get_trace_depth(),
                persona_name=self.persona_name,
                module=self.module,
                event="span_end",
                duration_ms=duration_ms
            )

        if self.ctx:
            self.ctx.__exit__(exc_type, exc_val, exc_tb)

        return False


def traced(
    name: Optional[str] = None,
    module: Optional[str] = None,
    persona_name: Optional[str] = None,
    attributes: Optional[dict] = None
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__qualname__
            with trace_span(span_name, module, persona_name, attributes):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    raise

        return wrapper
    return decorator


def log_trace(
    message: str,
    level: LogLevel = LogLevel.INFO,
    module: Optional[str] = None,
    persona_name: Optional[str] = None,
    event: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
):
    _print_trace(
        level,
        message,
        trace_id=get_current_trace_id(),
        span_id=get_current_span_id(),
        depth=get_trace_depth(),
        persona_name=persona_name,
        module=module,
        event=event,
        attributes=attributes,
        error=error
    )


def log_trace_event(
    name: str,
    module: Optional[str] = None,
    persona_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    _print_trace(
        LogLevel.INFO,
        f"Event: {name}",
        trace_id=get_current_trace_id(),
        span_id=get_current_span_id(),
        depth=get_trace_depth(),
        persona_name=persona_name,
        module=module,
        event=name,
        attributes=attributes
    )


def set_tracing_enabled(enabled: bool):
    global TRACING_ENABLED
    TRACING_ENABLED = enabled


def is_tracing_enabled() -> bool:
    return TRACING_ENABLED
