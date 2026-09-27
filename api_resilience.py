import os
import random
import threading
import time
from contextlib import contextmanager

from openai import APIConnectionError, APIStatusError

# 408 timeout, 409 conflict, 425 too early, 429 rate limit, 5xx server errors
TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _env_float(name, default):
    return float(os.getenv(name, default))


class RequestThrottle:
    """Process-wide request gate shared by every thread.

    Caps concurrent requests, spaces out request starts, and lets one thread's
    rate limit pause everyone instead of each thread hammering the API alone.
    """

    def __init__(self, max_concurrency, min_interval):
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._lock = threading.Lock()
        self._min_interval = min_interval
        self._next_start = 0.0

    @contextmanager
    def slot(self):
        with self._slots:
            with self._lock:
                now = time.monotonic()
                start = max(now, self._next_start)
                self._next_start = start + self._min_interval
            if start > now:
                time.sleep(start - now)
            yield

    def cool_down(self, seconds):
        with self._lock:
            self._next_start = max(self._next_start, time.monotonic() + seconds)


THROTTLE = RequestThrottle(
    max_concurrency=int(os.getenv("OLLAMA_MAX_CONCURRENCY", "1")),
    min_interval=_env_float("OLLAMA_MIN_REQUEST_INTERVAL", "2"),
)


def is_transient(error):
    # APIConnectionError also covers timeouts and wrapped transport errors such
    # as httpx RemoteProtocolError ("Server disconnected without sending a response")
    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in TRANSIENT_STATUS_CODES
    return isinstance(error, (ConnectionError, TimeoutError))


def _retry_after_seconds(error):
    response = getattr(error, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def backoff_delay(attempt, base, cap):
    # Exponential backoff with "equal jitter": half fixed, half random, so
    # retries never collapse to ~0s but threads still spread out
    delay = min(cap, base * (2**attempt))
    return delay / 2 + random.uniform(0, delay / 2)


def retry_api_call(func, *args, **kwargs):
    """Call func through the shared throttle, retrying transient API failures.

    Non-transient errors (bad request, auth, unknown model) raise immediately.
    Transient ones are retried with exponential backoff + jitter; rate limits
    also pause all other requests. Raises the last error once retries run out.
    """
    max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "10"))
    base = _env_float("OLLAMA_RETRY_BASE_SECONDS", "5")
    cap = _env_float("OLLAMA_RETRY_MAX_SECONDS", "300")

    for attempt in range(max_retries + 1):
        try:
            with THROTTLE.slot():
                return func(*args, **kwargs)
        except Exception as e:
            if not is_transient(e) or attempt == max_retries:
                raise
            delay = _retry_after_seconds(e) or backoff_delay(attempt, base, cap)
            reason = (
                "Rate limit reached"
                if getattr(e, "status_code", None) == 429
                else f"Transient API error ({type(e).__name__}: {e})"
            )
            print(
                f"{reason}. Pausing all requests for {delay:.0f}s "
                f"(retry {attempt + 1}/{max_retries})..."
            )
            # The next THROTTLE.slot() waits out the cool-down for every thread
            THROTTLE.cool_down(delay)
