"""Control-loop pacing helpers for runtime inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
import time
import warnings

from ...core.errors import InterfaceValidationError
from ...shared.common import validate_positive_number


@dataclass(slots=True)
class RealtimeController:
    """Simple pacing helper for a target-hz control loop."""

    hz: float | None = None
    period_s: float | None = None
    warning_interval_s: float = 5.0
    clock: Callable[[], float] = time.perf_counter
    sleeper: Callable[[float], None] = time.sleep
    warning_emitter: Callable[[str], None] | None = None
    _start_time: float | None = field(default=None, init=False, repr=False)
    _step_index: int = field(default=1, init=False, repr=False)
    _last_warning_time: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the target control rate."""

        if self.hz is None and self.period_s is None:
            raise InterfaceValidationError(
                "RealtimeController requires either hz or period_s."
            )
        if self.hz is not None and self.period_s is not None:
            raise InterfaceValidationError(
                "RealtimeController accepts hz or period_s, not both."
            )

        if self.hz is None:
            validated_period = validate_positive_number(self.period_s, "period_s")
            self.period_s = validated_period
            self.hz = 1.0 / validated_period
        else:
            self.hz = validate_positive_number(self.hz, "hz")
            self.period_s = 1.0 / self.hz

        self.warning_interval_s = validate_positive_number(
            self.warning_interval_s,
            "warning_interval_s",
        )

    def reset(self) -> None:
        """Reset the timing schedule."""

        self._start_time = None
        self._step_index = 1
        self._last_warning_time = None

    def _emit_warning(self, message: str) -> None:
        """Emit one throttled warning message."""

        if self.warning_emitter is not None:
            self.warning_emitter(message)
            return
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def wait(self) -> float:
        """Sleep until the next control deadline and return the sleep time."""

        now = float(self.clock())
        if not math.isfinite(now):
            raise InterfaceValidationError(
                f"clock() must return a finite float, got {now!r}."
            )

        if self._start_time is None:
            self._start_time = now

        target_time = self._start_time + self._step_index * self.period_s
        remaining = target_time - now
        slept = 0.0
        if remaining > 0.0:
            self.sleeper(remaining)
            slept = remaining
        else:
            behind_s = -remaining
            if (
                self._last_warning_time is None
                or now - self._last_warning_time >= self.warning_interval_s
            ):
                self._emit_warning(
                    "RealtimeController cannot maintain the target "
                    f"{self.hz:.3f} Hz loop; behind by {behind_s:.6f} s."
                )
                self._last_warning_time = now

        now_after = float(self.clock())
        elapsed = max(now_after - self._start_time, 0.0)
        elapsed_index = int(elapsed // self.period_s) + 1
        self._step_index = max(self._step_index + 1, elapsed_index)
        return slept


__all__ = ["RealtimeController"]
