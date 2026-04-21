"""RTC-specific request payload helpers for chunk scheduling."""

from __future__ import annotations

from collections.abc import Sequence
import warnings

from ....core.errors import InterfaceValidationError
from ....core.schema import Action
from ..protocols import RtcArgs


class ChunkSchedulerRtcOpsMixin:
    """Helpers that build RTC request payloads and validate RTC shape rules."""

    __slots__ = ()

    def _build_rtc_args(
        self,
        *,
        remaining_chunk: Sequence[Action],
        inference_delay: int,
        rtc_seed_chunk: Sequence[Action] | None = None,
    ) -> RtcArgs | None:
        """Build optional RTC hints for one policy request."""

        if not self.enable_rtc:
            return None
        if self.execution_steps is None:
            raise InterfaceValidationError(
                "RTC request construction requires execution_steps."
            )

        source_chunk = remaining_chunk if remaining_chunk else rtc_seed_chunk
        if not source_chunk:
            return None

        prev_action_chunk, execute_horizon = self._build_prev_action_chunk(
            source_chunk=source_chunk,
        )
        return RtcArgs(
            prev_action_chunk=prev_action_chunk,
            inference_delay=min(
                max(int(inference_delay), 1),
                execute_horizon,
            ),
            execute_horizon=execute_horizon,
        )

    def _build_prev_action_chunk(
        self,
        *,
        source_chunk: Sequence[Action],
    ) -> tuple[list[Action], int]:
        """Build one RTC prev-action chunk and its execution horizon."""

        cloned_chunk = self._clone_actions(source_chunk)
        if not cloned_chunk:
            raise InterfaceValidationError(
                "RTC prev_action_chunk source must contain at least one action."
            )
        if self.execution_steps is None:
            raise InterfaceValidationError(
                "RTC prev_action_chunk construction requires execution_steps."
            )

        execute_horizon = self.execution_steps
        window = cloned_chunk[:execute_horizon]
        pad_count = execute_horizon - len(window)
        if pad_count > 0:
            pad_action = self._clone_action(window[0])
            window = [self._clone_action(pad_action) for _ in range(pad_count)] + window
        total_length = (
            self._rtc_chunk_total_length
            if self._rtc_chunk_total_length is not None
            else len(cloned_chunk)
        )
        if total_length < execute_horizon:
            raise InterfaceValidationError(
                "RTC locked chunk_total_length must be >= execution_steps, got "
                f"chunk_total_length={total_length!r}, "
                f"execution_steps={execute_horizon!r}."
            )
        total_pad_count = total_length - len(window)
        if total_pad_count > 0:
            total_pad_action = self._clone_action(window[0])
            window = (
                [self._clone_action(total_pad_action) for _ in range(total_pad_count)]
                + window
            )
        return window, execute_horizon

    def _validate_chunk_length(self, chunk_length: int) -> None:
        """Validate and lock RTC raw chunk length constraints."""

        if not self.enable_rtc:
            return
        self._lock_rtc_chunk_total_length(chunk_length)

    def _lock_rtc_chunk_total_length(self, chunk_length: int) -> None:
        """Lock one fixed RTC raw chunk length and enforce later consistency."""

        if not self.enable_rtc:
            return
        if self.execution_steps is None:
            raise InterfaceValidationError(
                "RTC chunk length locking requires execution_steps."
            )
        if chunk_length <= 0:
            raise InterfaceValidationError(
                f"RTC chunk_total_length must be > 0, got {chunk_length!r}."
            )
        if self._rtc_chunk_total_length is None:
            self._rtc_chunk_total_length = chunk_length
            self._validate_rtc_execution_window_structure(chunk_length)
            return
        if chunk_length != self._rtc_chunk_total_length:
            raise InterfaceValidationError(
                "RTC requires a stable source raw chunk length once the first "
                "chunk is accepted. Got "
                f"chunk_total_length={chunk_length!r}, "
                f"locked_chunk_total_length={self._rtc_chunk_total_length!r}."
            )

    def _validate_rtc_execution_window_structure(
        self,
        chunk_total_length: int,
    ) -> None:
        """Validate the fixed RTC request window against the locked chunk size."""

        if not self.enable_rtc or self.execution_steps is None:
            return
        if self.execution_steps >= (chunk_total_length - self.steps_before_request):
            raise InterfaceValidationError(
                "RTC requires execution_steps < chunk_total_length - "
                "steps_before_request, got "
                f"execution_steps={self.execution_steps!r}, "
                f"chunk_total_length={chunk_total_length!r}, "
                f"steps_before_request={self.steps_before_request!r}."
            )

    def _check_execution_window_delay(
        self,
        *,
        raw_delay_steps: int,
    ) -> None:
        """Warn when predicted raw delay exceeds the RTC execution horizon."""

        if not self.enable_rtc or self.execution_steps is None:
            return
        if raw_delay_steps <= self.execution_steps:
            return

        message = (
            "Estimated raw-step inference delay exceeds execution_steps: "
            f"inference_delay={raw_delay_steps}, "
            f"execution_steps={self.execution_steps}. "
            "The transmitted RTC inference_delay will still be clamped into "
            "[1, execute_horizon]."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
