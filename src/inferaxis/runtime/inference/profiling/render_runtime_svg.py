"""SVG rendering for live runtime inference profiles."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from .render_common import _format_profile_value

if TYPE_CHECKING:
    from .models import RuntimeInferenceProfile


def _runtime_action_channels(
    profile: "RuntimeInferenceProfile",
) -> list[tuple[str, str, list[float | None]]]:
    """Return action command channels in stable target/dimension order."""

    steps = profile.action_steps
    keys: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str, int]] = set()
    for step in steps:
        for command in step.action_commands:
            for dim_index, _ in enumerate(command.value):
                key = (command.target, command.command, dim_index)
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)

    channels: list[tuple[str, str, list[float | None]]] = []
    for target, command_name, dim_index in keys:
        values: list[float | None] = []
        for step in steps:
            value: float | None = None
            for command in step.action_commands:
                if command.target != target or command.command != command_name:
                    continue
                if dim_index < len(command.value):
                    value = command.value[dim_index]
                break
            values.append(value)
        channels.append((f"{target}[{dim_index}]", command_name, values))
    return channels


def _runtime_chunk_action_channel_keys(
    profile: "RuntimeInferenceProfile",
) -> list[tuple[str, str, int]]:
    """Return stable channel keys present in returned chunk actions."""

    keys: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str, int]] = set()
    for action in profile.chunk_actions:
        for command in action.commands:
            for dim_index, _ in enumerate(command.value):
                key = (command.target, command.command, dim_index)
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
    return keys


def _runtime_combined_step_trace(
    profile: "RuntimeInferenceProfile",
    channels: list[tuple[str, str, list[float | None]]],
    *,
    y_start: int,
    width: int,
) -> tuple[str, int]:
    """Render buffer size and action values on one shared x/y chart."""

    steps = profile.action_steps
    margin_left = 72
    margin_right = 210
    chart_top = y_start + 66
    chart_width = width - margin_left - margin_right
    chart_height = 260
    footer_height = 42
    section_height = 66 + chart_height + footer_height
    max_action_channels = 8
    plotted_channels = channels[:max_action_channels]

    fragments: list[str] = [
        f'<text x="24" y="{y_start + 24}" font-size="16" font-weight="700" '
        'fill="#111827">Step Trace</text>',
        f'<text x="24" y="{y_start + 46}" font-size="12" fill="#4b5563">'
        "x=runtime step, y=numeric value. Buffer size and action values share one axis.</text>",
    ]

    if not steps:
        fragments.append(
            f'<text x="24" y="{y_start + 74}" font-size="12" fill="#6b7280">'
            "No emitted actions were recorded.</text>"
        )
        return "".join(fragments), section_height

    series: list[tuple[str, list[float | None], str, str]] = [
        (
            "buffer_size",
            [
                None if step.buffer_size is None else float(step.buffer_size)
                for step in steps
            ],
            "#111827",
            "6 4",
        )
    ]
    palette = [
        "#2563eb",
        "#059669",
        "#d97706",
        "#7c3aed",
        "#dc2626",
        "#0891b2",
        "#4d7c0f",
        "#be123c",
    ]
    for index, (label, _command_name, values) in enumerate(plotted_channels):
        series.append((label, values, palette[index % len(palette)], ""))

    finite_values = [
        value
        for _label, values, _color, _dash in series
        for value in values
        if value is not None
    ]
    if not finite_values:
        finite_values = [0.0]
    low = min(finite_values)
    high = max(finite_values)
    if low == high:
        padding = max(abs(low) * 0.1, 1.0)
        low -= padding
        high += padding
    else:
        padding = (high - low) * 0.08
        low -= padding
        high += padding
    span = high - low

    def x_of(step_index: int) -> float:
        if len(steps) <= 1:
            return margin_left + chart_width / 2.0
        return margin_left + chart_width * step_index / float(len(steps) - 1)

    def y_of(value: float) -> float:
        return chart_top + chart_height * (1.0 - ((value - low) / span))

    chart_bottom = chart_top + chart_height
    fragments.append(
        f'<line x1="{margin_left}" y1="{chart_bottom:.2f}" '
        f'x2="{margin_left + chart_width}" y2="{chart_bottom:.2f}" '
        'stroke="#111827" stroke-width="1.2" />'
    )
    fragments.append(
        f'<line x1="{margin_left}" y1="{chart_top}" '
        f'x2="{margin_left}" y2="{chart_bottom:.2f}" '
        'stroke="#111827" stroke-width="1.2" />'
    )

    y_tick_count = 4
    for tick_index in range(y_tick_count + 1):
        ratio = tick_index / y_tick_count
        value = low + span * ratio
        y = y_of(value)
        fragments.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" '
            f'x2="{margin_left + chart_width}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        fragments.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" font-size="10" '
            'text-anchor="end" fill="#4b5563">'
            f"{_format_profile_value(value)}</text>"
        )

    x_tick_count = min(max(len(steps) - 1, 1), 8)
    for tick_index in range(x_tick_count + 1):
        step_index = round((len(steps) - 1) * tick_index / x_tick_count)
        x = x_of(step_index)
        fragments.append(
            f'<line x1="{x:.2f}" y1="{chart_top}" x2="{x:.2f}" '
            f'y2="{chart_bottom:.2f}" stroke="#f3f4f6" stroke-width="1" />'
        )
        fragments.append(
            f'<text x="{x:.2f}" y="{chart_bottom + 18:.2f}" font-size="10" '
            'text-anchor="middle" fill="#4b5563">'
            f"{step_index}</text>"
        )

    show_point_values = len(steps) <= 12
    legend_x = margin_left + chart_width + 22
    legend_y = chart_top + 4
    for series_index, (label, values, color, dash) in enumerate(series):
        points = [
            f"{x_of(step_index):.2f},{y_of(value):.2f}"
            for step_index, value in enumerate(values)
            if value is not None
        ]
        if points:
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            fragments.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.2"'
                f'{dash_attr} points="{" ".join(points)}" />'
            )
        for step_index, value in enumerate(values):
            if value is None:
                continue
            x = x_of(step_index)
            y = y_of(value)
            fragments.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.1" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1" />'
            )
            if show_point_values:
                fragments.append(
                    f'<text x="{x:.2f}" y="{y - 7:.2f}" font-size="9" '
                    'text-anchor="middle" fill="#374151">'
                    f"{_format_profile_value(value)}</text>"
                )

        row_y = legend_y + series_index * 18
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        fragments.append(
            f'<line x1="{legend_x}" y1="{row_y:.2f}" '
            f'x2="{legend_x + 28}" y2="{row_y:.2f}" '
            f'stroke="{color}" stroke-width="2.2"{dash_attr} />'
        )
        fragments.append(
            f'<text x="{legend_x + 36}" y="{row_y + 4:.2f}" '
            'font-size="10" fill="#374151">'
            f"{escape(label)}</text>"
        )

    if len(channels) > max_action_channels:
        fragments.append(
            f'<text x="{legend_x}" y="{legend_y + len(series) * 18 + 8:.2f}" '
            'font-size="10" fill="#6b7280">'
            f"+{len(channels) - max_action_channels} action channels below/in JSON</text>"
        )

    fragments.append(
        f'<text x="{margin_left + chart_width / 2.0:.2f}" '
        f'y="{chart_bottom + 36:.2f}" font-size="11" '
        'text-anchor="middle" fill="#111827">runtime step</text>'
    )
    fragments.append(
        f'<text x="18" y="{chart_top - 10}" font-size="11" fill="#111827">'
        "value</text>"
    )

    return "".join(fragments), section_height


def _runtime_action_trace_section(
    profile: "RuntimeInferenceProfile",
    *,
    y_start: int,
    width: int,
) -> tuple[str, int]:
    """Render emitted action values as per-channel step traces."""

    steps = profile.action_steps
    margin_left = 236
    margin_right = 176
    chart_width = width - margin_left - margin_right
    header_height = 76
    row_height = 44
    footer_height = 34
    combined_y = y_start + 54
    max_channels = 16
    channels = _runtime_action_channels(profile)
    combined_section, combined_height = _runtime_combined_step_trace(
        profile,
        channels,
        y_start=combined_y,
        width=width,
    )
    visible_channels = channels[:max_channels]
    hidden_count = max(len(channels) - len(visible_channels), 0)
    body_height = max(len(visible_channels), 1) * row_height
    row_section_y = combined_y + combined_height + 28
    section_height = (
        (row_section_y - y_start)
        + header_height
        + body_height
        + footer_height
    )

    fragments: list[str] = [
        f'<text x="24" y="{y_start + 24}" font-size="16" font-weight="700" '
        'fill="#111827">Executed Action Values</text>'
    ]

    if not steps:
        fragments.append(
            f'<text x="24" y="{y_start + 48}" font-size="12" fill="#6b7280">'
            "No emitted actions were recorded.</text>"
        )
        return "".join(fragments), section_height

    subtitle = (
        f"steps={len(steps)} channels={len(channels)} "
        "source=final action returned by act_fn"
    )
    if hidden_count:
        subtitle += f" showing_first={len(visible_channels)}"
    fragments.append(
        f'<text x="24" y="{y_start + 46}" font-size="12" fill="#4b5563">'
        f"{escape(subtitle)}</text>"
    )
    fragments.append(
        combined_section
    )
    fragments.append(
        f'<text x="24" y="{row_section_y + 24}" font-size="14" font-weight="700" '
        'fill="#111827">Per-Channel Detail</text>'
    )
    fragments.append(
        f'<text x="{margin_left}" y="{row_section_y + 68}" font-size="11" fill="#6b7280">'
        "runtime step</text>"
    )

    def x_of(step_index: int) -> float:
        if len(steps) <= 1:
            return margin_left + chart_width / 2.0
        return margin_left + chart_width * step_index / float(len(steps) - 1)

    tick_count = min(max(len(steps) - 1, 1), 8)
    for tick_index in range(tick_count + 1):
        step_index = round((len(steps) - 1) * tick_index / tick_count)
        x = x_of(step_index)
        fragments.append(
            f'<line x1="{x:.2f}" y1="{row_section_y + header_height - 6}" '
            f'x2="{x:.2f}" y2="{row_section_y + header_height + body_height - 10}" '
            'stroke="#f3f4f6" stroke-width="1" />'
        )
        fragments.append(
            f'<text x="{x:.2f}" y="{row_section_y + header_height + body_height + 12}" '
            'font-size="10" text-anchor="middle" fill="#6b7280">'
            f"{step_index}</text>"
        )

    palette = [
        "#2563eb",
        "#059669",
        "#d97706",
        "#7c3aed",
        "#dc2626",
        "#0891b2",
        "#4d7c0f",
        "#be123c",
    ]
    show_point_values = len(steps) <= 10
    row_top = row_section_y + header_height
    for channel_index, (label, command_name, values) in enumerate(visible_channels):
        y = row_top + channel_index * row_height
        plot_top = y + 7
        plot_height = row_height - 18
        finite_values = [value for value in values if value is not None]
        if not finite_values:
            continue
        low = min(finite_values)
        high = max(finite_values)
        if low == high:
            padding = max(abs(low) * 0.1, 1.0)
            low -= padding
            high += padding
        span = high - low

        def y_of(value: float) -> float:
            return plot_top + plot_height * (1.0 - ((value - low) / span))

        color = palette[channel_index % len(palette)]
        fragments.append(
            f'<rect x="0" y="{y:.2f}" width="{width}" height="{row_height}" '
            f'fill="{"#f9fafb" if channel_index % 2 == 0 else "#ffffff"}" />'
        )
        fragments.append(
            f'<text x="{margin_left - 12}" y="{y + 18:.2f}" font-size="11" '
            'text-anchor="end" fill="#111827">'
            f"{escape(label)}</text>"
        )
        fragments.append(
            f'<text x="{margin_left - 12}" y="{y + 33:.2f}" font-size="9" '
            'text-anchor="end" fill="#6b7280">'
            f"{escape(command_name)}</text>"
        )
        fragments.append(
            f'<line x1="{margin_left}" y1="{y + row_height / 2.0:.2f}" '
            f'x2="{width - margin_right}" y2="{y + row_height / 2.0:.2f}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )

        points = [
            f"{x_of(step_index):.2f},{y_of(value):.2f}"
            for step_index, value in enumerate(values)
            if value is not None
        ]
        fragments.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'points="{" ".join(points)}" />'
        )
        for step_index, value in enumerate(values):
            if value is None:
                continue
            x = x_of(step_index)
            point_y = y_of(value)
            fragments.append(
                f'<circle cx="{x:.2f}" cy="{point_y:.2f}" r="3.3" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1" />'
            )
            if show_point_values:
                fragments.append(
                    f'<text x="{x:.2f}" y="{point_y - 6:.2f}" font-size="9" '
                    'text-anchor="middle" fill="#374151">'
                    f"{_format_profile_value(value)}</text>"
                )

        last_value = finite_values[-1]
        right_summary = (
            f"last={_format_profile_value(last_value)} "
            f"min={_format_profile_value(min(finite_values))} "
            f"max={_format_profile_value(max(finite_values))}"
        )
        fragments.append(
            f'<text x="{width - 18}" y="{y + 22:.2f}" font-size="10" '
            'text-anchor="end" fill="#374151">'
            f"{escape(right_summary)}</text>"
        )

    if hidden_count:
        fragments.append(
            f'<text x="24" y="{y_start + section_height - 10}" '
            'font-size="11" fill="#6b7280">'
            f"{hidden_count} more channels are available in runtime_profile.json.</text>"
        )

    return "".join(fragments), section_height


def _runtime_profile_svg(profile: "RuntimeInferenceProfile") -> str:
    """Render one lightweight SVG timeline for live async request profiling."""

    requests = profile.requests
    width = 1120
    row_height = 28
    header_height = 118
    footer_height = 48
    margin_left = 280
    margin_right = 28
    chart_width = width - margin_left - margin_right
    request_section_height = header_height + footer_height + max(len(requests), 1) * row_height
    action_section_y = request_section_height + 32
    action_section, action_section_height = _runtime_action_trace_section(
        profile,
        y_start=action_section_y,
        width=width,
    )
    height = action_section_y + action_section_height

    duration_candidates = [
        request.usable_latency_s
        for request in requests
        if request.usable_latency_s is not None
    ]
    duration_candidates.extend(
        request.request_duration_s
        for request in requests
        if request.request_duration_s is not None
    )
    max_duration_s = max(duration_candidates, default=1e-6)
    max_duration_s = max(max_duration_s, 1e-6)

    rows: list[str] = []
    for index, request in enumerate(requests):
        y = header_height + index * row_height
        if index % 2 == 0:
            rows.append(
                f'<rect x="0" y="{y - 14}" width="{width}" height="{row_height}" '
                'fill="#f9fafb" />'
            )

        label = (
            f"req {request.request_index} "
            f"(step={request.request_step}, hint={request.latency_hint_raw_steps})"
        )
        rows.append(
            f'<text x="{margin_left - 10}" y="{y + 4}" font-size="12" '
            'text-anchor="end" fill="#111827">'
            f"{escape(label)}</text>"
        )

        bar_y = y - 7
        cursor_x = float(margin_left)
        if request.request_duration_s is not None:
            request_width = max(
                chart_width * request.request_duration_s / max_duration_s,
                1.0,
            )
            rows.append(
                f'<rect x="{cursor_x:.2f}" y="{bar_y}" width="{request_width:.2f}" '
                'height="6" fill="#2563eb" rx="2" />'
            )
            cursor_x += request_width
        if request.prepare_duration_s is not None:
            prepare_width = max(
                chart_width * request.prepare_duration_s / max_duration_s,
                1.0,
            )
            rows.append(
                f'<rect x="{cursor_x:.2f}" y="{bar_y}" width="{prepare_width:.2f}" '
                'height="6" fill="#f59e0b" rx="2" />'
            )
            cursor_x += prepare_width
        if request.accept_delay_s is not None:
            accept_width = max(
                chart_width * request.accept_delay_s / max_duration_s,
                1.0,
            )
            rows.append(
                f'<rect x="{cursor_x:.2f}" y="{bar_y}" width="{accept_width:.2f}" '
                'height="6" fill="#10b981" rx="2" />'
            )

        status = "accepted"
        status_fill = "#059669"
        if request.error is not None:
            status = "error"
            status_fill = "#dc2626"
        elif request.dropped_as_stale:
            status = "stale"
            status_fill = "#d97706"
        elif not request.accepted:
            status = "unused"
            status_fill = "#6b7280"
        rows.append(
            f'<text x="{width - 18}" y="{y + 4}" font-size="11" '
            f'text-anchor="end" fill="{status_fill}">{escape(status)}</text>'
        )

    grid: list[str] = []
    tick_count = 5
    for tick_index in range(tick_count + 1):
        ratio = tick_index / tick_count
        x = margin_left + chart_width * ratio
        tick_duration_ms = max_duration_s * ratio * 1000.0
        grid.append(
            f'<line x1="{x:.2f}" y1="{header_height - 18}" x2="{x:.2f}" '
            f'y2="{request_section_height - footer_height + 6}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{x:.2f}" y="{header_height - 28}" font-size="11" '
            'text-anchor="middle" fill="#6b7280">'
            f"{tick_duration_ms:.1f} ms</text>"
        )

    summary = profile.summary()
    summary_text = (
        f"requests={summary['total_requests']} "
        f"accepted={summary['accepted_requests']} "
        f"stale={summary['dropped_stale_requests']} "
        f"failed={summary['failed_requests']}"
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Live runtime profile">'
        '<rect width="100%" height="100%" fill="#ffffff" />'
        '<text x="24" y="34" font-size="20" font-weight="700" fill="#111827">'
        "InferenceRuntime Live Profile</text>"
        f'<text x="24" y="58" font-size="12" fill="#4b5563">{escape(summary_text)}</text>'
        '<text x="24" y="80" font-size="11" fill="#6b7280">'
        "Blue=request duration  Amber=prepare duration  Green=accept delay</text>"
        '<text x="24" y="100" font-size="11" fill="#6b7280">'
        "Each row is one async request launched by the runtime.</text>"
        + "".join(grid)
        + "".join(rows)
        + f'<line x1="24" y1="{request_section_height + 16}" '
        f'x2="{width - 24}" y2="{request_section_height + 16}" '
        'stroke="#e5e7eb" stroke-width="1" />'
        + action_section
        + "</svg>"
    )

__all__ = [
    "_runtime_action_channels",
    "_runtime_action_trace_section",
    "_runtime_chunk_action_channel_keys",
    "_runtime_combined_step_trace",
    "_runtime_profile_svg",
]
