"""SVG rendering helpers for profiling traces."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import AsyncBufferTrace, RuntimeInferenceProfile


def _step_plot_points(
    values: list[int],
    *,
    x_count: int,
    y_max: int,
    margin_left: int,
    margin_top: int,
    plot_width: int,
    plot_height: int,
) -> str:
    """Build one simple step-plot polyline point string."""

    if not values:
        return ""

    def x_of(step: int) -> float:
        if x_count <= 0:
            return float(margin_left)
        return margin_left + (plot_width * step / x_count)

    def y_of(value: int) -> float:
        return margin_top + plot_height * (1.0 - (value / y_max))

    points: list[str] = []
    first_y = y_of(values[0])
    points.append(f"{x_of(0):.2f},{first_y:.2f}")
    for index, value in enumerate(values):
        left_x = x_of(index)
        right_x = x_of(index + 1)
        y = y_of(value)
        points.append(f"{left_x:.2f},{y:.2f}")
        points.append(f"{right_x:.2f},{y:.2f}")
    return " ".join(points)


def _async_buffer_trace_svg(trace: "AsyncBufferTrace") -> str:
    """Render one lightweight SVG visualization for the async buffer trace."""

    steps = trace.steps
    width = 960
    height = 588
    margin_left = 64
    margin_right = 24
    header_height = 102
    margin_top = header_height
    margin_bottom = 140
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_count = max(len(steps), 1)

    max_buffer = max(
        [1]
        + [step.buffer_after_accept for step in steps]
        + [step.buffer_after_execute for step in steps]
        + [request.chunk_steps for request in trace.requests]
    )
    y_max = max(max_buffer, 1)

    buffer_points = _step_plot_points(
        [step.buffer_after_accept for step in steps],
        x_count=x_count,
        y_max=y_max,
        margin_left=margin_left,
        margin_top=margin_top,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    threshold_points = _step_plot_points(
        [step.steps_before_request for step in steps],
        x_count=x_count,
        y_max=y_max,
        margin_left=margin_left,
        margin_top=margin_top,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    blended_mask = [
        step.blended_from_request_index is not None
        for step in steps
    ]

    def x_of(step: int) -> float:
        return margin_left + (plot_width * step / x_count)

    def y_of(value: int) -> float:
        return margin_top + plot_height * (1.0 - (value / y_max))

    def request_color(request_index: int) -> str:
        palette = [
            "#2563eb",
            "#10b981",
            "#f59e0b",
            "#ef4444",
            "#8b5cf6",
            "#14b8a6",
            "#f97316",
            "#84cc16",
        ]
        return palette[request_index % len(palette)]

    blended_segments: list[str] = []
    for index, step in enumerate(steps):
        if not blended_mask[index]:
            continue
        left_x = x_of(index)
        right_x = x_of(index + 1)
        y = y_of(step.buffer_after_accept)
        blended_segments.append(
            f'<line x1="{left_x:.2f}" y1="{y:.2f}" '
            f'x2="{right_x:.2f}" y2="{y:.2f}" '
            'stroke="#f97316" stroke-width="3.5" stroke-linecap="round" />'
        )
        if index > 0 and blended_mask[index - 1]:
            previous_y = y_of(steps[index - 1].buffer_after_accept)
            blended_segments.append(
                f'<line x1="{left_x:.2f}" y1="{previous_y:.2f}" '
                f'x2="{left_x:.2f}" y2="{y:.2f}" '
                'stroke="#f97316" stroke-width="3.5" stroke-linecap="round" />'
            )

    grid_lines: list[str] = []
    for y_tick in range(y_max + 1):
        y = y_of(y_tick)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" '
            f'y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" '
            'font-size="11" text-anchor="end" fill="#6b7280">'
            f"{y_tick}</text>"
        )

    x_axis_label_y = height - margin_bottom + 20
    step_label_y = height - margin_bottom + 40
    lane_label_y = height - margin_bottom + 62
    lane_top = height - margin_bottom + 74
    lane_height = 16

    x_ticks: list[str] = []
    x_tick_count = min(max(len(steps), 2), 8)
    for tick_index in range(x_tick_count + 1):
        step = round(len(steps) * tick_index / x_tick_count)
        x = x_of(step)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
            f'y2="{height - margin_bottom}" stroke="#f3f4f6" stroke-width="1" />'
        )
        x_ticks.append(
            f'<text x="{x:.2f}" y="{x_axis_label_y}" '
            'font-size="11" text-anchor="middle" fill="#6b7280">'
            f"{step}</text>"
        )

    markers: list[str] = []
    for step in steps:
        x = x_of(step.step_index)
        y = y_of(step.buffer_after_accept)
        if step.request_started:
            markers.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" '
                'fill="#10b981" stroke="#065f46" stroke-width="1.2" />'
            )
        if step.request_completed:
            size = 5.0
            markers.append(
                "<polygon "
                f'points="{x:.2f},{y - size:.2f} {x + size:.2f},{y:.2f} '
                f'{x:.2f},{y + size:.2f} {x - size:.2f},{y:.2f}" '
                'fill="#f59e0b" stroke="#92400e" stroke-width="1.2" />'
            )
        if step.underrun:
            y0 = y_of(0)
            size = 5.0
            markers.append(
                f'<line x1="{x - size:.2f}" y1="{y0 - size:.2f}" '
                f'x2="{x + size:.2f}" y2="{y0 + size:.2f}" '
                'stroke="#dc2626" stroke-width="2" />'
            )
            markers.append(
                f'<line x1="{x - size:.2f}" y1="{y0 + size:.2f}" '
                f'x2="{x + size:.2f}" y2="{y0 - size:.2f}" '
                'stroke="#dc2626" stroke-width="2" />'
            )

    request_summary = (
        f"requests={len(trace.requests)} steps={len(trace.steps)} "
        f"target_hz={trace.target_hz:.2f} steps_before_request="
        f"{trace.steps_before_request} "
        f"latency_beta={trace.latency_ema_beta:.2f}"
    )

    legend_row_y = 66
    legend_cursor_x = float(margin_left)
    legend_items: list[str] = []

    def add_legend_item(
        label: str,
        marker_svg: str,
        *,
        marker_width: float,
    ) -> None:
        nonlocal legend_cursor_x
        label_x = legend_cursor_x + marker_width + 8.0
        legend_items.append(marker_svg)
        legend_items.append(
            f'<text x="{label_x:.2f}" y="{legend_row_y}" font-size="11" fill="#374151">'
            f"{label}</text>"
        )
        legend_cursor_x = label_x + max(len(label) * 5.8, 24.0) + 28.0

    add_legend_item(
        "buffer_after_accept",
        (
            f'<line x1="{legend_cursor_x:.2f}" y1="{legend_row_y - 4}" '
            f'x2="{legend_cursor_x + 36.0:.2f}" y2="{legend_row_y - 4}" '
            'stroke="#2563eb" stroke-width="2.5" />'
        ),
        marker_width=36.0,
    )
    add_legend_item(
        "steps_before_request",
        (
            f'<line x1="{legend_cursor_x:.2f}" y1="{legend_row_y - 4}" '
            f'x2="{legend_cursor_x + 36.0:.2f}" y2="{legend_row_y - 4}" '
            'stroke="#6b7280" stroke-width="2" stroke-dasharray="6 4" />'
        ),
        marker_width=36.0,
    )
    add_legend_item(
        "request_started",
        (
            f'<circle cx="{legend_cursor_x + 5.0:.2f}" cy="{legend_row_y - 4}" r="4.5" '
            'fill="#10b981" stroke="#065f46" stroke-width="1.2" />'
        ),
        marker_width=10.0,
    )
    add_legend_item(
        "request_completed",
        (
            f'<polygon points="{legend_cursor_x + 5.0:.2f},{legend_row_y - 9} '
            f'{legend_cursor_x + 10.0:.2f},{legend_row_y - 4} '
            f'{legend_cursor_x + 5.0:.2f},{legend_row_y + 1} '
            f'{legend_cursor_x:.2f},{legend_row_y - 4}" fill="#f59e0b" '
            'stroke="#92400e" stroke-width="1.2" />'
        ),
        marker_width=12.0,
    )
    add_legend_item(
        "underrun",
        (
            f'<line x1="{legend_cursor_x:.2f}" y1="{legend_row_y - 9}" '
            f'x2="{legend_cursor_x + 10.0:.2f}" y2="{legend_row_y + 1}" '
            'stroke="#dc2626" stroke-width="2" />'
            f'<line x1="{legend_cursor_x:.2f}" y1="{legend_row_y + 1}" '
            f'x2="{legend_cursor_x + 10.0:.2f}" y2="{legend_row_y - 9}" '
            'stroke="#dc2626" stroke-width="2" />'
        ),
        marker_width=12.0,
    )
    add_legend_item(
        "ensemble",
        (
            f'<line x1="{legend_cursor_x:.2f}" y1="{legend_row_y - 4}" '
            f'x2="{legend_cursor_x + 36.0:.2f}" y2="{legend_row_y - 4}" '
            'stroke="#f97316" stroke-width="3.5" stroke-linecap="round" />'
        ),
        marker_width=36.0,
    )

    lane_rects: list[str] = []
    lane_labels: list[str] = []
    segment_start = 0
    while segment_start < len(steps):
        segment_step = steps[segment_start]
        segment_key = (
            segment_step.executed_request_index,
            segment_step.blended_from_request_index,
        )
        segment_end = segment_start + 1
        while segment_end < len(steps):
            other = steps[segment_end]
            other_key = (
                other.executed_request_index,
                other.blended_from_request_index,
            )
            if other_key != segment_key:
                break
            segment_end += 1

        x0 = x_of(segment_start)
        x1 = x_of(segment_end)
        label = ""
        if segment_key[0] is not None:
            if segment_key[1] is None:
                label = f"r{segment_key[0]}"
            else:
                label = f"r{segment_key[1]}+r{segment_key[0]}"
        if segment_key[0] is None:
            lane_rects.append(
                f'<rect x="{x0:.2f}" y="{lane_top:.2f}" width="{max(x1 - x0, 1.0):.2f}" '
                f'height="{lane_height}" fill="#ffffff" stroke="#d1d5db" stroke-width="0.8" />'
            )
        elif segment_key[1] is None:
            lane_rects.append(
                f'<rect x="{x0:.2f}" y="{lane_top:.2f}" width="{max(x1 - x0, 1.0):.2f}" '
                f'height="{lane_height}" fill="{request_color(segment_key[0])}" '
                'stroke="#ffffff" stroke-width="0.4" />'
            )
        else:
            split_x = x0 + max((x1 - x0) / 2.0, 0.5)
            lane_rects.append(
                f'<rect x="{x0:.2f}" y="{lane_top:.2f}" width="{max(split_x - x0, 0.5):.2f}" '
                f'height="{lane_height}" fill="{request_color(segment_key[1])}" '
                'stroke="#ffffff" stroke-width="0.2" />'
            )
            lane_rects.append(
                f'<rect x="{split_x:.2f}" y="{lane_top:.2f}" width="{max(x1 - split_x, 0.5):.2f}" '
                f'height="{lane_height}" fill="{request_color(segment_key[0])}" '
                'stroke="#ffffff" stroke-width="0.2" />'
            )
            lane_rects.append(
                f'<rect x="{x0:.2f}" y="{lane_top:.2f}" width="{max(x1 - x0, 1.0):.2f}" '
                f'height="{lane_height}" fill="none" stroke="#111827" stroke-width="0.35" />'
            )

        if label and x1 - x0 >= 26.0:
            lane_labels.append(
                f'<text x="{(x0 + x1) / 2.0:.2f}" y="{lane_top + 11:.2f}" '
                'font-size="9" text-anchor="middle" fill="#ffffff">'
                f"{escape(label)}</text>"
            )

        segment_start = segment_end

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Async buffer trace">'
        '<rect width="100%" height="100%" fill="#ffffff" />'
        f'<text x="{margin_left}" y="26" font-size="18" font-weight="700" '
        'fill="#111827">Async Buffer Trace</text>'
        f'<text x="{margin_left}" y="44" font-size="12" fill="#4b5563">'
        f"{escape(request_summary)}</text>"
        + "".join(legend_items)
        + (
            f'<line x1="{margin_left}" y1="{header_height - 8}" '
            f'x2="{width - margin_right}" y2="{header_height - 8}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        + "".join(grid_lines)
        + "".join(x_ticks)
        + (
            f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" '
            f'points="{buffer_points}" />'
            if buffer_points
            else ""
        )
        + "".join(blended_segments)
        + (
            f'<polyline fill="none" stroke="#6b7280" stroke-width="2" '
            'stroke-dasharray="6 4" '
            f'points="{threshold_points}" />'
            if threshold_points
            else ""
        )
        + "".join(markers)
        + f'<text x="{margin_left}" y="{lane_label_y:.2f}" font-size="11" '
        'font-weight="600" fill="#111827">chunk lane</text>'
        + f'<rect x="{margin_left + 82}" y="{lane_label_y - 10:.2f}" width="16" height="12" '
        'fill="#2563eb" rx="2" ry="2" />'
        + f'<text x="{margin_left + 108}" y="{lane_label_y:.2f}" font-size="11" fill="#374151">'
        "color = emitted action origin</text>"
        + f'<rect x="{margin_left + 318}" y="{lane_label_y - 10:.2f}" width="8" height="12" '
        'fill="#10b981" rx="1.5" ry="1.5" />'
        + f'<rect x="{margin_left + 326}" y="{lane_label_y - 10:.2f}" width="8" height="12" '
        'fill="#f59e0b" rx="1.5" ry="1.5" />'
        + f'<text x="{margin_left + 346}" y="{lane_label_y:.2f}" font-size="11" fill="#374151">'
        "split = overlap-blended handoff</text>"
        + "".join(lane_rects)
        + "".join(lane_labels)
        + f'<text x="{margin_left + (plot_width / 2.0):.2f}" y="{step_label_y}" '
        'font-size="12" text-anchor="middle" fill="#111827">'
        "step</text>"
        + f'<text x="18" y="{margin_top - 12}" font-size="12" fill="#111827">'
        "buffer</text>"
        + "</svg>"
    )


def _format_profile_value(value: float) -> str:
    """Format one numeric profile value compactly for SVG labels."""

    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.4g}"


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


def _runtime_request_status(request: object) -> tuple[str, str]:
    """Return a compact status label and color for one profiled request."""

    error = getattr(request, "error", None)
    if error is not None:
        return "error", "#dc2626"
    if getattr(request, "dropped_as_stale", False):
        return "stale", "#d97706"
    if not getattr(request, "accepted", False):
        return "unused", "#6b7280"
    return "accepted", "#059669"


def _seconds_to_ms(value: float | None) -> float:
    """Convert an optional duration in seconds into milliseconds."""

    if value is None:
        return 0.0
    return max(float(value) * 1000.0, 0.0)


def _runtime_profile_html(profile: "RuntimeInferenceProfile") -> str:
    """Render one interactive Plotly report for live async runtime profiling."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    requests = profile.requests
    action_steps = profile.action_steps
    channels = _runtime_action_channels(profile)
    summary = profile.summary()

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.34, 0.66],
        vertical_spacing=0.14,
        subplot_titles=(
            "Request Timing",
            "Step Trace: buffer size + action values",
        ),
    )

    request_labels = [
        f"req {request.request_index}"
        for request in requests
    ]
    request_customdata = [
        [
            request.request_step,
            request.latency_hint_raw_steps,
            _runtime_request_status(request)[0],
            request.returned_chunk_length,
            request.accepted_chunk_length,
        ]
        for request in requests
    ]
    duration_specs = [
        ("request", "request_duration_s", "#2563eb"),
        ("prepare", "prepare_duration_s", "#f59e0b"),
        ("accept wait", "accept_delay_s", "#10b981"),
    ]
    for label, attr_name, color in duration_specs:
        fig.add_trace(
            go.Bar(
                name=f"{label} ms",
                x=request_labels,
                y=[
                    _seconds_to_ms(getattr(request, attr_name))
                    for request in requests
                ],
                marker_color=color,
                customdata=request_customdata,
                hovertemplate=(
                    "%{x}<br>"
                    f"{label}: " + "%{y:.3f} ms<br>"
                    "request_step=%{customdata[0]}<br>"
                    "latency_hint=%{customdata[1]}<br>"
                    "status=%{customdata[2]}<br>"
                    "returned=%{customdata[3]} accepted=%{customdata[4]}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    request_totals = [
        _seconds_to_ms(request.request_duration_s)
        + _seconds_to_ms(request.prepare_duration_s)
        + _seconds_to_ms(request.accept_delay_s)
        for request in requests
    ]
    if requests:
        statuses = [
            _runtime_request_status(request)[0]
            for request in requests
        ]
        status_colors = [
            _runtime_request_status(request)[1]
            for request in requests
        ]
        fig.add_trace(
            go.Scatter(
                name="request status",
                x=request_labels,
                y=[
                    total * 1.04 if total > 0.0 else 0.02
                    for total in request_totals
                ],
                mode="text",
                text=statuses,
                textfont={"color": status_colors, "size": 11},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    step_x = [
        step.step_index
        for step in action_steps
    ]
    buffer_y = [
        None if step.buffer_size is None else float(step.buffer_size)
        for step in action_steps
    ]
    if action_steps:
        fig.add_trace(
            go.Scatter(
                name="buffer_size",
                x=step_x,
                y=buffer_y,
                mode="lines+markers",
                line={"color": "#111827", "width": 3, "dash": "dash"},
                marker={"size": 8},
                hovertemplate=(
                    "step=%{x}<br>buffer_size=%{y}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
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
        "#0f766e",
        "#b45309",
        "#4338ca",
        "#be185d",
    ]
    for index, (label, command_name, values) in enumerate(channels):
        fig.add_trace(
            go.Scatter(
                name=label,
                x=step_x,
                y=values,
                mode="lines+markers",
                visible=True if index < 16 else "legendonly",
                line={
                    "color": palette[index % len(palette)],
                    "width": 2,
                },
                marker={"size": 6},
                customdata=[
                    command_name
                    for _ in values
                ],
                hovertemplate=(
                    "step=%{x}<br>"
                    f"{label}=%" + "{y:.6g}<br>"
                    "command=%{customdata}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    if not requests:
        fig.add_annotation(
            text="No async requests were recorded.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.82,
            showarrow=False,
            font={"color": "#6b7280"},
        )
    if not action_steps:
        fig.add_annotation(
            text="No emitted actions were recorded.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.26,
            showarrow=False,
            font={"color": "#6b7280"},
        )

    title = (
        "InferenceRuntime Live Profile"
        f" | requests={summary['total_requests']}"
        f" accepted={summary['accepted_requests']}"
        f" stale={summary['dropped_stale_requests']}"
        f" failed={summary['failed_requests']}"
    )
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        barmode="stack",
        hovermode="x unified",
        height=860,
        margin={"l": 70, "r": 32, "t": 90, "b": 72},
        legend={
            "orientation": "v",
            "x": 1.01,
            "y": 1.0,
            "xanchor": "left",
            "yanchor": "top",
            "itemsizing": "constant",
        },
    )
    fig.update_xaxes(title_text="request", row=1, col=1)
    fig.update_yaxes(title_text="milliseconds", row=1, col=1)
    fig.update_xaxes(
        title_text="runtime step",
        row=2,
        col=1,
        rangeslider={"visible": len(action_steps) > 30},
    )
    fig.update_yaxes(title_text="value", row=2, col=1)

    return fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )
