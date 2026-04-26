"""SVG rendering for async buffer traces."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import AsyncBufferTrace


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
    blended_mask = [step.blended_from_request_index is not None for step in steps]

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
            f"{legend_cursor_x + 10.0:.2f},{legend_row_y - 4} "
            f"{legend_cursor_x + 5.0:.2f},{legend_row_y + 1} "
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
        "buffer</text>" + "</svg>"
    )


__all__ = ["_async_buffer_trace_svg", "_step_plot_points"]
