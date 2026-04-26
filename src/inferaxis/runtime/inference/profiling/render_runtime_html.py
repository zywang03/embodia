"""HTML rendering for live runtime inference profiles."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from .render_runtime_svg import (
    _runtime_action_channels,
    _runtime_chunk_action_channel_keys,
)

if TYPE_CHECKING:
    from .models import RuntimeInferenceProfile


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
    chunk_channel_keys = _runtime_chunk_action_channel_keys(profile)
    summary = profile.summary()

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.34, 0.66],
        vertical_spacing=0.14,
        subplot_titles=(
            "Request Timing",
            "Step Trace: buffer size + chunk actions",
        ),
    )

    request_labels = [f"req {request.request_index}" for request in requests]
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
                y=[_seconds_to_ms(getattr(request, attr_name)) for request in requests],
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
        statuses = [_runtime_request_status(request)[0] for request in requests]
        status_colors = [_runtime_request_status(request)[1] for request in requests]
        fig.add_trace(
            go.Scatter(
                name="request status",
                x=request_labels,
                y=[total * 1.04 if total > 0.0 else 0.02 for total in request_totals],
                mode="text",
                text=statuses,
                textfont={"color": status_colors, "size": 11},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    step_x = [step.step_index for step in action_steps]
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
                hovertemplate=("step=%{x}<br>buffer_size=%{y}<extra></extra>"),
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
    status_dash = {
        "accepted": "solid",
        "dropped": "dash",
        "unused": "dot",
    }
    status_rank = {
        "dropped": 0,
        "accepted": 1,
        "unused": 2,
    }
    if profile.chunk_actions and chunk_channel_keys:
        request_indices = sorted(
            {action.request_index for action in profile.chunk_actions}
        )
        color_by_request = {
            request_index: palette[index % len(palette)]
            for index, request_index in enumerate(request_indices)
        }
        shown_legend_items: set[tuple[int, str]] = set()
        for request_index in request_indices:
            request_actions = [
                action
                for action in profile.chunk_actions
                if action.request_index == request_index
            ]
            statuses = sorted(
                {action.status for action in request_actions},
                key=lambda status: status_rank.get(status, 99),
            )
            for status in statuses:
                status_actions = [
                    action for action in request_actions if action.status == status
                ]
                for channel_index, (
                    target,
                    command_name,
                    dim_index,
                ) in enumerate(chunk_channel_keys):
                    x_values: list[int] = []
                    y_values: list[float] = []
                    customdata: list[list[object]] = []
                    for action in status_actions:
                        value: float | None = None
                        for command in action.commands:
                            if (
                                command.target == target
                                and command.command == command_name
                                and dim_index < len(command.value)
                            ):
                                value = command.value[dim_index]
                                break
                        if value is None:
                            continue
                        x_values.append(action.step_index)
                        y_values.append(value)
                        customdata.append(
                            [
                                request_index,
                                action.action_index,
                                status,
                                f"{target}[{dim_index}]",
                                command_name,
                            ]
                        )
                    if not x_values:
                        continue
                    legend_key = (request_index, status)
                    showlegend = legend_key not in shown_legend_items
                    shown_legend_items.add(legend_key)
                    fig.add_trace(
                        go.Scatter(
                            name=f"chunk {request_index} {status}",
                            x=x_values,
                            y=y_values,
                            mode="lines+markers",
                            visible=True if channel_index < 16 else "legendonly",
                            legendgroup=f"chunk-{request_index}",
                            showlegend=showlegend,
                            line={
                                "color": color_by_request[request_index],
                                "width": 2.4,
                                "dash": status_dash.get(status, "dash"),
                            },
                            marker={"size": 6},
                            customdata=customdata,
                            hovertemplate=(
                                "step=%{x}<br>"
                                "%{customdata[3]}=%{y:.6g}<br>"
                                "chunk=%{customdata[0]} "
                                "chunk_action=%{customdata[1]}<br>"
                                "status=%{customdata[2]}<br>"
                                "command=%{customdata[4]}"
                                "<extra></extra>"
                            ),
                        ),
                        row=2,
                        col=1,
                    )
    else:
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
                    customdata=[command_name for _ in values],
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
            "groupclick": "togglegroup",
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


__all__ = [
    "_runtime_profile_html",
    "_runtime_request_status",
    "_seconds_to_ms",
]
