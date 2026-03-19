import math
from html import escape
from textwrap import dedent
from typing import List, Optional, Tuple


def error_banner_html(title: str, message: str) -> str:
    return (
        "<div class='error-banner'>"
        f"<div class='error-title'>{escape(title)}</div>"
        f"<div class='error-copy'>{escape(message)}</div>"
        "<div class='error-copy'>建议检查依赖是否完整；如有预训练模型，请放入 artifacts/models/xgboost/ 或 models/ 目录；否则系统将继续使用演示模型逻辑。</div>"
        "</div>"
    )


def metric_box(label: str, value: str, copy: str) -> str:
    return (
        "<div class='metric-box'>"
        f"<div class='metric-label'>{escape(label)}</div>"
        f"<div class='metric-value'>{escape(value)}</div>"
        f"<div class='metric-copy'>{escape(copy)}</div>"
        "</div>"
    )


def clinical_metric_card(
    label: str,
    title: str,
    value: str,
    copy: str,
    range_text: str = "",
    wide: bool = False,
    badges: Optional[List[str]] = None,
) -> str:
    badge_html = ""
    if badges:
        badge_html = "<div class='clinical-badges'>" + "".join(
            f"<span class='clinical-badge'>{escape(item)}</span>" for item in badges
        ) + "</div>"
    wide_class = " wide" if wide else ""
    range_html = f"<div class='clinical-range'>{escape(range_text)}</div>" if range_text else ""
    return (
        f"<div class='reserve-clinical-card{wide_class}'>"
        f"<div class='clinical-label'>{escape(label)}</div>"
        f"<div class='clinical-title'>{escape(title)}</div>"
        f"<div class='clinical-value'>{escape(value)}</div>"
        f"{range_html}"
        f"<div class='clinical-copy'>{escape(copy)}</div>"
        f"{badge_html}"
        "</div>"
    )


def ranking_card_html(title: str, subtitle: str, rows: List[Tuple[str, int]], traffic: bool = False) -> str:
    max_value = max(value for _, value in rows)
    fill_class = "rank-fill-traffic" if traffic else ""
    body = []
    for city, value in rows:
        width = 0 if max_value == 0 else value / max_value * 100
        body.append(
            "<div class='rank-row'>"
            f"<div class='rank-head'><div class='rank-city'>{escape(city)}</div><div class='rank-value'>{value:,}</div></div>"
            "<div class='rank-track'>"
            f"<div class='rank-fill {fill_class}' style='width:{width:.1f}%'></div>"
            "</div>"
            "</div>"
        )
    return (
        "<div class='analytics-card'>"
        f"<div class='card-label'>{escape(subtitle)}</div>"
        f"<div class='card-title'>{escape(title)}</div>"
        f"{''.join(body)}"
        "</div>"
    )


def svg_circular_progress_html(
    value: float,
    size: int = 270,
    stroke_width: int = 20,
    caption: str = "Reserve Index",
) -> str:
    clamped = max(0.0, min(100.0, float(value)))
    size = max(int(size), 120)
    stroke_width = max(int(stroke_width), 6)
    radius = max((size - stroke_width) / 2 - 2, 12)
    center = size / 2
    circumference = 2 * math.pi * radius
    dash_offset = circumference * (1 - clamped / 100.0)
    font_size = size * 0.195
    caption_size = size * 0.06
    caption_y = center + size * 0.14
    value_y = center - size * 0.03
    return dedent(
        f"""
        <div class='svg-progress-shell' style='--progress-size:{size}px;'>
            <svg viewBox='0 0 {size} {size}' role='img' aria-label='Circular progress {clamped:.1f}%'>
                <circle class='svg-progress-track' cx='{center:.2f}' cy='{center:.2f}' r='{radius:.2f}' fill='none' stroke-width='{stroke_width}'></circle>
                <circle class='svg-progress-bar' cx='{center:.2f}' cy='{center:.2f}' r='{radius:.2f}' fill='none' stroke-width='{stroke_width}' stroke-linecap='round' stroke-dasharray='{circumference:.3f}' stroke-dashoffset='{dash_offset:.3f}' transform='rotate(-90 {center:.2f} {center:.2f})'></circle>
                <text class='svg-progress-value' x='50%' y='{value_y:.2f}' text-anchor='middle' dominant-baseline='middle' style='font-size:{font_size:.1f}px;'>{clamped:.1f}%</text>
                <text class='svg-progress-caption' x='50%' y='{caption_y:.2f}' text-anchor='middle' dominant-baseline='middle' style='font-size:{caption_size:.1f}px;'>{escape(caption.upper())}</text>
            </svg>
        </div>
        """
    ).strip()
