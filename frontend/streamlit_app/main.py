import base64
import hashlib
import json
import math
import os
import re
import sys
from html import escape
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[2]

# 确保能导入项目包
sys.path.insert(0, str(REPO_ROOT))

from ovarian_prediction.config import BACKGROUND_IMAGES_DIR, BRAND_IMAGES_DIR, LEGACY_MODEL_DIR, MODEL_ARTIFACT_DIR

from frontend.streamlit_app.components import layout as layout_components
from frontend.streamlit_app.components import metric_cards as metric_cards_component
from frontend.streamlit_app.components import result_panels as result_panel_components
from frontend.streamlit_app.services import system_loader, upload_parser as upload_parser_service
from frontend.streamlit_app.state import session_state as session_state_utils
from frontend.streamlit_app.utils import media as media_utils


APP_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = MODEL_ARTIFACT_DIR
LOCAL_MODEL_PATH = LOCAL_MODEL_DIR / "PORDM.pkl"
LEGACY_MODEL_PATH = LEGACY_MODEL_DIR / "PORDM.pkl"
TITLE_LOGO_PATH = BRAND_IMAGES_DIR / "fuyou_logo_fullseal.png"
RAW_TITLE_LOGO_PATH = BRAND_IMAGES_DIR / "fuyou_logo_raw.png"

POR_THRESHOLDS = {"low": 0.20, "high": 0.40}
HOR_THRESHOLDS = {"low": 0.20, "high": 0.35}
OVAREPRED_API_URL = "http://121.43.113.123:8888/api/portal/ovcalculate"

EXAMPLE_DEFAULTS = {
    "Age": 32,
    "AMH": 2.50,
    "FSH": 8.0,
    "AFC": 10,
    "cycle_length": 28,
    "menarche_age": 13,
    "BMI": 21.3,
    "LH": 6.0,
}

FORM_INPUT_KEYS = {
    "reserve_amh": "reserve_amh_input",
    "reserve_age": "reserve_age_input",
    "reserve_fsh": "reserve_fsh_input",
    "reserve_afc": "reserve_afc_input",
    "reserve_cycle": "reserve_cycle_input",
    "reserve_menarche": "reserve_menarche_input",
    "plan_age": "plan_age_input",
    "plan_cycle": "plan_cycle_input",
    "plan_bmi": "plan_bmi_input",
    "plan_afc": "plan_afc_input",
    "plan_fsh": "plan_fsh_input",
    "plan_lh": "plan_lh_input",
    "plan_amh": "plan_amh_input",
}


st.set_page_config(
    page_title="卵巢储备功能与促排卵方案AI辅助系统",
    page_icon="🏥",
    layout="wide",
)


def encode_file_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def encode_bytes_to_data_uri(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def blur_image_to_data_uri(path: Path) -> str:
    try:
        from PIL import Image, ImageFilter
    except Exception:
        return encode_file_to_data_uri(path)

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((1680, 1080))
        image = image.filter(ImageFilter.GaussianBlur(radius=12))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    return encode_bytes_to_data_uri(buffer.getvalue(), "image/jpeg")


def build_fallback_background() -> str:
    svg = """
    <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 980'>
      <defs>
        <linearGradient id='sky' x1='0' y1='0' x2='0' y2='1'>
          <stop offset='0%' stop-color='#6a8fc8'/>
          <stop offset='50%' stop-color='#a7bfdc'/>
          <stop offset='100%' stop-color='#e5edf6'/>
        </linearGradient>
        <linearGradient id='glass' x1='0' y1='0' x2='1' y2='1'>
          <stop offset='0%' stop-color='#55697f'/>
          <stop offset='100%' stop-color='#cad8e8'/>
        </linearGradient>
        <linearGradient id='wall' x1='0' y1='0' x2='1' y2='0'>
          <stop offset='0%' stop-color='#d9e0e8'/>
          <stop offset='100%' stop-color='#fbfdff'/>
        </linearGradient>
      </defs>
      <rect width='1600' height='980' fill='url(#sky)'/>
      <g opacity='0.72' fill='#ffffff'>
        <ellipse cx='220' cy='110' rx='220' ry='52'/>
        <ellipse cx='650' cy='98' rx='260' ry='60'/>
        <ellipse cx='1140' cy='155' rx='260' ry='58'/>
      </g>
      <polygon points='20,330 1090,128 1548,250 1575,565 0,642' fill='url(#wall)' opacity='0.94'/>
      <polygon points='0,500 1600,320 1600,520 15,715' fill='url(#glass)' opacity='0.86'/>
      <polygon points='720,180 1315,96 1445,190 818,270' fill='#eef4fa' opacity='0.92'/>
      <polygon points='1110,305 1600,252 1600,688 1035,736' fill='#f3f7fb' opacity='0.94'/>
      <g opacity='0.44' fill='#607895'>
        <rect x='75' y='560' width='70' height='325' rx='8'/>
        <rect x='165' y='538' width='66' height='345' rx='8'/>
        <rect x='262' y='515' width='64' height='368' rx='8'/>
        <rect x='370' y='496' width='64' height='388' rx='8'/>
        <rect x='482' y='480' width='60' height='404' rx='8'/>
        <rect x='598' y='462' width='60' height='422' rx='8'/>
        <rect x='718' y='444' width='60' height='440' rx='8'/>
        <rect x='838' y='430' width='60' height='454' rx='8'/>
        <rect x='965' y='410' width='62' height='474' rx='8'/>
        <rect x='1096' y='392' width='70' height='492' rx='8'/>
      </g>
    </svg>
    """
    return f"data:image/svg+xml;utf8,{quote(svg)}"


def get_background_uri() -> str:
    candidates = [
        APP_DIR / "fuyou.png",
        APP_DIR / "background.jpg",
        APP_DIR / "background.png",
        APP_DIR / "assets" / "background.jpg",
        APP_DIR / "assets" / "background.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return blur_image_to_data_uri(candidate)
    return build_fallback_background()


def build_title_logo_path() -> Optional[Path]:
    if TITLE_LOGO_PATH.exists():
        return TITLE_LOGO_PATH
    if not RAW_TITLE_LOGO_PATH.exists():
        return None
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return RAW_TITLE_LOGO_PATH

    img = Image.open(RAW_TITLE_LOGO_PATH).convert("RGBA")
    center_x = int(img.size[0] * 0.502)
    center_y = int(img.size[1] * 0.492)
    radius = int(min(img.size) * 0.49)
    left = max(0, center_x - radius)
    upper = max(0, center_y - radius)
    right = min(img.size[0], center_x + radius)
    lower = min(img.size[1], center_y + radius)

    cropped = img.crop((left, upper, right, lower))
    mask = Image.new("L", cropped.size, 0)
    draw = ImageDraw.Draw(mask)
    pad = 1
    draw.ellipse((pad, pad, cropped.size[0] - pad - 1, cropped.size[1] - pad - 1), fill=255)

    result = Image.new("RGBA", cropped.size, (0, 0, 0, 0))
    result.paste(cropped, (0, 0), mask)
    result.save(TITLE_LOGO_PATH)
    return TITLE_LOGO_PATH


BACKGROUND_URI = media_utils.get_background_uri()
TITLE_LOGO_FILE = media_utils.build_title_logo_path()
TITLE_LOGO_URI = media_utils.encode_file_to_data_uri(TITLE_LOGO_FILE) if TITLE_LOGO_FILE and TITLE_LOGO_FILE.exists() else ""


CSS = """
<style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        min-height: 100%;
    }

    body {
        overflow: auto;
    }

    .stApp {
        color: #10253d;
        font-family: "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
        background-image:
            linear-gradient(135deg, rgba(247, 250, 253, 0.28), rgba(237, 243, 248, 0.24)),
            linear-gradient(180deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.08)),
            url("__BACKGROUND__");
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
    }

    .block-container {
        max-width: 1680px;
        padding: 1rem 1.2rem 1.25rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    button[title="View fullscreen"] {
        display: none;
    }

    .title-frame {
        margin: 0 0 1rem;
        width: 100%;
        padding: 0.82rem 1.3rem 0.76rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.72);
        background: linear-gradient(180deg, rgba(252, 254, 255, 0.80), rgba(244, 248, 252, 0.72));
        box-shadow: 0 18px 42px rgba(16, 45, 76, 0.12);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    .title-row {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 1rem;
    }

    .title-logo-wrap {
        width: 74px;
        height: 74px;
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .title-logo {
        width: 100%;
        height: auto;
        display: block;
        filter: drop-shadow(0 10px 22px rgba(56, 98, 68, 0.14));
    }

    .title-copy {
        min-width: 0;
    }

    .page-title {
        margin: 0.04rem 0 0.14rem;
        text-align: left;
        font-family: "Georgia", "STSong", serif;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        color: #0d2746;
    }

    .page-subtitle {
        margin: 0;
        text-align: left;
        font-size: 0.92rem;
        letter-spacing: 0.08em;
        color: rgba(16, 37, 61, 0.68);
    }

    .panel-label {
        margin: 0 0 0.34rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: rgba(16, 37, 61, 0.58);
    }

    .panel-title {
        margin: 0 0 0.72rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: #10253d;
    }

    .panel-note {
        margin: 0.52rem 0 0.1rem;
        font-size: 0.8rem;
        line-height: 1.6;
        color: rgba(16, 37, 61, 0.68);
    }

    .form-head {
        margin-bottom: 0.9rem;
    }

    .form-align-gap {
        height: 4.85rem;
    }

    .upload-toolbar {
        margin: 0 0 0.9rem;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        width: 100%;
    }

    .upload-toolbar-note {
        margin: 0 0 0.55rem;
        text-align: right;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(16, 37, 61, 0.56);
    }

    .upload-popover-copy {
        margin: 0.1rem 0 0.7rem;
        font-size: 0.83rem;
        line-height: 1.65;
        color: rgba(255, 255, 255, 0.86);
    }

    .upload-template-link {
        display: inline-block;
        margin: 0 0 0.85rem;
        font-size: 0.84rem;
        font-weight: 700;
        color: #1a5f8f;
        text-decoration: underline;
        text-underline-offset: 0.18rem;
    }

    div[data-testid="stPopover"] {
        width: min(100%, 264px);
        margin-left: auto;
    }

    div[data-testid="stPopover"] button {
        min-height: 3.05rem;
        width: 100%;
        border-radius: 16px;
        border: 1px solid rgba(16, 37, 61, 0.10) !important;
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #111111 !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 24px rgba(16, 45, 76, 0.08) !important;
    }

    div[data-testid="stPopover"] button:hover,
    div[data-testid="stPopover"] button:focus,
    div[data-testid="stPopover"] button:focus-visible,
    div[data-testid="stPopover"] button:active,
    div[data-testid="stPopover"] button[aria-expanded="true"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1px solid rgba(16, 37, 61, 0.12) !important;
        box-shadow: 0 10px 24px rgba(16, 45, 76, 0.08) !important;
        outline: none !important;
    }

    div[data-testid="stPopover"] button * {
        color: #111111 !important;
    }

    div[data-testid="stPopover"] button p {
        margin: 0;
    }

    div[data-testid="stPopover"] button svg {
        color: #111111 !important;
        fill: #111111 !important;
        stroke: #111111 !important;
    }

    div[data-testid="stPopover"] button::before {
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 0.55rem;
        vertical-align: -2px;
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23111111' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M12 16V6'/><path d='m8.5 9.5 3.5-3.5 3.5 3.5'/><path d='M6 18.5h12'/><rect x='3.5' y='14.5' width='17' height='6' rx='2.5'/></svg>");
    }

    .title-upload-status {
        margin-top: 0.55rem;
        text-align: right;
        font-size: 0.8rem;
        line-height: 1.55;
        color: rgba(16, 37, 61, 0.68);
    }

    .section-gap {
        height: 0.6rem;
    }

    .page-footer {
        margin: 1rem 0 0.2rem;
        padding: 0.85rem 1rem 0.2rem;
        text-align: center;
        font-size: 0.83rem;
        letter-spacing: 0.08em;
        color: rgba(16, 37, 61, 0.64);
    }

    div[data-testid="stForm"] {
        min-height: 560px;
        padding: 1rem 1rem 0.9rem !important;
        border-radius: 28px !important;
        background: linear-gradient(180deg, rgba(252, 254, 255, 0.82), rgba(245, 248, 252, 0.74)) !important;
        border: 1px solid rgba(255, 255, 255, 0.70) !important;
        box-shadow: 0 24px 64px rgba(24, 52, 87, 0.14) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 0.75rem !important;
    }

    label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: rgba(16, 37, 61, 0.86) !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        min-height: 3.05rem !important;
        border-radius: 16px !important;
        background: #ffffff !important;
        border: 1px solid rgba(16, 37, 61, 0.10) !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] * {
        color: #10253d !important;
    }

    div[data-testid="stTextInput"] [data-baseweb="input"] {
        background: #ffffff !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }

    div[data-testid="stTextInput"] [data-baseweb="input"] > div,
    div[data-testid="stTextInput"] [data-baseweb="input"] > div > div {
        background: #ffffff !important;
    }

    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: #10253d !important;
        -webkit-text-fill-color: #10253d !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
        padding: 0.86rem 0.95rem !important;
        box-shadow: none !important;
        caret-color: #1b76a5 !important;
        cursor: text !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: rgba(16, 37, 61, 0.36) !important;
        -webkit-text-fill-color: rgba(16, 37, 61, 0.36) !important;
        font-weight: 500 !important;
    }

    div[data-testid="stTextInput"] [data-baseweb="input"] > div:focus-within {
        border-color: rgba(27, 118, 165, 0.42) !important;
        box-shadow: 0 0 0 3px rgba(52, 144, 220, 0.14) !important;
    }

    div[data-testid="stFormSubmitButton"] {
        margin-top: 0.6rem;
    }

    div[data-testid="stFormSubmitButton"] button {
        min-height: 2.9rem;
        border-radius: 16px !important;
        border: 0 !important;
        background: linear-gradient(135deg, #0b5a87 0%, #1c739f 54%, #2d91ac 100%) !important;
        color: #ffffff !important;
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        box-shadow: 0 16px 34px rgba(20, 92, 140, 0.24) !important;
    }

    div[data-testid="stFileUploaderDropzone"] {
        min-height: 138px;
        border-radius: 22px;
        border: 1.4px dashed rgba(46, 139, 135, 0.48);
        background: linear-gradient(180deg, rgba(248, 252, 251, 0.76), rgba(243, 248, 250, 0.68));
        padding: 1.15rem 0.9rem;
    }

    div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {
        color: rgba(16, 37, 61, 0.72);
        font-size: 0.84rem;
    }

    .status-strip {
        margin: 0 0 0.72rem;
        padding: 0.72rem 0.85rem;
        border-radius: 18px;
        border: 1px solid rgba(16, 37, 61, 0.10);
        background: rgba(251, 253, 255, 0.76);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    .status-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #10253d;
    }

    .status-copy {
        margin-top: 0.22rem;
        font-size: 0.82rem;
        line-height: 1.58;
        color: rgba(16, 37, 61, 0.68);
    }

    .status-idle {
        background: linear-gradient(180deg, rgba(251, 253, 255, 0.68), rgba(244, 248, 252, 0.58));
    }

    .status-loading {
        background: linear-gradient(180deg, rgba(235, 245, 252, 0.76), rgba(241, 248, 252, 0.62));
        border-color: rgba(27, 118, 165, 0.18);
    }

    .status-loaded {
        background: linear-gradient(180deg, rgba(239, 249, 246, 0.74), rgba(244, 251, 248, 0.60));
        border-color: rgba(46, 139, 135, 0.18);
    }

    .status-error {
        background: linear-gradient(180deg, rgba(255, 243, 242, 0.82), rgba(252, 245, 244, 0.68));
        border-color: rgba(207, 94, 87, 0.20);
    }

    .error-banner {
        margin-bottom: 0.8rem;
        padding: 0.9rem 1rem;
        border-radius: 20px;
        border: 1px solid rgba(207, 94, 87, 0.18);
        background: linear-gradient(180deg, rgba(255, 244, 243, 0.84), rgba(252, 246, 245, 0.70));
        box-shadow: 0 18px 40px rgba(84, 36, 32, 0.08);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }

    .error-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #922b21;
    }

    .error-copy {
        margin-top: 0.24rem;
        font-size: 0.84rem;
        line-height: 1.6;
        color: rgba(76, 36, 32, 0.80);
    }

    .result-card,
    .analytics-card {
        padding: 1rem 1.05rem;
        border-radius: 28px;
        background: linear-gradient(180deg, rgba(250, 253, 255, 0.84), rgba(241, 246, 252, 0.76));
        border: 1px solid rgba(255, 255, 255, 0.72);
        box-shadow: 0 24px 64px rgba(24, 52, 87, 0.14);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        overflow: hidden;
    }

    .result-zone {
        --result-stack-height: 880px;
    }

    .reserve-result-card {
        min-height: var(--result-stack-height);
        display: flex;
        flex-direction: column;
    }

    .plan-stack {
        min-height: var(--result-stack-height);
        display: grid;
        grid-template-rows: minmax(220px, 0.52fr) minmax(0, 1.48fr);
        gap: 0.85rem;
    }

    .mini-card {
        min-height: 0;
        display: flex;
        flex-direction: column;
    }

    .analytics-card {
        min-height: 360px;
    }

    .placeholder-card {
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: rgba(16, 37, 61, 0.56);
        font-size: 0.96rem;
        line-height: 1.8;
    }

    .card-label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: rgba(16, 37, 61, 0.58);
    }

    .card-title {
        margin-top: 0.28rem;
        font-size: 1.24rem;
        font-weight: 700;
        color: #10253d;
    }

    .card-note {
        margin-top: 0.24rem;
        font-size: 0.88rem;
        color: rgba(16, 37, 61, 0.66);
        line-height: 1.6;
    }

    .reserve-layout {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-top: 0.9rem;
        flex: 1 1 auto;
    }

    .reserve-score-card {
        width: 100%;
        padding: 1.15rem 1rem 1rem;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.62), rgba(245, 249, 253, 0.56));
        border: 1px solid rgba(16, 37, 61, 0.08);
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.36);
    }

    .reserve-score-layout {
        display: grid;
        grid-template-columns: minmax(250px, 0.92fr) minmax(280px, 1.08fr);
        gap: 1.1rem;
        align-items: center;
    }

    .reserve-score-visual {
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }

    .reserve-ring-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }

    .svg-progress-shell {
        position: relative;
        width: var(--progress-size, 270px);
        height: var(--progress-size, 270px);
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .svg-progress-shell svg {
        width: 100%;
        height: 100%;
        display: block;
        overflow: visible;
    }

    .svg-progress-track {
        stroke: rgba(16, 37, 61, 0.10);
    }

    .svg-progress-bar {
        stroke: #e45c57;
        transition: stroke-dashoffset 680ms ease;
    }

    .svg-progress-value {
        fill: #12304e;
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .svg-progress-caption {
        fill: rgba(16, 37, 61, 0.56);
        font-size: 0.33rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .svg-progress-demo-card {
        margin-top: 0.9rem;
        padding: 0.92rem 1rem 1rem;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(250, 253, 255, 0.84), rgba(241, 246, 252, 0.76));
        border: 1px solid rgba(255, 255, 255, 0.72);
        box-shadow: 0 20px 48px rgba(24, 52, 87, 0.12);
    }

    .svg-progress-demo-copy {
        margin: 0 0 0.7rem;
        font-size: 0.82rem;
        line-height: 1.55;
        color: rgba(16, 37, 61, 0.66);
    }

    .svg-progress-demo-stage {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.55rem 0 0.2rem;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.56);
        border: 1px solid rgba(16, 37, 61, 0.08);
    }

    .reserve-copy {
        width: 100%;
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
    }

    .reserve-grade-panel {
        display: flex;
        flex-direction: column;
        gap: 0.85rem;
        min-width: 0;
    }

    .reserve-grade-label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: rgba(16, 37, 61, 0.56);
    }

    .reserve-grade-current {
        padding: 1rem 1.05rem;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.64);
        border: 1px solid rgba(16, 37, 61, 0.08);
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.36);
    }

    .reserve-grade-meta {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }

    .reserve-grade-letter {
        width: 4.25rem;
        height: 4.25rem;
        border-radius: 18px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 2.1rem;
        font-weight: 800;
        color: #10253d;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(16, 37, 61, 0.08);
        box-shadow: 0 10px 24px rgba(16, 37, 61, 0.08);
    }

    .reserve-grade-copy {
        min-width: 0;
    }

    .reserve-grade-name {
        font-size: 0.96rem;
        font-weight: 700;
        color: #143151;
    }

    .reserve-grade-conclusion {
        margin-top: 0.22rem;
        font-size: 1.12rem;
        font-weight: 800;
        color: #10253d;
    }

    .reserve-grade-note {
        margin-top: 0.24rem;
        font-size: 0.84rem;
        line-height: 1.6;
        color: rgba(16, 37, 61, 0.64);
    }

    .reserve-grade-list {
        display: grid;
        gap: 0.52rem;
    }

    .reserve-grade-item {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        padding: 0.58rem 0.72rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.42);
        border: 1px solid rgba(16, 37, 61, 0.06);
    }

    .reserve-grade-item.active {
        background: rgba(255, 255, 255, 0.72);
        border-color: rgba(16, 37, 61, 0.12);
        box-shadow: 0 10px 22px rgba(16, 37, 61, 0.06);
    }

    .reserve-grade-swatch {
        width: 0.78rem;
        height: 0.78rem;
        border-radius: 999px;
        flex: 0 0 auto;
    }

    .reserve-grade-item-text {
        display: flex;
        align-items: baseline;
        gap: 0.42rem;
        min-width: 0;
        color: #10253d;
    }

    .reserve-grade-item-label {
        font-size: 0.84rem;
        font-weight: 800;
    }

    .reserve-grade-item-title {
        font-size: 0.84rem;
        color: rgba(16, 37, 61, 0.72);
    }

    .reserve-headline {
        font-size: 1.02rem;
        font-weight: 700;
        color: #183556;
        text-align: left;
    }

    .reserve-clinical-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
    }

    .reserve-clinical-card {
        padding: 0.92rem 0.95rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.64);
        border: 1px solid rgba(16, 37, 61, 0.08);
        min-height: 134px;
    }

    .reserve-clinical-card.wide {
        grid-column: 1 / -1;
        min-height: 156px;
    }

    .clinical-label {
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(16, 37, 61, 0.52);
    }

    .clinical-title {
        margin-top: 0.18rem;
        font-size: 0.98rem;
        font-weight: 700;
        line-height: 1.45;
        color: #10253d;
    }

    .clinical-value {
        margin-top: 0.42rem;
        font-size: 1.24rem;
        font-weight: 800;
        color: #12304e;
        line-height: 1.1;
    }

    .clinical-range {
        margin-top: 0.28rem;
        font-size: 0.83rem;
        color: rgba(16, 37, 61, 0.62);
    }

    .clinical-copy {
        margin-top: 0.38rem;
        font-size: 0.83rem;
        line-height: 1.55;
        color: rgba(16, 37, 61, 0.64);
    }

    .clinical-badges {
        margin-top: 0.72rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .clinical-badge {
        padding: 0.34rem 0.62rem;
        border-radius: 999px;
        background: rgba(223, 91, 87, 0.10);
        border: 1px solid rgba(223, 91, 87, 0.12);
        font-size: 0.78rem;
        color: #b1484d;
    }

    .ohss-shell {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        height: 100%;
    }

    .ohss-header {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 0.8rem;
        margin-top: 0.9rem;
    }

    .ohss-value {
        font-size: 2.55rem;
        font-weight: 800;
        line-height: 1;
        color: #10253d;
    }

    .ohss-level {
        margin-top: 0.2rem;
        font-size: 0.9rem;
        color: rgba(16, 37, 61, 0.66);
    }

    .risk-pill {
        padding: 0.46rem 0.72rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.70);
        border: 1px solid rgba(16, 37, 61, 0.08);
        font-size: 0.82rem;
        color: rgba(16, 37, 61, 0.72);
    }

    .risk-bar {
        margin-top: 0.82rem;
        height: 14px;
        border-radius: 999px;
        background: rgba(16, 37, 61, 0.08);
        overflow: hidden;
        box-shadow: inset 0 1px 4px rgba(16, 37, 61, 0.10);
    }

    .risk-bar-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #ffce80 0%, #f0945a 42%, #d85e57 100%);
        box-shadow: 0 0 18px rgba(216, 94, 87, 0.28);
    }

    .mini-note {
        margin-top: 0.65rem;
        font-size: 0.86rem;
        color: rgba(16, 37, 61, 0.64);
        line-height: 1.55;
        max-width: 96%;
    }

    .metric-grid {
        margin-top: 0.82rem;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem;
    }

    .metric-box {
        padding: 0.8rem 0.84rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.64);
        border: 1px solid rgba(16, 37, 61, 0.08);
        min-height: 112px;
    }

    .metric-label {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(16, 37, 61, 0.56);
    }

    .metric-value {
        margin-top: 0.46rem;
        font-size: 0.95rem;
        font-weight: 700;
        color: #10253d;
        line-height: 1.5;
    }

    .metric-copy {
        margin-top: 0.26rem;
        font-size: 0.82rem;
        color: rgba(16, 37, 61, 0.64);
        line-height: 1.45;
    }

    .rank-row {
        margin-top: 0.95rem;
    }

    .rank-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 0.38rem;
    }

    .rank-city {
        font-size: 0.9rem;
        font-weight: 600;
        color: #13304e;
    }

    .rank-value {
        font-size: 0.85rem;
        font-weight: 700;
        color: rgba(16, 37, 61, 0.68);
    }

    .rank-track {
        height: 10px;
        border-radius: 999px;
        background: rgba(16, 37, 61, 0.08);
        overflow: hidden;
    }

    .rank-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #2a7aa8 0%, #44a4b3 100%);
    }

    .rank-fill-traffic {
        background: linear-gradient(90deg, #eb9c54 0%, #da6458 100%);
    }

    @media (max-width: 1200px) {
        .title-row {
            flex-direction: column;
            text-align: center;
        }

        .page-title,
        .page-subtitle {
            text-align: center;
        }

        .upload-toolbar-note,
        .title-upload-status {
            text-align: center;
        }

        .ohss-header {
            flex-direction: column;
            align-items: flex-start;
        }

        .reserve-clinical-grid {
            grid-template-columns: 1fr;
        }

        .reserve-score-layout {
            grid-template-columns: 1fr;
        }

        .reserve-score-visual {
            justify-content: center;
        }

        .svg-progress-shell {
            width: min(var(--progress-size, 270px), 228px);
            height: min(var(--progress-size, 270px), 228px);
        }

        .form-align-gap {
            height: 0;
        }
    }

</style>
"""

st.markdown(CSS.replace("__BACKGROUND__", BACKGROUND_URI), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_system():
    try:
        return system_loader.load_system()
    except Exception as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


def fmt_value(value, digits=1, unit="") -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value)}{unit}"
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def parse_numeric_text(
    raw: str,
    label: str,
    min_value=None,
    max_value=None,
    integer: bool = False,
    default_value=None,
):
    value_text = (raw or "").strip().replace(",", "")
    if value_text == "":
        if default_value is None:
            raise ValueError(f"{label} 不能为空")
        number = int(default_value) if integer else float(default_value)
    else:
        try:
            number = int(value_text) if integer else float(value_text)
        except ValueError as exc:
            raise ValueError(f"{label} 请输入有效数字") from exc
    if min_value is not None and number < min_value:
        raise ValueError(f"{label} 不能小于 {min_value}")
    if max_value is not None and number > max_value:
        raise ValueError(f"{label} 不能大于 {max_value}")
    return number


def normalize_header_name(text: str) -> str:
    value = str(text or "").strip().lower()
    value = (
        value.replace("（", "(")
        .replace("）", ")")
        .replace("／", "/")
        .replace("—", "-")
        .replace("–", "-")
    )
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value)


UPLOAD_HEADER_ALIASES = {
    "patient_name": {
        "姓名",
        "患者姓名",
        "patientname",
        "name",
    },
    "patient_id": {
        "患者编号",
        "编号",
        "病历号",
        "id",
        "patientid",
        "caseid",
    },
    "menarche_age": {
        "初潮年龄",
        "月经年龄",
        "menarcheage",
        "menstrualage",
    },
    "cycle_length": {
        "月经周期",
        "月经周期天",
        "月经周期天数",
        "cyclelength",
        "menstrualcycle",
        "menstrualcycledays",
    },
    "BMI": {
        "bmi",
        "体重指数",
    },
    "AMH": {
        "amh",
        "amhngml",
        "antimullerianhormone",
    },
    "AFC": {
        "afc",
        "窦卵泡数",
        "窦卵泡计数",
        "antralfolliclecount",
    },
    "LH": {
        "lh",
        "basallh",
        "基础lh",
    },
    "FSH": {
        "fsh",
        "bfsh",
        "基础fsh",
        "basalfsh",
        "bfs",
    },
    "Age": {
        "年龄",
        "age",
        "年龄years",
        "womanage",
        "patientage",
    },
}


def canonical_upload_field(header: str) -> Optional[str]:
    normalized = normalize_header_name(header)
    if not normalized:
        return None
    for field, aliases in UPLOAD_HEADER_ALIASES.items():
        if normalized in aliases:
            return field
    for field, aliases in UPLOAD_HEADER_ALIASES.items():
        if any(alias and alias in normalized for alias in aliases):
            return field
    return None


def coerce_optional_number(value) -> Optional[Union[int, float]]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def input_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


@st.cache_data(show_spinner=False)
def build_upload_template_bytes() -> bytes:
    template = pd.DataFrame(
        [
            {
                "patient_id": "FJFY-001",
                "patient_name": "患者A",
                "Age": 32,
                "AMH": 2.5,
                "FSH": 8.0,
                "AFC": 10,
                "cycle_length": 28,
                "menarche_age": 13,
                "BMI": 21.3,
                "LH": 6.0,
            },
            {
                "patient_id": "FJFY-002",
                "patient_name": "患者B",
                "Age": 37,
                "AMH": 1.4,
                "FSH": 9.8,
                "AFC": 7,
                "cycle_length": 29,
                "menarche_age": 14,
                "BMI": 22.1,
                "LH": 5.4,
            },
        ]
    )
    buffer = BytesIO()
    template.to_excel(buffer, index=False)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def load_uploaded_patients(file_bytes: bytes, filename: str) -> List[Dict[str, Union[str, int, float, None]]]:
    suffix = Path(filename).suffix.lower()
    bio = BytesIO(file_bytes)
    if suffix == ".csv":
        frame = pd.read_csv(bio)
    else:
        frame = pd.read_excel(bio)

    columns = {}
    for column in frame.columns:
        field = canonical_upload_field(str(column))
        if field and field not in columns:
            columns[field] = column

    patients: List[Dict[str, Union[str, int, float, None]]] = []
    for row_index, (_, row) in enumerate(frame.iterrows(), start=1):
        patient = {
            "row_index": row_index,
            "patient_name": None,
            "patient_id": None,
            "Age": None,
            "AMH": None,
            "FSH": None,
            "AFC": None,
            "cycle_length": None,
            "menarche_age": None,
            "BMI": None,
            "LH": None,
        }
        for field, column in columns.items():
            raw_value = row[column]
            if field in {"patient_name", "patient_id"}:
                if raw_value is not None and not pd.isna(raw_value):
                    text = str(raw_value).strip()
                    patient[field] = text or None
            else:
                patient[field] = coerce_optional_number(raw_value)
        if any(
            patient.get(key) is not None
            for key in ["Age", "AMH", "FSH", "AFC", "cycle_length", "menarche_age", "BMI", "LH"]
        ):
            label = patient.get("patient_name") or patient.get("patient_id") or f"患者 {row_index}"
            patient["patient_label"] = label
            patients.append(patient)
    return patients


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def interpolate_curve(x_value: float, anchors: List[Tuple[float, float]]) -> float:
    if x_value <= anchors[0][0]:
        return anchors[0][1]
    for (left_x, left_y), (right_x, right_y) in zip(anchors, anchors[1:]):
        if x_value <= right_x:
            span = right_x - left_x
            if span == 0:
                return right_y
            ratio = (x_value - left_x) / span
            return left_y + ratio * (right_y - left_y)
    return anchors[-1][1]


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_ovarepred_reference(
    age: int,
    amh: float,
    fsh: float,
    afc: int,
    cycle_length: int,
    menarche_age: int,
) -> Optional[Dict[str, Union[float, str]]]:
    payload = {
        "age": int(age),
        "amh": float(amh),
        "fsh": float(fsh),
        "afc": int(afc),
        "startDay": int(cycle_length),
        "endDay": int(cycle_length),
        "menarcheAge": int(menarche_age),
        "calType": "aafamodel",
    }
    request = Request(
        OVAREPRED_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json;charset=UTF-8"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=3) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    if raw.get("code") != 200 or not isinstance(raw.get("data"), dict):
        return None

    data = raw["data"]
    try:
        return {
            "source": "OvaRePred",
            "mode": "reference",
            "dor_age": float(data["dorBeginAge"]),
            "dor_age_lower": float(data["dorBeginAgeLower"]),
            "dor_age_upper": float(data["dorBeginAgeUpper"]),
            "peri_age": float(data["perimenopauseAge"]),
            "peri_age_lower": float(data["perimenopauseAgeLower"]),
            "peri_age_upper": float(data["perimenopauseAgeUpper"]),
            "tool_score": float(data["v1"]),
            "tool_level_cn": str(data.get("levelch") or ""),
            "tool_level_en": str(data.get("levelen") or ""),
        }
    except Exception:
        return None


def estimate_reserve_reference(age: int, reserve_score: float) -> Dict[str, Union[float, str]]:
    score = clamp(float(reserve_score), 0.0, 100.0)
    dor_gap = clamp((score - 30.0) * 0.22, 0.0, 18.0)
    peri_gap = clamp(dor_gap + 11.0, 8.0, 28.0)
    interval = clamp(3.6 - score * 0.02, 1.4, 3.6)
    dor_age = clamp(float(age) + dor_gap, float(age), 52.0)
    peri_age = clamp(float(age) + peri_gap, dor_age + 5.0, 60.0)
    return {
        "source": "Local estimate",
        "mode": "estimate",
        "dor_age": dor_age,
        "dor_age_lower": max(float(age), dor_age - interval),
        "dor_age_upper": dor_age + interval,
        "peri_age": peri_age,
        "peri_age_lower": max(dor_age + 3.5, peri_age - interval - 0.8),
        "peri_age_upper": peri_age + interval + 0.8,
        "tool_score": score,
        "tool_level_cn": "",
        "tool_level_en": "",
    }


def age_based_embryo_aneuploidy(age: float) -> Dict[str, Union[float, str]]:
    risk = interpolate_curve(
        float(age),
        [
            (25.0, 28.0),
            (30.0, 32.0),
            (35.0, 40.0),
            (37.0, 48.0),
            (40.0, 62.0),
            (42.0, 72.0),
            (45.0, 84.0),
            (50.0, 92.0),
        ],
    )
    risk = clamp(risk, 8.0, 96.0)
    euploid = 100.0 - risk
    if risk < 35:
        label = "相对较低"
    elif risk < 50:
        label = "逐步升高"
    elif risk < 70:
        label = "明显升高"
    else:
        label = "高风险区间"
    return {"risk": risk, "euploid": euploid, "label": label}


def reserve_profile(por_prob: float) -> Dict[str, Union[float, str]]:
    score = max(0.0, min(100.0, 100.0 - por_prob * 100.0))
    if score >= 75:
        title = "储备功能较好"
    elif score >= 55:
        title = "储备功能中等"
    else:
        title = "储备功能偏低"
    return {"score": score, "title": title, "accent": "#df5b57"}


def reserve_grade(score: float) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    score = clamp(float(score), 0.0, 100.0)
    levels = [
        {"letter": "A", "label": "A级", "title": "储备功能优秀", "accent": "#96d66a", "threshold": 88},
        {"letter": "B", "label": "B级", "title": "储备功能较好", "accent": "#e0d85f", "threshold": 75},
        {"letter": "C", "label": "C级", "title": "储备功能中等", "accent": "#f2a276", "threshold": 55},
        {"letter": "D", "label": "D级", "title": "储备功能偏低", "accent": "#ef6463", "threshold": 0},
    ]
    current = levels[-1]
    for item in levels:
        if score >= int(item["threshold"]):
            current = item
            break
    return {
        "letter": str(current["letter"]),
        "label": str(current["label"]),
        "title": str(current["title"]),
        "accent": str(current["accent"]),
        "levels": levels,
    }


def ohss_profile(hor_prob: float) -> Dict[str, Union[float, str]]:
    percent = max(0.0, min(100.0, hor_prob * 100.0))
    if hor_prob < HOR_THRESHOLDS["low"]:
        level = "OHSS 风险较低"
    elif hor_prob < HOR_THRESHOLDS["high"]:
        level = "OHSS 风险中等"
    else:
        level = "OHSS 风险较高"
    return {"percent": percent, "level": level}


def error_banner_html(title: str, message: str) -> str:
    return (
        "<div class='error-banner'>"
        f"<div class='error-title'>{escape(title)}</div>"
        f"<div class='error-copy'>{escape(message)}</div>"
        "<div class='error-copy'>建议检查依赖是否完整；如有预训练模型，请放入 models/ 目录；否则系统将继续使用演示模型逻辑。</div>"
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
    clamped = clamp(float(value), 0.0, 100.0)
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
                <circle
                    class='svg-progress-track'
                    cx='{center:.2f}'
                    cy='{center:.2f}'
                    r='{radius:.2f}'
                    fill='none'
                    stroke-width='{stroke_width}'
                ></circle>
                <circle
                    class='svg-progress-bar'
                    cx='{center:.2f}'
                    cy='{center:.2f}'
                    r='{radius:.2f}'
                    fill='none'
                    stroke-width='{stroke_width}'
                    stroke-linecap='round'
                    stroke-dasharray='{circumference:.3f}'
                    stroke-dashoffset='{dash_offset:.3f}'
                    transform='rotate(-90 {center:.2f} {center:.2f})'
                ></circle>
                <text
                    class='svg-progress-value'
                    x='50%'
                    y='{value_y:.2f}'
                    text-anchor='middle'
                    dominant-baseline='middle'
                    style='font-size:{font_size:.1f}px;'
                >{clamped:.1f}%</text>
                <text
                    class='svg-progress-caption'
                    x='50%'
                    y='{caption_y:.2f}'
                    text-anchor='middle'
                    dominant-baseline='middle'
                    style='font-size:{caption_size:.1f}px;'
                >{escape(caption.upper())}</text>
            </svg>
        </div>
        """
    ).strip()


def sync_patient_to_form_inputs(patient: Dict[str, Union[str, int, float, None]]) -> None:
    st.session_state[FORM_INPUT_KEYS["reserve_amh"]] = input_text(patient.get("AMH"))
    st.session_state[FORM_INPUT_KEYS["reserve_age"]] = input_text(patient.get("Age"))
    st.session_state[FORM_INPUT_KEYS["reserve_fsh"]] = input_text(patient.get("FSH"))
    st.session_state[FORM_INPUT_KEYS["reserve_afc"]] = input_text(patient.get("AFC"))
    st.session_state[FORM_INPUT_KEYS["reserve_cycle"]] = input_text(patient.get("cycle_length"))
    st.session_state[FORM_INPUT_KEYS["reserve_menarche"]] = input_text(patient.get("menarche_age"))

    st.session_state[FORM_INPUT_KEYS["plan_age"]] = input_text(patient.get("Age"))
    st.session_state[FORM_INPUT_KEYS["plan_cycle"]] = input_text(patient.get("cycle_length"))
    st.session_state[FORM_INPUT_KEYS["plan_bmi"]] = input_text(patient.get("BMI"))
    st.session_state[FORM_INPUT_KEYS["plan_afc"]] = input_text(patient.get("AFC"))
    st.session_state[FORM_INPUT_KEYS["plan_fsh"]] = input_text(patient.get("FSH"))
    st.session_state[FORM_INPUT_KEYS["plan_lh"]] = input_text(patient.get("LH"))
    st.session_state[FORM_INPUT_KEYS["plan_amh"]] = input_text(patient.get("AMH"))


def update_reserve_state(age, amh, fsh, afc, cycle_length=None, menarche_age=None) -> None:
    st.session_state.reserve_error = None
    reserve_patient_display = {
        "Age": age,
        "AMH": amh,
        "FSH": fsh,
        "AFC": afc,
        "cycle_length": cycle_length,
        "menarche_age": menarche_age,
        "embryo_aneuploidy": age_based_embryo_aneuploidy(age),
    }
    if cycle_length is not None and menarche_age is not None:
        reserve_reference = fetch_ovarepred_reference(age, amh, fsh, afc, cycle_length, menarche_age)
        if reserve_reference is not None:
            reserve_patient_display["reserve_reference"] = reserve_reference
    st.session_state.reserve_patient = reserve_patient_display
    if ensure_model_ready():
        try:
            st.session_state.reserve_report = st.session_state.cds.evaluate_patient(
                build_reserve_patient(age, amh, fsh, afc)
            )
        except Exception as exc:
            st.session_state.reserve_report = None
            st.session_state.reserve_error = str(exc).strip() or "未知错误"
    else:
        st.session_state.reserve_report = None
        st.session_state.reserve_error = st.session_state.model_error or "模型初始化失败"


def update_plan_state(age, amh, fsh, afc, lh, bmi=None, cycle_length=None) -> None:
    st.session_state.plan_error = None
    plan_patient_display = {
        "Age": age,
        "AMH": amh,
        "FSH": fsh,
        "AFC": afc,
        "LH": lh,
        "BMI": bmi,
        "cycle_length": cycle_length,
    }
    st.session_state.plan_patient = plan_patient_display
    if ensure_model_ready():
        try:
            st.session_state.plan_report = st.session_state.cds.evaluate_patient(
                build_plan_patient(age, amh, fsh, afc, lh)
            )
        except Exception as exc:
            st.session_state.plan_report = None
            st.session_state.plan_error = str(exc).strip() or "未知错误"
    else:
        st.session_state.plan_report = None
        st.session_state.plan_error = st.session_state.model_error or "模型初始化失败"


def activate_uploaded_patient(index: int) -> None:
    patients = st.session_state.uploaded_patients
    if not patients:
        return
    safe_index = max(0, min(index, len(patients) - 1))
    patient = patients[safe_index]
    st.session_state.current_patient_index = safe_index
    st.session_state.upload_notice = None
    sync_patient_to_form_inputs(patient)

    reserve_missing = [label for key, label in [("Age", "年龄"), ("AMH", "AMH"), ("FSH", "FSH"), ("AFC", "AFC")] if patient.get(key) is None]
    if reserve_missing:
        st.session_state.reserve_report = None
        st.session_state.reserve_patient = None
        st.session_state.reserve_error = f"当前患者缺少储备评估必要字段：{'、'.join(reserve_missing)}"
    else:
        update_reserve_state(
            int(patient["Age"]),
            float(patient["AMH"]),
            float(patient["FSH"]),
            int(patient["AFC"]),
            int(patient["cycle_length"]) if patient.get("cycle_length") is not None else None,
            int(patient["menarche_age"]) if patient.get("menarche_age") is not None else None,
        )

    plan_missing = [label for key, label in [("Age", "年龄"), ("AMH", "AMH"), ("FSH", "FSH"), ("AFC", "AFC"), ("LH", "LH")] if patient.get(key) is None]
    if plan_missing:
        st.session_state.plan_report = None
        st.session_state.plan_patient = None
        st.session_state.plan_error = f"当前患者缺少促排方案必要字段：{'、'.join(plan_missing)}"
    else:
        update_plan_state(
            int(patient["Age"]),
            float(patient["AMH"]),
            float(patient["FSH"]),
            int(patient["AFC"]),
            float(patient["LH"]),
            float(patient["BMI"]) if patient.get("BMI") is not None else None,
            int(patient["cycle_length"]) if patient.get("cycle_length") is not None else None,
        )

    missing_optional = []
    if patient.get("cycle_length") is None:
        missing_optional.append("月经周期")
    if patient.get("menarche_age") is None:
        missing_optional.append("初潮年龄")
    if patient.get("BMI") is None:
        missing_optional.append("BMI")
    if missing_optional:
        st.session_state.upload_notice = f"当前患者未识别到 { '、'.join(missing_optional) }，相关展示项将按已识别数据输出。"


def reserve_result_html(report: Optional[Dict], patient: Optional[Dict]) -> str:
    if not report or not patient:
        return (
            "<div class='result-zone'>"
            "<div class='result-card reserve-result-card placeholder-card'>"
            "录入左侧指标后，这里生成卵巢储备功能百分比结果。"
            "</div>"
            "</div>"
        )

    por_prob = report["risk_levels"]["POR"]["prob"]
    profile = reserve_profile(por_prob)
    filled_percent = max(0.0, min(100.0, float(profile["score"])))
    grade = reserve_grade(filled_percent)
    reference = patient.get("reserve_reference") or estimate_reserve_reference(
        int(patient.get("Age") or 32),
        float(profile["score"]),
    )
    embryo = patient.get("embryo_aneuploidy") or age_based_embryo_aneuploidy(float(patient.get("Age") or 32))
    source_text = "公开工具参考" if reference.get("mode") == "reference" else "本地趋势估算"
    grade_list_html = "".join(
        (
            "<div class='reserve-grade-item"
            + (" active" if str(item["letter"]) == str(grade["letter"]) else "")
            + "'>"
            + f"<span class='reserve-grade-swatch' style='background:{escape(str(item['accent']))};'></span>"
            + "<div class='reserve-grade-item-text'>"
            + f"<span class='reserve-grade-item-label'>{escape(str(item['label']))}</span>"
            + f"<span class='reserve-grade-item-title'>{escape(str(item['title']))}</span>"
            + "</div>"
            + "</div>"
        )
        for item in grade["levels"]
    )

    return dedent(
        f"""
        <div class='result-zone'>
            <div class='result-card reserve-result-card'>
                <div class='card-label'>Ovarian Reserve Output</div>
                <div class='card-title'>卵巢储备功能结果</div>
                <div class='reserve-layout'>
                    <div class='reserve-score-card'>
                        <div class='reserve-score-layout'>
                            <div class='reserve-score-visual'>
                                <div class='reserve-ring-wrap'>
                                    {svg_circular_progress_html(filled_percent, size=252, stroke_width=22, caption="Reserve Index")}
                                </div>
                            </div>
                            <div class='reserve-grade-panel'>
                                <div class='reserve-grade-label'>级别评级</div>
                                <div class='reserve-grade-current'>
                                    <div class='reserve-grade-meta'>
                                        <div class='reserve-grade-letter' style='color:{escape(str(grade["accent"]))};'>{escape(str(grade["letter"]))}</div>
                                        <div class='reserve-grade-copy'>
                                            <div class='reserve-grade-name'>{escape(str(grade["label"]))}</div>
                                            <div class='reserve-grade-conclusion'>{escape(str(profile["title"]))}</div>
                                            <div class='reserve-grade-note'>当前储备指数位于该评级区间，评级结果与环形百分比同步对应。</div>
                                        </div>
                                    </div>
                                </div>
                                <div class='reserve-grade-list'>
                                    {grade_list_html}
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class='reserve-copy'>
                        <div class='reserve-headline'>预测结果</div>
                        <div class='reserve-clinical-grid'>
                            {clinical_metric_card(
                                "Starting age of Diminished Ovarian Reserve",
                                "卵巢储备减退起始年龄",
                                f"{float(reference['dor_age']):.1f} 岁",
                                "表示储备功能进入明显下降阶段的参考起点。",
                                f"区间 [{float(reference['dor_age_lower']):.1f}, {float(reference['dor_age_upper']):.1f}]",
                            )}
                            {clinical_metric_card(
                                "Starting age of peri-menopause",
                                "围绝经期起始年龄",
                                f"{float(reference['peri_age']):.1f} 岁",
                                "表示围绝经期变化开始出现的参考年龄节点。",
                                f"区间 [{float(reference['peri_age_lower']):.1f}, {float(reference['peri_age_upper']):.1f}]",
                            )}
                            {clinical_metric_card(
                                "Age-related embryo aneuploidy",
                                "按年龄估算的胚胎染色体异常率",
                                f"{float(embryo['risk']):.1f}%",
                                "基于女性年龄的趋势估算，不等同于实际 PGT-A 检测结果。",
                                f"预估整倍体胚胎比例 {float(embryo['euploid']):.1f}%",
                                wide=True,
                                badges=[
                                    f"风险分层 {escape(str(embryo['label']))}",
                                    f"年龄 {fmt_value(patient.get('Age'))} 岁",
                                    source_text,
                                ],
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    ).strip()


def plan_result_html(report: Optional[Dict], patient: Optional[Dict]) -> str:
    if not report or not patient:
        return dedent(
            """
            <div class='result-zone'>
                <div class='plan-stack'>
                    <div class='result-card mini-card placeholder-card'>录入右侧指标后，这里显示 OHSS 风险百分比。</div>
                    <div class='result-card mini-card placeholder-card'>生成后，这里显示获卵数、推荐方案、起始 FSH、LH 支持与 FSH 类型。</div>
                </div>
            </div>
            """
        ).strip()

    hor_prob = report["risk_levels"]["HOR"]["prob"]
    profile = ohss_profile(hor_prob)
    recs = report["recommendations"]
    metrics = [
        metric_box("预期获卵数", recs.get("est_oocytes") or "--", "基于 AMH / AFC / 年龄估算"),
        metric_box("推荐方案", recs["protocol"], "当前更匹配的促排方向"),
        metric_box("起始 FSH", recs["fsh_dose"], "推荐起始剂量区间"),
        metric_box("LH 支持", recs["lh_support"], "是否考虑外源性 LH"),
        metric_box("FSH 类型", recs["fsh_type"], "药物来源建议"),
        metric_box(
            "录入摘要",
            f"年龄 {fmt_value(patient.get('Age'))} / BMI {fmt_value(patient.get('BMI'), 1)} / 周期 {fmt_value(patient.get('cycle_length'))}",
            "月经周期与 BMI 当前仅作病例记录",
        ),
    ]

    return dedent(
        f"""
            <div class='result-zone'>
                <div class='plan-stack'>
                    <div class='result-card mini-card'>
                    <div class='card-label'>Ovulation Stimulation Plan</div>
                    <div class='card-title'>促排卵方案</div>
                    <div class='ohss-shell'>
                        <div class='ohss-header'>
                            <div>
                                <div class='ohss-value'>{profile["percent"]:.1f}%</div>
                                <div class='ohss-level'>{escape(str(profile["level"]))}</div>
                            </div>
                        </div>
                        <div class='risk-bar'>
                            <div class='risk-bar-fill' style='width:{profile["percent"]:.1f}%'></div>
                        </div>
                        <div class='mini-note'>
                            当前后端模型直接使用年龄、AMH、AFC、FSH、LH；月经周期和 BMI 暂未作为模型特征参与计算。
                        </div>
                    </div>
                </div>
                <div class='result-card mini-card'>
                    <div class='card-label'>Stimulation Plan</div>
                    <div class='card-title'>方案建议</div>
                    <div class='metric-grid'>
                        {"".join(metrics)}
                    </div>
                </div>
            </div>
        </div>
        """
    ).strip()


def ensure_model_ready() -> bool:
    if st.session_state.cds is not None:
        st.session_state.model_ready = True
        st.session_state.model_error = None
        return True

    try:
        cds, model_source = load_system()
        st.session_state.cds = cds
        st.session_state.model_ready = True
        st.session_state.model_source = model_source
        st.session_state.model_error = None
        return True
    except Exception as exc:
        error_text = str(exc).strip() or "未知错误"
        st.session_state.cds = None
        st.session_state.model_ready = False
        st.session_state.model_source = "初始化失败"
        st.session_state.model_error = error_text
        return False


SESSION_DEFAULTS = {
    "cds": None,
    "model_ready": False,
    "model_source": "未加载",
    "model_error": None,
    "reserve_report": None,
    "reserve_patient": None,
    "reserve_error": None,
    "plan_report": None,
    "plan_patient": None,
    "plan_error": None,
    "uploaded_patients": [],
    "uploaded_file_digest": "",
    "uploaded_file_name": "",
    "current_patient_index": 0,
    "upload_notice": None,
}

SESSION_DEFAULTS.update({key: "" for key in FORM_INPUT_KEYS.values()})

session_state_utils.initialize_session_state(FORM_INPUT_KEYS)


def build_reserve_patient(age, amh, fsh, afc) -> Dict:
    return {
        "Age": age,
        "AMH": amh,
        "FSH": fsh,
        "AFC": afc,
        "LH": None,
        "P": None,
        "Weight": None,
        "WBC": None,
        "Duration": None,
        "POIorDOR": "No",
        "PCOS": "No",
        "PLT": None,
        "DBP": None,
        "ALT": None,
        "RBC": None,
    }


def build_plan_patient(age, amh, fsh, afc, lh) -> Dict:
    return {
        "Age": age,
        "AMH": amh,
        "FSH": fsh,
        "AFC": afc,
        "LH": lh,
        "Weight": None,
        "Duration": None,
        "POIorDOR": "No",
        "PCOS": "No",
        "PLT": None,
        "P": None,
        "WBC": None,
        "DBP": None,
        "ALT": None,
        "RBC": None,
    }


def inject_scroll_persistence_script() -> None:
    components.html(
        dedent(
            """
            <script>
            (function() {
              const parentWindow = window.parent;
              const key = "ovarian_app_scroll_y";
              try {
                if (!parentWindow.__ovarianScrollPersistBound) {
                  parentWindow.__ovarianScrollPersistBound = true;
                  let ticking = false;
                  parentWindow.addEventListener("scroll", () => {
                    if (ticking) {
                      return;
                    }
                    ticking = true;
                    parentWindow.requestAnimationFrame(() => {
                      parentWindow.sessionStorage.setItem(
                        key,
                        String(parentWindow.scrollY || parentWindow.pageYOffset || 0)
                      );
                      ticking = false;
                    });
                  }, { passive: true });
                }
                const saved = parentWindow.sessionStorage.getItem(key);
                if (saved !== null) {
                  const top = Number(saved) || 0;
                  parentWindow.requestAnimationFrame(() => {
                    parentWindow.scrollTo({ top, behavior: "auto" });
                  });
                }
              } catch (error) {}
            })();
            </script>
            """
        ).strip(),
        height=0,
    )


def inject_figma_capture_script() -> None:
    components.html(
        dedent(
            """
            <script>
            (function() {
              const parentDoc = window.parent && window.parent.document;
              if (!parentDoc) {
                return;
              }
              const scriptId = "figma-mcp-capture-script";
              if (parentDoc.getElementById(scriptId)) {
                return;
              }
              const script = parentDoc.createElement("script");
              script.id = scriptId;
              script.src = "https://mcp.figma.com/mcp/html-to-design/capture.js";
              script.async = true;
              parentDoc.head.appendChild(script);
            })();
            </script>
            """
        ).strip(),
        height=0,
    )


layout_components.inject_scroll_persistence_script()
layout_components.inject_figma_capture_script()
layout_components.render_title_frame(TITLE_LOGO_URI)

toolbar_left_col, toolbar_right_col = st.columns([1, 1], gap="medium")

with toolbar_right_col:
    narrow_left, narrow_right = st.columns([0.62, 0.38], gap="small")
    with narrow_right:
        st.markdown("<div class='upload-toolbar'><div class='upload-toolbar-note'>File Input</div></div>", unsafe_allow_html=True)
        template_download_uri = media_utils.encode_bytes_to_data_uri(
            upload_parser_service.build_upload_template_bytes(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        with st.popover("文件输入"):
            st.markdown(
                "<div class='upload-popover-copy'>支持 Excel 或 CSV 文件。单患者文件会直接回填；多患者文件默认先载入第 1 位。</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<a class='upload-template-link' href='{template_download_uri}' download='ovarian_batch_template.xlsx'>点击下载模板</a>",
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "上传患者文件",
                type=["xlsx", "xls", "csv"],
                help="支持常见中英文列名自动识别。",
                label_visibility="collapsed",
                key="batch_patient_file",
            )

            if uploaded_file is not None:
                uploaded_bytes = uploaded_file.getvalue()
                uploaded_digest = hashlib.md5(uploaded_bytes).hexdigest()
                if uploaded_digest != st.session_state.uploaded_file_digest:
                    try:
                        patients = upload_parser_service.load_uploaded_patients(uploaded_bytes, uploaded_file.name)
                        st.session_state.uploaded_patients = patients
                        st.session_state.uploaded_file_digest = uploaded_digest
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.session_state.current_patient_index = 0
                        st.session_state.uploaded_patient_selector = 0
                        if patients:
                            activate_uploaded_patient(0)
                            st.session_state.upload_notice = f"已识别 {len(patients)} 位患者，当前载入第 1 位。"
                        else:
                            st.session_state.reserve_report = None
                            st.session_state.reserve_patient = None
                            st.session_state.plan_report = None
                            st.session_state.plan_patient = None
                            st.session_state.upload_notice = "未识别到有效患者行，请检查 Excel 列名和内容。"
                    except Exception as exc:
                        st.session_state.uploaded_patients = []
                        st.session_state.uploaded_file_digest = ""
                        st.session_state.uploaded_file_name = ""
                        st.session_state.reserve_report = None
                        st.session_state.reserve_patient = None
                        st.session_state.plan_report = None
                        st.session_state.plan_patient = None
                        st.session_state.upload_notice = f"文件解析失败：{str(exc).strip() or '未知错误'}"

            if st.session_state.uploaded_patients:
                options = list(range(len(st.session_state.uploaded_patients)))
                selected_index = st.selectbox(
                    "选择患者",
                    options=options,
                    index=min(st.session_state.current_patient_index, len(options) - 1),
                    format_func=lambda idx: str(st.session_state.uploaded_patients[idx].get("patient_label") or f"患者 {idx + 1}"),
                    key="uploaded_patient_selector",
                )
                if selected_index != st.session_state.current_patient_index:
                    activate_uploaded_patient(selected_index)
        if st.session_state.upload_notice:
            st.markdown(
                f"<div class='title-upload-status'>{escape(str(st.session_state.upload_notice))}</div>",
                unsafe_allow_html=True,
            )

left_col, right_col = st.columns([1, 1], gap="medium")

with left_col:
    with st.form("reserve_form", clear_on_submit=False):
        st.markdown(
            "<div class='form-head'><div class='panel-label'>Reserve Section</div><div class='panel-title'>卵巢储备功能</div></div>",
            unsafe_allow_html=True,
        )
        row1 = st.columns(2)
        with row1[0]:
            reserve_amh_text = st.text_input(
                "AMH (ng/mL)",
                key=FORM_INPUT_KEYS["reserve_amh"],
                placeholder="eg 2.50",
            )
        with row1[1]:
            reserve_age_text = st.text_input("年龄", key=FORM_INPUT_KEYS["reserve_age"], placeholder="eg 32")

        row2 = st.columns(2)
        with row2[0]:
            reserve_fsh_text = st.text_input(
                "bFSH (IU/L)",
                key=FORM_INPUT_KEYS["reserve_fsh"],
                placeholder="eg 8.0",
            )
        with row2[1]:
            reserve_afc_text = st.text_input("AFC", key=FORM_INPUT_KEYS["reserve_afc"], placeholder="eg 10")

        row3 = st.columns(2)
        with row3[0]:
            reserve_cycle_text = st.text_input(
                "月经周期 (天)",
                key=FORM_INPUT_KEYS["reserve_cycle"],
                placeholder="eg 28",
            )
        with row3[1]:
            reserve_menarche_text = st.text_input(
                "初潮年龄 (岁)",
                key=FORM_INPUT_KEYS["reserve_menarche"],
                placeholder="eg 13",
            )

        st.markdown("<div class='form-align-gap'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='panel-note'>当前模型直接使用 AMH、年龄、bFSH、AFC。月经周期与初潮年龄暂不参与现有模型计算，仅作病例记录。</div>",
            unsafe_allow_html=True,
        )
        reserve_submitted = st.form_submit_button("生成卵巢储备功能结果", use_container_width=True)
    reserve_error_slot = st.empty()
    reserve_result_slot = st.empty()
    if st.session_state.reserve_error:
        reserve_error_slot.markdown(
            metric_cards_component.error_banner_html("卵巢储备功能结果生成失败", st.session_state.reserve_error),
            unsafe_allow_html=True,
        )
    reserve_result_slot.markdown(
        result_panel_components.reserve_result_html(st.session_state.reserve_report, st.session_state.reserve_patient),
        unsafe_allow_html=True,
    )

with right_col:
    with st.form("plan_form", clear_on_submit=False):
        st.markdown(
            "<div class='form-head'><div class='panel-label'>Plan Section</div><div class='panel-title'>促排卵方案</div></div>",
            unsafe_allow_html=True,
        )
        row1 = st.columns(2)
        with row1[0]:
            plan_age_text = st.text_input("年龄 ", key=FORM_INPUT_KEYS["plan_age"], placeholder="eg 32")
        with row1[1]:
            plan_cycle_text = st.text_input("月经周期", key=FORM_INPUT_KEYS["plan_cycle"], placeholder="eg 28")

        row2 = st.columns(2)
        with row2[0]:
            plan_bmi_text = st.text_input("BMI", key=FORM_INPUT_KEYS["plan_bmi"], placeholder="eg 21.3")
        with row2[1]:
            plan_afc_text = st.text_input("AFC ", key=FORM_INPUT_KEYS["plan_afc"], placeholder="eg 10")

        row3 = st.columns(2)
        with row3[0]:
            plan_fsh_text = st.text_input("FSH", key=FORM_INPUT_KEYS["plan_fsh"], placeholder="eg 8.0")
        with row3[1]:
            plan_lh_text = st.text_input("LH", key=FORM_INPUT_KEYS["plan_lh"], placeholder="eg 6.0")

        plan_amh_text = st.text_input("AMH", key=FORM_INPUT_KEYS["plan_amh"], placeholder="eg 2.50")

        st.markdown(
            "<div class='panel-note'>当前模型直接使用年龄、AFC、FSH、LH、AMH。月经周期与 BMI 暂未进入当前后端模型，仅作病例记录。</div>",
            unsafe_allow_html=True,
        )
        plan_submitted = st.form_submit_button("生成促排卵方案结果", use_container_width=True)
    plan_error_slot = st.empty()
    plan_result_slot = st.empty()
    if st.session_state.plan_error:
        plan_error_slot.markdown(
            metric_cards_component.error_banner_html("促排卵方案生成失败", st.session_state.plan_error),
            unsafe_allow_html=True,
        )
    plan_result_slot.markdown(
        result_panel_components.plan_result_html(st.session_state.plan_report, st.session_state.plan_patient),
        unsafe_allow_html=True,
    )

if reserve_submitted:
    try:
        reserve_amh = parse_numeric_text(
            reserve_amh_text,
            "AMH",
            min_value=0.0,
            default_value=EXAMPLE_DEFAULTS["AMH"],
        )
        reserve_age = parse_numeric_text(
            reserve_age_text,
            "年龄",
            min_value=18,
            max_value=55,
            integer=True,
            default_value=EXAMPLE_DEFAULTS["Age"],
        )
        reserve_fsh = parse_numeric_text(
            reserve_fsh_text,
            "bFSH",
            min_value=0.0,
            default_value=EXAMPLE_DEFAULTS["FSH"],
        )
        reserve_afc = parse_numeric_text(
            reserve_afc_text,
            "AFC",
            min_value=0,
            max_value=60,
            integer=True,
            default_value=EXAMPLE_DEFAULTS["AFC"],
        )
        reserve_cycle = parse_numeric_text(
            reserve_cycle_text,
            "月经周期",
            min_value=18,
            max_value=60,
            integer=True,
            default_value=EXAMPLE_DEFAULTS["cycle_length"],
        )
        reserve_menarche = parse_numeric_text(
            reserve_menarche_text,
            "初潮年龄",
            min_value=8,
            max_value=20,
            integer=True,
            default_value=EXAMPLE_DEFAULTS["menarche_age"],
        )
        update_reserve_state(
            reserve_age,
            reserve_amh,
            reserve_fsh,
            reserve_afc,
            reserve_cycle,
            reserve_menarche,
        )
    except Exception as exc:
        st.session_state.reserve_report = None
        st.session_state.reserve_error = str(exc).strip() or "输入校验失败"


if plan_submitted:
    try:
        plan_age = parse_numeric_text(
            plan_age_text,
            "年龄",
            min_value=18,
            max_value=55,
            integer=True,
            default_value=EXAMPLE_DEFAULTS["Age"],
        )
        plan_cycle = parse_numeric_text(
            plan_cycle_text,
            "月经周期",
            min_value=18,
            max_value=60,
            integer=True,
            default_value=EXAMPLE_DEFAULTS["cycle_length"],
        )
        plan_bmi = parse_numeric_text(
            plan_bmi_text,
            "BMI",
            min_value=0.0,
            default_value=EXAMPLE_DEFAULTS["BMI"],
        )
        plan_afc = parse_numeric_text(
            plan_afc_text,
            "AFC",
            min_value=0,
            max_value=60,
            integer=True,
            default_value=EXAMPLE_DEFAULTS["AFC"],
        )
        plan_fsh = parse_numeric_text(
            plan_fsh_text,
            "FSH",
            min_value=0.0,
            default_value=EXAMPLE_DEFAULTS["FSH"],
        )
        plan_lh = parse_numeric_text(
            plan_lh_text,
            "LH",
            min_value=0.0,
            default_value=EXAMPLE_DEFAULTS["LH"],
        )
        plan_amh = parse_numeric_text(
            plan_amh_text,
            "AMH",
            min_value=0.0,
            default_value=EXAMPLE_DEFAULTS["AMH"],
        )
        update_plan_state(
            plan_age,
            plan_amh,
            plan_fsh,
            plan_afc,
            plan_lh,
            plan_bmi,
            plan_cycle,
        )
    except Exception as exc:
        st.session_state.plan_report = None
        st.session_state.plan_error = str(exc).strip() or "输入校验失败"

if st.session_state.reserve_error:
    reserve_error_slot.markdown(
        metric_cards_component.error_banner_html("卵巢储备功能结果生成失败", st.session_state.reserve_error),
        unsafe_allow_html=True,
    )
else:
    reserve_error_slot.empty()

if st.session_state.plan_error:
    plan_error_slot.markdown(
        metric_cards_component.error_banner_html("促排卵方案生成失败", st.session_state.plan_error),
        unsafe_allow_html=True,
    )
else:
    plan_error_slot.empty()

reserve_result_slot.markdown(
    result_panel_components.reserve_result_html(st.session_state.reserve_report, st.session_state.reserve_patient),
    unsafe_allow_html=True,
)
plan_result_slot.markdown(
    result_panel_components.plan_result_html(st.session_state.plan_report, st.session_state.plan_patient),
    unsafe_allow_html=True,
)


st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

usage_col, traffic_col = st.columns([1, 1], gap="medium")

with usage_col:
    st.markdown(
        metric_cards_component.ranking_card_html(
            "地区使用排名",
            "Regional Usage Ranking",
            [
                ("福州", 3248),
                ("泉州", 2760),
                ("厦门", 2412),
                ("莆田", 1865),
                ("宁德", 1542),
                ("龙岩", 1188),
            ],
            traffic=False,
        ),
        unsafe_allow_html=True,
    )

with traffic_col:
    st.markdown(
        metric_cards_component.ranking_card_html(
            "平台浏览量",
            "Regional Traffic Ranking",
            [
                ("厦门", 23168),
                ("福州", 19640),
                ("泉州", 18425),
                ("漳州", 15980),
                ("莆田", 12860),
                ("宁德", 10942),
            ],
            traffic=True,
        ),
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='page-footer'>平台由福州智视医学科技有限公司提供</div>",
    unsafe_allow_html=True,
)


def main() -> None:
    """Compatibility entrypoint for the root app.py wrapper."""


if __name__ == "__main__":
    main()
