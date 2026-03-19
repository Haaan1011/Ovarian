import base64
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from ovarian_prediction.config import BACKGROUND_IMAGES_DIR, BRAND_IMAGES_DIR


TITLE_LOGO_PATH = BRAND_IMAGES_DIR / "fuyou_logo_fullseal.png"
RAW_TITLE_LOGO_PATH = BRAND_IMAGES_DIR / "fuyou_logo_raw.png"


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
        BACKGROUND_IMAGES_DIR / "fuyou.png",
        BACKGROUND_IMAGES_DIR / "background.jpg",
        BACKGROUND_IMAGES_DIR / "background.png",
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
