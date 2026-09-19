from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

PAYMENT_QR_DIR = Path(r"E:\Agent_DSM\收款码")

_WECHAT_KEYS = ("微信", "wechat", "weixin", "wx")
_ALIPAY_KEYS = ("支付宝", "alipay", "ali")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _find_image(keys: tuple[str, ...]) -> str:
    try:
        if not PAYMENT_QR_DIR.exists():
            return ""
        files = [p for p in PAYMENT_QR_DIR.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
        # Prefer keyword matched names.
        for p in files:
            low = p.name.lower()
            if any(k.lower() in low for k in keys):
                return str(p)
        return str(files[0]) if files else ""
    except Exception:
        return ""


def _to_data_uri(path: str) -> str:
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""


def get_payment_qr_payload() -> dict[str, Any]:
    """Return QR-code image payload for the Pro activation window."""
    wechat_path = _find_image(_WECHAT_KEYS)
    alipay_path = _find_image(_ALIPAY_KEYS)
    # If only one image exists, keep it as WeChat and leave Alipay empty unless its name matched Alipay.
    if wechat_path and alipay_path and Path(wechat_path) == Path(alipay_path):
        low = Path(wechat_path).name.lower()
        if any(k.lower() in low for k in _WECHAT_KEYS) and not any(k.lower() in low for k in _ALIPAY_KEYS):
            alipay_path = ""
        elif any(k.lower() in low for k in _ALIPAY_KEYS) and not any(k.lower() in low for k in _WECHAT_KEYS):
            wechat_path = ""
    return {
        "wechat_path": wechat_path,
        "alipay_path": alipay_path,
        "wechat_src": _to_data_uri(wechat_path),
        "alipay_src": _to_data_uri(alipay_path),
        "qr_dir": str(PAYMENT_QR_DIR),
        "ok": bool(wechat_path or alipay_path),
    }
