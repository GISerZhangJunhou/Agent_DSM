from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
import os
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from config.settings import DATA_DIR

AUTH_DIR = DATA_DIR / "auth"
USERS_PATH = AUTH_DIR / "users.json"
CODES_PATH = AUTH_DIR / "verification_codes.json"
CODE_RESEND_COOLDOWN_SECONDS = int(os.getenv("DSM_CODE_RESEND_COOLDOWN_SECONDS", "30") or "30")

PLAN_STANDARD = "standard"
PLAN_PRO = "pro"
PLAN_LABELS = {
    PLAN_STANDARD: "标准版",
    PLAN_PRO: "Pro 版",
    "free": "标准版",
    "basic": "标准版",
    "vip": "Pro 版",
}
PRO_PRICE_RMB_MONTH = 20
PRO_PERIOD_DAYS = 30
PRO_PERIOD_SECONDS = PRO_PERIOD_DAYS * 24 * 60 * 60


def normalize_plan(plan: str | None) -> str:
    value = _norm_key(plan)
    if value in {"pro", "vip"}:
        return PLAN_PRO
    return PLAN_STANDARD


def plan_label(plan: str | None) -> str:
    return PLAN_LABELS.get(normalize_plan(plan), "标准版")


def is_pro_plan(plan: str | None) -> bool:
    return normalize_plan(plan) == PLAN_PRO


def _is_pro_not_expired(user: dict[str, Any] | None) -> bool:
    if not user or not is_pro_plan(user.get("plan")):
        return False
    expires_at = int(user.get("pro_expires_at") or 0)
    # 兼容 v54 之前已经升级但没有过期时间的本地账号：默认视为有效。
    if expires_at <= 0:
        return True
    return expires_at >= int(time.time())


def is_pro_user(user: dict[str, Any] | None) -> bool:
    return _is_pro_not_expired(user)


def _ensure_store() -> None:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_PATH.exists():
        USERS_PATH.write_text(json.dumps({"users": []}, ensure_ascii=False, indent=2), encoding="utf-8")


def _load() -> dict[str, Any]:
    _ensure_store()
    try:
        data = json.loads(USERS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"users": []}
        data.setdefault("users", [])
        return data
    except Exception:
        return {"users": []}


def _save(data: dict[str, Any]) -> None:
    _ensure_store()
    tmp = USERS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(USERS_PATH)


def _norm(value: str | None) -> str:
    return (value or "").strip()


def _norm_key(value: str | None) -> str:
    return _norm(value).lower()


def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 160000)
    return f"pbkdf2_sha256${salt}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt, expected = stored.split("$", 2)
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 160000).hex()
        return hmac.compare_digest(digest, expected)
    except Exception:
        return False


def _looks_like_email(value: str) -> bool:
    value = _norm(value)
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value))


def _looks_like_contact(value: str) -> bool:
    # 正式交付版当前只开放邮箱验证码。手机号注册/登录不启用。
    return _looks_like_email(value)


def find_user_by_username(username: str) -> dict[str, Any] | None:
    key = _norm_key(username)
    for user in _load().get("users", []):
        if _norm_key(user.get("username")) == key:
            return user
    return None


def find_user_by_contact(contact: str) -> dict[str, Any] | None:
    key = _norm_key(contact)
    for user in _load().get("users", []):
        if _norm_key(user.get("contact")) == key:
            return user
    return None


def find_user_by_login(login_name: str) -> dict[str, Any] | None:
    return find_user_by_username(login_name) or find_user_by_contact(login_name)



def _load_codes() -> dict[str, Any]:
    _ensure_store()
    if not CODES_PATH.exists():
        return {"codes": {}}
    try:
        data = json.loads(CODES_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"codes": {}}
        data.setdefault("codes", {})
        return data
    except Exception:
        return {"codes": {}}


def _save_codes(data: dict[str, Any]) -> None:
    _ensure_store()
    tmp = CODES_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CODES_PATH)


def _check_code_resend_cooldown(contact: str, purpose: str | None = None) -> tuple[bool, int, str]:
    """Return whether a new verification code may be sent now.

    The cooldown is enforced server-side so refreshing the page or repeatedly
    clicking the button cannot bypass it.  It is keyed by normalized email
    address and, when available, by purpose.
    """
    try:
        cooldown = max(0, int(CODE_RESEND_COOLDOWN_SECONDS))
    except Exception:
        cooldown = 30
    if cooldown <= 0:
        return True, 0, ""
    data = _load_codes()
    rec = (data.get("codes") or {}).get(_norm_key(contact))
    if not isinstance(rec, dict):
        return True, 0, ""
    if purpose and rec.get("purpose") not in {purpose, "any", None}:
        return True, 0, ""
    created_at = int(rec.get("created_at") or 0)
    elapsed = int(time.time()) - created_at
    remain = cooldown - elapsed
    if remain > 0:
        return False, remain, f"验证码已发送，请 {remain} 秒后再获取。"
    return True, 0, ""


def _send_email_code(contact: str, code: str) -> tuple[bool, str]:
    """Send verification code through configured SMTP.

    Required environment variables:
    DSM_SMTP_HOST, DSM_SMTP_PORT, DSM_SMTP_USER, DSM_SMTP_PASSWORD.
    Optional: DSM_SMTP_FROM, DSM_SMTP_SSL.
    """
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", contact):
        return False, "当前仅支持邮箱验证码。请填写可接收邮件的邮箱地址。"
    host = os.getenv("DSM_SMTP_HOST", "").strip()
    user = os.getenv("DSM_SMTP_USER", "").strip()
    pwd = os.getenv("DSM_SMTP_PASSWORD", "").strip()
    if not host or not user or not pwd:
        return False, "未配置 SMTP 邮件服务。请在 .env 中配置 DSM_SMTP_HOST、DSM_SMTP_USER、DSM_SMTP_PASSWORD。"
    port = int(os.getenv("DSM_SMTP_PORT", "465") or "465")
    sender = os.getenv("DSM_SMTP_FROM", user).strip() or user
    use_ssl = os.getenv("DSM_SMTP_SSL", "1").strip() != "0"
    msg = MIMEText(f"你的数字土壤制图智能体验证码为：{code}。验证码 10 分钟内有效。", "plain", "utf-8")
    msg["Subject"] = "数字土壤制图智能体验证码"
    msg["From"] = sender
    msg["To"] = contact
    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=20) as s:
                s.login(user, pwd)
                s.sendmail(sender, [contact], msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.starttls()
                s.login(user, pwd)
                s.sendmail(sender, [contact], msg.as_string())
        return True, "验证码已发送至邮箱，请查收。"
    except Exception as exc:
        return False, f"验证码发送失败：{exc}"


def _store_verification_code(contact: str, code: str, purpose: str) -> None:
    data = _load_codes()
    key = _norm_key(contact)
    data.setdefault("codes", {})[key] = {
        "code_hash": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "purpose": purpose,
        "created_at": int(time.time()),
        "expires_at": int(time.time()) + 10 * 60,
        "used": False,
    }
    _save_codes(data)


def _verify_code(contact: str, code: str, purpose: str) -> tuple[bool, str]:
    contact = _norm(contact)
    code = _norm(code)
    data = _load_codes()
    rec = (data.get("codes") or {}).get(_norm_key(contact))
    if not rec:
        return False, "请先获取验证码。"
    if rec.get("used"):
        return False, "验证码已使用，请重新获取。"
    if int(rec.get("expires_at") or 0) < int(time.time()):
        return False, "验证码已过期，请重新获取。"
    if rec.get("purpose") not in {purpose, "any"}:
        return False, "验证码用途不匹配，请重新获取。"
    if hashlib.sha256(code.encode("utf-8")).hexdigest() != rec.get("code_hash"):
        return False, "验证码不正确。"
    rec["used"] = True
    rec["used_at"] = int(time.time())
    data.setdefault("codes", {})[_norm_key(contact)] = rec
    _save_codes(data)
    return True, "验证码校验通过。"


def register_user(username: str, contact: str, password: str, confirm_password: str, code: str, agreed: bool) -> dict[str, Any]:
    username = _norm(username)
    contact = _norm(contact)
    code = _norm(code)
    password = password or ""
    confirm_password = confirm_password or ""

    if not contact:
        return {"ok": False, "error": "请填写邮箱地址。"}
    if not _looks_like_contact(contact):
        return {"ok": False, "error": "邮箱格式不正确，请填写可接收邮件的邮箱地址。"}
    if not code or not re.match(r"^\d{4}$", code):
        return {"ok": False, "error": "请输入 4 位验证码。请输入验证码。"}
    ok_code, msg_code = _verify_code(contact, code, "register")
    if not ok_code:
        return {"ok": False, "error": msg_code}
    if not username:
        return {"ok": False, "error": "请填写用户名。"}
    if len(username) < 2:
        return {"ok": False, "error": "用户名至少需要 2 个字符。"}
    if len(password) < 6:
        return {"ok": False, "error": "密码至少需要 6 位。"}
    if password != confirm_password:
        return {"ok": False, "error": "两次密码不一致。"}
    if not agreed:
        return {"ok": False, "error": "请先阅读并同意用户协议。"}

    data = _load()
    for user in data.get("users", []):
        if _norm_key(user.get("username")) == _norm_key(username):
            return {"ok": False, "error": "用户名已重复，请修改。", "reason": "username_exists"}
        if _norm_key(user.get("contact")) == _norm_key(contact):
            return {"ok": False, "error": "该邮箱已注册，是否需要找回密码？", "reason": "contact_exists"}

    now = int(time.time())
    user = {
        "id": secrets.token_hex(8),
        "username": username,
        "contact": contact,
        "password_hash": _hash_password(password),
        "plan": PLAN_STANDARD,
        "plan_label": plan_label(PLAN_STANDARD),
        "created_at": now,
        "last_login_at": None,
        "is_active": True,
        "pro_paid_at": None,
        "pro_expires_at": None,
    }
    data.setdefault("users", []).append(user)
    _save(data)
    return {"ok": True, "user": public_user(user), "message": "注册成功，请使用新账号登录。"}


def authenticate(login_name: str, password: str) -> dict[str, Any]:
    login_name = _norm(login_name)
    if not login_name or not password:
        return {"ok": False, "error": "请输入用户名或邮箱和密码。"}
    data = _load()
    user_index = None
    user = None
    for i, item in enumerate(data.get("users", [])):
        if _norm_key(item.get("username")) == _norm_key(login_name) or _norm_key(item.get("contact")) == _norm_key(login_name):
            user_index = i
            user = item
            break
    if not user or not _verify_password(password, user.get("password_hash", "")):
        return {"ok": False, "error": "用户名或者密码错误。"}
    if not user.get("is_active", True):
        return {"ok": False, "error": "该账号已被停用。"}
    # 如果 Pro 到期，自动回落为标准版，避免前端误判。
    if is_pro_plan(user.get("plan")) and not _is_pro_not_expired(user):
        user["plan"] = PLAN_STANDARD
        user["plan_label"] = plan_label(PLAN_STANDARD)
        user["pro_expired_at"] = int(time.time())
    user["last_login_at"] = int(time.time())
    data["users"][user_index] = user
    _save(data)
    return {"ok": True, "user": public_user(user), "message": "登录成功。"}


def reset_password(contact: str, code: str, password: str, confirm_password: str) -> dict[str, Any]:
    contact = _norm(contact)
    code = _norm(code)
    password = password or ""
    confirm_password = confirm_password or ""
    if not contact:
        return {"ok": False, "error": "请填写注册时使用的邮箱。"}
    if not code or not re.match(r"^\d{4}$", code):
        return {"ok": False, "error": "请输入 4 位验证码。请输入验证码。"}
    ok_code, msg_code = _verify_code(contact, code, "reset")
    if not ok_code:
        return {"ok": False, "error": msg_code}
    if len(password) < 6:
        return {"ok": False, "error": "新密码至少需要 6 位。"}
    if password != confirm_password:
        return {"ok": False, "error": "两次密码不一致。"}
    data = _load()
    for i, user in enumerate(data.get("users", [])):
        if _norm_key(user.get("contact")) == _norm_key(contact):
            user["password_hash"] = _hash_password(password)
            user["password_reset_at"] = int(time.time())
            data["users"][i] = user
            _save(data)
            return {"ok": True, "user": public_user(user), "message": "密码已重置，请使用新密码登录。"}
    return {"ok": False, "error": "未找到使用该邮箱注册的账号。"}


def _pro_remaining_days(user: dict[str, Any]) -> int | None:
    expires_at = int(user.get("pro_expires_at") or 0)
    if not is_pro_plan(user.get("plan")):
        return None
    if expires_at <= 0:
        return None
    remain = max(0, expires_at - int(time.time()))
    return int((remain + 24 * 60 * 60 - 1) // (24 * 60 * 60))


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    plan = normalize_plan(user.get("plan"))
    pro_active = _is_pro_not_expired(user)
    if plan == PLAN_PRO and not pro_active:
        plan = PLAN_STANDARD
    return {
        "id": user.get("id"),
        "username": user.get("username"),
        "contact": user.get("contact"),
        "plan": plan,
        "plan_label": plan_label(plan),
        "last_login_at": user.get("last_login_at"),
        "pro_price_rmb_month": PRO_PRICE_RMB_MONTH,
        "pro_period_days": PRO_PERIOD_DAYS,
        "pro_paid_at": user.get("pro_paid_at"),
        "pro_expires_at": user.get("pro_expires_at"),
        "pro_remaining_days": _pro_remaining_days(user),
        "is_pro_active": pro_active,
    }


def get_user_plan_status(user_id: str | None) -> dict[str, Any]:
    user_id = _norm(user_id)
    if not user_id:
        return {"ok": False, "error": "缺少用户 ID。"}
    for user in _load().get("users", []):
        if _norm(user.get("id")) == user_id:
            return {"ok": True, "user": public_user(user)}
    return {"ok": False, "error": "未找到该用户。"}


def upgrade_user_to_pro(user_id: str | None) -> dict[str, Any]:
    """Pro 开通成功后把账号升级为 Pro；重复点击等价于续费 30 天。"""
    user_id = _norm(user_id)
    if not user_id:
        return {"ok": False, "error": "无法识别当前登录账号，请重新登录。"}
    data = _load()
    now = int(time.time())
    for i, user in enumerate(data.get("users", [])):
        if _norm(user.get("id")) == user_id:
            old_exp = int(user.get("pro_expires_at") or 0)
            base = max(now, old_exp)
            user["plan"] = PLAN_PRO
            user["plan_label"] = plan_label(PLAN_PRO)
            user["pro_paid_at"] = now
            user["pro_expires_at"] = base + PRO_PERIOD_SECONDS
            user["pro_price_rmb_month"] = PRO_PRICE_RMB_MONTH
            user.setdefault("pro_payment_history", []).append({
                "paid_at": now,
                "amount_rmb": PRO_PRICE_RMB_MONTH,
                "period_days": PRO_PERIOD_DAYS,
                "mode": "pro_activation",
            })
            data["users"][i] = user
            _save(data)
            return {
                "ok": True,
                "user": public_user(user),
                "message": f"Pro 开通成功，账号已开通 Pro 版，有效期 {PRO_PERIOD_DAYS} 天。",
            }
    return {"ok": False, "error": "未找到当前账号，升级失败。"}


def simulate_send_code(contact: str, require_registered: bool = False) -> dict[str, Any]:
    contact = _norm(contact)
    if not contact:
        return {"ok": False, "error": "请先填写邮箱地址。"}
    if not _looks_like_contact(contact):
        return {"ok": False, "error": "邮箱格式不正确，请填写可接收邮件的邮箱地址。"}
    if require_registered and not find_user_by_contact(contact):
        return {"ok": False, "error": "该邮箱尚未注册。"}
    if (not require_registered) and find_user_by_contact(contact):
        return {"ok": False, "error": "该邮箱已注册。"}
    purpose = "reset" if require_registered else "register"
    can_send, remain, cooldown_msg = _check_code_resend_cooldown(contact, purpose)
    if not can_send:
        return {"ok": False, "error": cooldown_msg, "cooldown_remaining_seconds": remain}
    code = f"{secrets.randbelow(10000):04d}"

    sent, message = _send_email_code(contact, code)

    if not sent:
        # 不自动通过验证码。邮箱验证码必须通过已配置的 SMTP 服务真实发送。
        return {"ok": False, "error": message}
    _store_verification_code(contact, code, purpose)
    return {"ok": True, "message": message}
