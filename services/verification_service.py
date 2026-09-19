from __future__ import annotations

"""平台注册/登录验证码读取模块。

用途：
- 读取 QQ 邮箱中“短信转发邮件”或平台验证码邮件；
- 提取 4-8 位数字验证码；
- 返回给 Playwright/Selenium/人工接管流程填入；
- 默认不在日志中打印验证码原文。

边界：
- 不破解图形验证码/滑块/人机验证；
- 不绕过平台审核；
- 读取失败时应降级人工输入。
"""

import email
import imaplib
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email.header import decode_header
from email.message import Message
from typing import Any

from utils.pro_console import pro_console_log

CODE_RE = re.compile(r"(?<!\d)(\d{4,8})(?!\d)")
DEFAULT_KEYWORDS = ["验证码", "校验码", "动态码", "verification", "code", "verify", "短信"]


@dataclass
class VerificationCodeResult:
    ok: bool
    code: str | None = None
    source: str = "email_imap"
    message: str = ""
    code_length: int | None = None
    matched_subject: str | None = None
    matched_from: str | None = None
    mailbox: str | None = None
    checked_messages: int = 0
    elapsed_seconds: float = 0.0
    fallback_required: bool = False

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if os.getenv("VERIFICATION_LOG_RAW_CODE", "0") != "1":
            data["code"] = "***" if self.code else None
        return data


def _decode_mime_header(value: str | None) -> str:
    if not value:
        return ""
    parts = []
    for chunk, enc in decode_header(value):
        if isinstance(chunk, bytes):
            for encoding in [enc, "utf-8", "gb18030", "gbk", "latin1"]:
                if not encoding:
                    continue
                try:
                    parts.append(chunk.decode(encoding, errors="ignore"))
                    break
                except Exception:
                    continue
        else:
            parts.append(str(chunk))
    return "".join(parts)


def _message_text(msg: Message) -> str:
    pieces: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            if ctype not in {"text/plain", "text/html"}:
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                text = payload.decode(charset, errors="ignore")
            except Exception:
                text = payload.decode("utf-8", errors="ignore")
            if ctype == "text/html":
                text = re.sub(r"<[^>]+>", " ", text)
            pieces.append(text)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                pieces.append(payload.decode(charset, errors="ignore"))
            except Exception:
                pieces.append(payload.decode("utf-8", errors="ignore"))
        else:
            pieces.append(str(msg.get_payload() or ""))
    return "\n".join(pieces)


def _extract_code(text: str, keywords: list[str]) -> str | None:
    text = text or ""
    # 优先：验证码关键词附近 80 字符窗口内的数字。
    lower_text = text.lower()
    for kw in keywords:
        k = kw.lower()
        start = 0
        while True:
            idx = lower_text.find(k, start)
            if idx < 0:
                break
            window = text[max(0, idx - 80): idx + 160]
            m = CODE_RE.search(window)
            if m:
                return m.group(1)
            start = idx + len(k)
    # 兜底：全文只有一个明显验证码时才取。
    matches = CODE_RE.findall(text)
    uniq = []
    for x in matches:
        if x not in uniq:
            uniq.append(x)
    if len(uniq) == 1:
        return uniq[0]
    return None


def _imap_since_date(lookback_minutes: int) -> str:
    dt = datetime.now() - timedelta(minutes=max(lookback_minutes, 1))
    # IMAP SINCE 只能按天，因此仍会取当天/昨日邮件，再按数量和关键词过滤。
    return dt.strftime("%d-%b-%Y")


def read_code_from_qq_mail_once(
    *,
    mail_address: str | None = None,
    auth_code: str | None = None,
    imap_host: str | None = None,
    imap_port: int | None = None,
    mailbox: str | None = None,
    lookback_minutes: int | None = None,
    max_messages: int | None = None,
    keywords: list[str] | None = None,
) -> VerificationCodeResult:
    t0 = time.time()
    mail_address = mail_address or os.getenv("QQ_MAIL_ADDRESS") or os.getenv("VERIFICATION_EMAIL_ADDRESS")
    auth_code = auth_code or os.getenv("QQ_MAIL_AUTH_CODE") or os.getenv("VERIFICATION_EMAIL_AUTH_CODE")
    imap_host = imap_host or os.getenv("QQ_MAIL_IMAP_HOST", "imap.qq.com")
    imap_port = int(imap_port or os.getenv("QQ_MAIL_IMAP_PORT", "993"))
    mailbox = mailbox or os.getenv("VERIFICATION_EMAIL_MAILBOX", "INBOX")
    lookback_minutes = int(lookback_minutes or os.getenv("SMS_CODE_LOOKBACK_MINUTES", "5"))
    max_messages = int(max_messages or os.getenv("VERIFICATION_EMAIL_MAX_MESSAGES", "30"))
    kw_env = os.getenv("SMS_CODE_ALLOWED_KEYWORDS") or os.getenv("VERIFICATION_CODE_KEYWORDS")
    if keywords is None:
        keywords = [x.strip() for x in kw_env.split(",") if x.strip()] if kw_env else DEFAULT_KEYWORDS

    if not mail_address or not auth_code:
        return VerificationCodeResult(False, message="未配置 QQ_MAIL_ADDRESS 或 QQ_MAIL_AUTH_CODE。", fallback_required=True)

    checked = 0
    try:
        pro_console_log("EMAIL_CODE", "开始连接 QQ 邮箱 IMAP", {"host": imap_host, "port": imap_port, "mailbox": mailbox, "lookback_minutes": lookback_minutes})
        client = imaplib.IMAP4_SSL(imap_host, imap_port)
        client.login(mail_address, auth_code)
        typ, _ = client.select(mailbox)
        if typ != "OK":
            raise RuntimeError(f"无法选择邮箱目录：{mailbox}")
        since = _imap_since_date(lookback_minutes)
        typ, data = client.search(None, "SINCE", since)
        if typ != "OK":
            raise RuntimeError("IMAP SEARCH 失败")
        ids = data[0].split()
        ids = ids[-max_messages:]
        for msg_id in reversed(ids):
            checked += 1
            typ, msg_data = client.fetch(msg_id, "(RFC822)")
            if typ != "OK" or not msg_data:
                continue
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)
            subject = _decode_mime_header(msg.get("Subject"))
            from_ = _decode_mime_header(msg.get("From"))
            body = _message_text(msg)
            combined = f"{subject}\n{from_}\n{body}"
            if keywords and not any(k.lower() in combined.lower() for k in keywords):
                continue
            code = _extract_code(combined, keywords)
            if code:
                try:
                    client.logout()
                except Exception:
                    pass
                res = VerificationCodeResult(
                    True,
                    code=code,
                    message="验证码读取成功。",
                    code_length=len(code),
                    matched_subject=subject[:120],
                    matched_from=from_[:120],
                    mailbox=mailbox,
                    checked_messages=checked,
                    elapsed_seconds=round(time.time() - t0, 2),
                )
                pro_console_log("EMAIL_CODE", "验证码读取成功", res.public_dict())
                return res
        try:
            client.logout()
        except Exception:
            pass
        return VerificationCodeResult(False, message="未在最近邮件中找到验证码。", mailbox=mailbox, checked_messages=checked, elapsed_seconds=round(time.time() - t0, 2), fallback_required=True)
    except Exception as exc:
        return VerificationCodeResult(False, message=f"邮箱验证码读取失败：{exc}", mailbox=mailbox, checked_messages=checked, elapsed_seconds=round(time.time() - t0, 2), fallback_required=True)


def wait_for_verification_code(timeout_seconds: int | None = None, poll_interval_seconds: int | None = None) -> VerificationCodeResult:
    timeout_seconds = int(timeout_seconds or os.getenv("VERIFICATION_TIMEOUT_SECONDS", "120"))
    poll_interval_seconds = int(poll_interval_seconds or os.getenv("VERIFICATION_POLL_INTERVAL_SECONDS", "5"))
    t0 = time.time()
    last: VerificationCodeResult | None = None
    pro_console_log("VERIFY", "开始等待邮箱/短信转发验证码", {"timeout_seconds": timeout_seconds, "poll_interval_seconds": poll_interval_seconds, "mode": os.getenv("VERIFICATION_SMS_MODE", "email_forward")})
    while time.time() - t0 <= timeout_seconds:
        last = read_code_from_qq_mail_once()
        if last.ok:
            return last
        time.sleep(max(poll_interval_seconds, 1))
    msg = "超时未读取到验证码；请切换为人工输入。"
    res = VerificationCodeResult(False, message=msg, checked_messages=(last.checked_messages if last else 0), elapsed_seconds=round(time.time() - t0, 2), fallback_required=True)
    pro_console_log("VERIFY", "验证码自动读取超时，切换人工输入", res.public_dict())
    return res
