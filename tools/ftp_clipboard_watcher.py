from __future__ import annotations

"""
V151 passive TPDC FTP account watcher + large-file downloader.

Design boundary:
- It may read clipboard text.
- It may read the current Chrome page through CDP/Playwright evaluate().
- It must never click, fill, press keys, reload, navigate, close a page, or close Chrome.
- When TPDC displays an FTP account block after the user clicks a download/FTP button,
  the watcher extracts host/port/username/password, writes handoff files, and optionally
  starts a resumable Python FTP download.
"""

import argparse
import base64
import ftplib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote


def _is_truth(v: object, default: str = "0") -> bool:
    s = str(v if v is not None and str(v) != "" else default).strip().lower()
    return s in {"1", "true", "yes", "on", "y"}


def _safe_name(name: str, max_len: int = 160) -> str:
    name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(name or "download"))
    return (name.strip("._-") or "download")[:max_len]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_json(path: Path, data: dict) -> None:
    """Write JSON with Windows-safe retry.

    Dash/UI polling and the FTP watcher may touch state.json at the same time on
    Windows.  A single tmp.replace() can raise WinError 32 if another process is
    reading the file.  Retrying prevents a successful FTP capture from crashing
    merely because the progress panel is polling.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    last_exc = None
    for i in range(10):
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{i}.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_exc = exc
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            time.sleep(0.08 * (i + 1))
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise
    # Last-resort non-atomic write.  This is preferable to crashing the watcher;
    # malformed reads are already guarded in _update_state.
    try:
        path.write_text(payload, encoding="utf-8")
        return
    except Exception:
        if last_exc:
            raise last_exc
        raise


def _mask_secret(v: str | None) -> str:
    s = str(v or "")
    if not s:
        return ""
    if len(s) <= 2:
        return "*" * len(s)
    return s[:1] + "*" * max(2, len(s) - 2) + s[-1:]


def _redact_accounts(accounts: list[dict]) -> list[dict]:
    out = []
    for a in accounts or []:
        b = dict(a)
        if "password" in b:
            b["password_masked"] = _mask_secret(b.get("password"))
            b.pop("password", None)
        out.append(b)
    return out




def _debug_print_ftp_accounts(accounts: list[dict], source: str = "") -> None:
    """Print captured FTP credentials to the backend console for local debugging.

    The user explicitly requested host/port/username/password to be visible in
    the local Python console to diagnose TPDC download failures.  This function
    must never be called from front-end rendering code.
    """
    try:
        print(f"[TPDC_FTP_DEBUG] source={source} account_count={len(accounts or [])}", flush=True)
        for i, a in enumerate(accounts or [], 1):
            print(
                "[TPDC_FTP_DEBUG] "
                f"#{i} host={a.get('host')} port={a.get('port')} "
                f"username={a.get('username')} password={a.get('password')}",
                flush=True,
            )
    except Exception as exc:
        print(f"[TPDC_FTP_DEBUG] print_failed: {exc}", flush=True)


def _is_pid_alive(pid: int | str | None) -> bool:
    """Best-effort local process liveness check without extra dependencies."""
    try:
        pid_i = int(pid or 0)
        if pid_i <= 0:
            return False
        if os.name == "nt":
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid_i}", "/NH"],
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=3,
            )
            return str(pid_i) in out
        os.kill(pid_i, 0)
        return True
    except Exception:
        return False


def _download_status_is_final(status: str | None) -> bool:
    st = str(status or "").lower()
    return st in {"ftp_download_completed", "ftp_download_partial", "ftp_download_failed", "ftp_download_process_failed_to_start", "ftp_account_waiting_timeout"} or st.endswith("failed") or st.endswith("completed")


def _read_json_safe(path: Path) -> dict:
    try:
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}
    return {}


def _try_create_downloader_lock(download_dir: Path) -> tuple[bool, Path, dict]:
    """Ensure one TPDC FTP downloader per task directory.

    The watcher and an external capture process can both notice the same FTP
    ticket.  Without a directory-level lock they may start two identical Python
    downloaders, which explains duplicate `downloader_try_primary_1` and
    duplicate backend downloader logs.  This lock is intentionally local to
    the download directory.
    """
    lock = download_dir / ".ftp_downloader.lock.json"
    status = _read_json_safe(download_dir / "ftp_download_status.json")
    old = _read_json_safe(lock)
    pid = old.get("pid")
    if old and _is_pid_alive(pid) and not _download_status_is_final(status.get("status")):
        return False, lock, old
    payload = {"pid": os.getpid(), "created_at": _now(), "status": "lock_acquired"}
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True, lock, payload
    except FileExistsError:
        old = _read_json_safe(lock)
        if old and _is_pid_alive(old.get("pid")) and not _download_status_is_final(status.get("status")):
            return False, lock, old
        try:
            _write_json(lock, payload)
            return True, lock, payload
        except Exception:
            return False, lock, old or {"status": "lock_exists"}
    except Exception as exc:
        # Do not block a valid download solely because a lock cannot be written,
        # but record the condition for diagnostics.
        return True, lock, {"status": "lock_write_failed_but_continue", "error": str(exc)}


def _release_downloader_lock(download_dir: Path) -> None:
    try:
        (download_dir / ".ftp_downloader.lock.json").unlink(missing_ok=True)
    except Exception:
        pass


def _download_process_active_or_started(download_dir: Path) -> tuple[bool, dict]:
    """Return True if this task already has a downloader process or active status.

    This parent-side guard prevents the clipboard watcher and the CDP page
    monitor from spawning separate Python FTP downloaders for the same TPDC
    ticket.  It is intentionally checked before starting a child process; the
    child also keeps its own transfer lock.
    """
    status = _read_json_safe(download_dir / "ftp_download_status.json")
    if status and not _download_status_is_final(status.get("status")):
        st = str(status.get("status") or "").lower()
        if st.startswith("ftp_download") or st in {"ftp_connecting", "ftp_downloading", "ftp_host_failed"}:
            return True, {"source": "ftp_download_status", "status": status}
    spawn_lock = _read_json_safe(download_dir / ".ftp_downloader.spawn.lock.json")
    pid = spawn_lock.get("pid")
    if spawn_lock and _is_pid_alive(pid):
        return True, {"source": "spawn_lock", "lock": spawn_lock}
    child_lock = _read_json_safe(download_dir / ".ftp_downloader.lock.json")
    if child_lock and _is_pid_alive(child_lock.get("pid")):
        return True, {"source": "child_lock", "lock": child_lock}
    return False, {}


def _claim_download_spawn(download_dir: Path, source: str = "watcher") -> tuple[bool, dict]:
    active, info = _download_process_active_or_started(download_dir)
    if active:
        return False, info
    lock = download_dir / ".ftp_downloader.spawn.lock.json"
    payload = {"pid": os.getpid(), "created_at": _now(), "source": source, "status": "spawn_claimed"}
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True, payload
    except FileExistsError:
        active, info = _download_process_active_or_started(download_dir)
        if active:
            return False, info
        try:
            _write_json(lock, payload)
            return True, payload
        except Exception as exc:
            return False, {"source": "spawn_lock", "error": str(exc)}
    except Exception as exc:
        # Continue only if there is no evidence of another active downloader.
        active, info = _download_process_active_or_started(download_dir)
        if active:
            return False, info
        return True, {"status": "spawn_lock_failed_but_continue", "error": str(exc)}


def _write_ftp_ticket_files(download_dir: Path, accounts: list[dict], source: str = "readonly_cdp_or_clipboard") -> dict:
    accounts = _tag_primary_backup_accounts(accounts)
    primary = accounts[0] if accounts else {}
    bits = [str(primary.get("host") or "ftp"), str(primary.get("username") or "user"), str(int(time.time()))]
    ticket_id = "tpdc_" + _safe_name("_".join(bits), 100)
    ticket_dir = download_dir / "ftp_tickets"
    ticket_dir.mkdir(parents=True, exist_ok=True)
    public_ticket = {
        "ticket_id": ticket_id,
        "platform": "国家青藏高原科学数据中心（TPDC）",
        "ticket_scope": "dataset_level_ftp_ticket",
        "source": source,
        "status": "ticket_created",
        "local_save_dir": str(download_dir),
        "accounts_count": len(accounts),
        "accounts": _redact_accounts(accounts),
        "password_policy": "public ticket masks FTP password; local secret file is machine-local and must not be shared",
        "created_at": _now(),
    }
    secret_ticket = dict(public_ticket)
    secret_ticket["accounts"] = accounts
    secret_ticket["password_policy"] = "contains full TPDC dataset FTP password; keep local only"
    public_path = ticket_dir / f"{ticket_id}.json"
    secret_path = ticket_dir / f"{ticket_id}.secret.json"
    latest_path = ticket_dir / "ftp_ticket_latest.json"
    _write_json(public_path, public_ticket)
    _write_json(secret_path, secret_ticket)
    _write_json(latest_path, public_ticket)
    return {
        "ftp_ticket_id": ticket_id,
        "ftp_ticket_path": str(public_path),
        "ftp_ticket_secret_path": str(secret_path),
        "ftp_ticket_latest_path": str(latest_path),
        "ftp_ticket_dir": str(ticket_dir),
    }


def _append_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": _now()}
    rec.update(record)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _update_state(state_json: Path | None, **kwargs) -> None:
    if not state_json:
        return
    data = {}
    try:
        if state_json.exists():
            data = json.loads(state_json.read_text(encoding="utf-8") or "{}")
    except Exception:
        data = {}
    data.update(kwargs)
    data["ftp_watcher_updated_at"] = _now()
    _write_json(state_json, data)


def _tpdc_fixed_ftp_hosts() -> tuple[str, str, int]:
    """Return the single fixed TPDC FTP host used by this agent.

    The current TPDC workflow is intentionally single-host: only the Chinese
    labels ``用户名`` and ``密码`` are captured from the page; host and port are
    not parsed from the DOM.  To avoid duplicate tickets, stale fallback hosts,
    and double downloader processes, the agent uses only ftp2.tpdc.ac.cn:6201.
    """
    primary = os.getenv("TPDC_FTP_PRIMARY_HOST", "ftp2.tpdc.ac.cn").strip() or "ftp2.tpdc.ac.cn"
    # Backup host is deliberately disabled in V232. Keep the return shape for
    # compatibility with older call sites, but return an empty backup string.
    backup = ""
    try:
        port = int(os.getenv("TPDC_FTP_PORT", "6201") or "6201")
    except Exception:
        port = 6201
    return primary, backup, port




_INVALID_TPDC_CREDENTIAL_TOKENS = {
    "", "icon", "select", "-select", "copy", "button", "btn", "password", "pass", "username", "user",
    "account", "download", "placeholder", "input", "text", "radio", "checkbox", "true", "false", "none", "null"
}

def _is_valid_tpdc_username(value: str) -> bool:
    v = str(value or "").strip()
    return bool(re.fullmatch(r"download_[0-9A-Za-z_\-]{4,64}", v))

def _is_valid_tpdc_password(value: str) -> bool:
    v = str(value or "").strip()
    if v.lower() in _INVALID_TPDC_CREDENTIAL_TOKENS:
        return False
    # TPDC temporary passwords observed in this workflow are compact numeric or
    # alphanumeric tokens.  Reject UI artefacts such as icon/-select.
    return bool(re.fullmatch(r"[0-9A-Za-z_@.\-]{6,64}", v))

def _clean_tpdc_password(value: str) -> str:
    return str(value or "").strip().strip("：:;；，,。[](){}<>\\\"'")

def _extract_ftp_accounts_from_text(text: str) -> list[dict]:
    """Extract TPDC FTP ticket by Chinese labels only.

    The TPDC page used here is Chinese.  To avoid DOM noise and stale English UI
    attributes (for example ``username=-select`` or ``password=icon``), this
    parser deliberately recognizes only the Chinese labels ``用户名`` and ``密码``.
    It does not capture host or port from the page; the single fixed TPDC host
    is constructed after one valid username/password ticket is found.
    """
    raw = str(text or "")
    if not raw.strip():
        return []
    # Keep line boundaries for label-neighbour parsing, but remove copy-button noise.
    line_body = re.sub(r"(?:复制|拷贝)", " ", raw)
    line_body = re.sub(r"[\u3000\t\r]+", " ", line_body)
    compact = re.sub(r"\s+", " ", line_body).strip()

    # Hard gate: only parse blocks that visibly contain the Chinese labels.
    # Do not use English labels, generic account/user fields, or bare download_*.
    if not re.search(r"用\s*户\s*名", compact) or not re.search(r"密\s*码", compact):
        return []

    primary_host, _backup_host, fixed_port = _tpdc_fixed_ftp_hosts()
    hosts: list[str] = [primary_host]

    def clean_token(value: str) -> str:
        return str(value or "").strip().strip("：:;；，,。[](){}<>\\\"'")

    def first_match(patterns: list[str], body: str) -> str:
        for pat in patterns:
            m = re.search(pat, body, flags=re.S)
            if m:
                return clean_token(m.group(1))
        return ""

    username = first_match([
        r"用\s*户\s*名\s*[:：]?\s*(download_[0-9A-Za-z_\-]{4,64})",
    ], compact)
    password = _clean_tpdc_password(first_match([
        r"密\s*码\s*[:：]?\s*([0-9A-Za-z_@.\-]{6,64})",
    ], compact))

    # Vue/Element UI may render label and value in separate sibling spans/lines.
    # Search only around Chinese label lines; never scan the full DOM for bare tokens.
    lines = [x.strip() for x in re.split(r"[\n]+", line_body) if x and x.strip()]
    if not _is_valid_tpdc_username(username):
        for i, line in enumerate(lines):
            if re.search(r"用\s*户\s*名", line):
                window = " ".join(lines[i:i + 5])
                candidate = ""
                if not candidate:
                    m = re.search(r"\b(download_[0-9A-Za-z_\-]{4,64})\b", window)
                    candidate = clean_token(m.group(1)) if m else ""
                if _is_valid_tpdc_username(candidate):
                    username = candidate
                    break
    if not _is_valid_tpdc_password(password):
        for i, line in enumerate(lines):
            if re.search(r"密\s*码", line):
                window = " ".join(lines[i:i + 5])
                candidate = _clean_tpdc_password(first_match([
                    r"密\s*码\s*[:：]?\s*([0-9A-Za-z_@.\-]{6,64})",
                ], window))
                if not candidate:
                    # Value may be in the next span/line after the Chinese label.
                    tokens = re.findall(r"\b[0-9A-Za-z_@.\-]{6,64}\b", window)
                    for tok in tokens:
                        if tok == username or tok.startswith("download_"):
                            continue
                        candidate = _clean_tpdc_password(tok)
                        break
                if _is_valid_tpdc_password(candidate):
                    password = candidate
                    break

    if not _is_valid_tpdc_username(username) or not _is_valid_tpdc_password(password):
        return []

    accounts = []
    for idx, host in enumerate(hosts):
        accounts.append({
            "host": host,
            "port": int(fixed_port or 6201),
            "username": username,
            "password": password,
            "priority": idx + 1,
            "primary": idx == 0,
            "host_role": "primary",
            "source": "tpdc_chinese_username_password_labels_only",
            "remote_dir": "",
            "ticket_count": 1,
            "fixed_host_policy": "tpdc_single_primary_fixed_host",
            "credential_parse_policy": "chinese_labels_only",
        })
    return accounts

def _tag_primary_backup_accounts(accounts: list[dict]) -> list[dict]:
    """Compatibility wrapper: keep only the single primary TPDC host."""
    tagged = []
    for i, acc in enumerate(_dedup_accounts(accounts or [])[:1], 1):
        item = dict(acc or {})
        item["host_role"] = "primary"
        item["host_attempt_index"] = 1
        item["host_total"] = 1
        tagged.append(item)
    return tagged


def _dedup_accounts(accounts: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for a in accounts:
        key = (str(a.get("host") or "").lower(), int(a.get("port") or 21), str(a.get("username") or ""), str(a.get("password") or ""))
        if not key[0] or not key[2] or not key[3] or key in seen:
            continue
        seen.add(key)
        b = dict(a)
        b["priority"] = len(out) + 1
        b["primary"] = len(out) == 0
        out.append(b)
    return out


def _clipboard_text() -> str:
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=3,
            )
            return out or ""
        except Exception:
            pass
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        try:
            return root.clipboard_get() or ""
        finally:
            root.destroy()
    except Exception:
        return ""


def _atomic_page_screenshot(page, path: Path) -> bool:
    """Capture a TPDC tab screenshot without navigating or focusing the page."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        page.screenshot(path=str(tmp), full_page=False)
        tmp.replace(path)
        return True
    except Exception:
        return False




def _cdp_page_screenshot(page, path: Path) -> tuple[bool, str]:
    """Fallback screenshot using raw Chrome DevTools Protocol.

    Playwright page.screenshot can fail on externally attached Chrome tabs.
    CDP Page.captureScreenshot is often more tolerant and does not navigate,
    click, or change the page.
    """
    try:
        session = page.context.new_cdp_session(page)
        try:
            session.send("Page.enable")
        except Exception:
            pass
        data = session.send("Page.captureScreenshot", {"format": "png", "fromSurface": True})
        raw = base64.b64decode(data.get("data") or b"")
        if not raw:
            return False, "empty_cdp_screenshot"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
        return True, "cdp_page_screenshot"
    except Exception as exc:
        return False, f"cdp_page_screenshot_failed:{exc}"

def _window_screenshot_by_pid(pid: str | int | None, path: Path) -> tuple[bool, str]:
    """Best-effort Windows whole-browser screenshot.

    This captures the Chrome top-level window including tab bar/address bar when
    Pillow's ImageGrab is available. It never controls the page. If unavailable,
    callers fall back to the best TPDC tab screenshot.
    """
    if os.name != "nt" or not pid:
        return False, "not_windows_or_no_pid"
    try:
        import ctypes
        from ctypes import wintypes
        try:
            from PIL import ImageGrab  # type: ignore
        except Exception as exc:
            return False, f"pillow_unavailable:{exc}"
        target = int(pid)
        user32 = ctypes.windll.user32
        hwnds = []
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def enum_proc(hwnd, lparam):
            try:
                if not user32.IsWindowVisible(hwnd):
                    return True
                proc_id = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
                if int(proc_id.value) == target:
                    length = user32.GetWindowTextLengthW(hwnd)
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value or ""
                    if title:
                        hwnds.append((hwnd, title))
            except Exception:
                pass
            return True
        user32.EnumWindows(EnumWindowsProc(enum_proc), 0)
        if not hwnds:
            return False, "no_window_for_pid"
        # Prefer a TPDC/Chrome window title if present.
        hwnd = next((h for h,t in hwnds if "tpdc" in t.lower() or "chrome" in t.lower() or "青藏" in t), hwnds[0][0])
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return False, "GetWindowRect_failed"
        bbox = (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return False, "empty_window_rect"
        img = ImageGrab.grab(bbox=bbox)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        img.save(tmp)
        tmp.replace(path)
        return True, "window_screenshot"
    except Exception as exc:
        return False, f"window_screenshot_failed:{exc}"


def _window_screenshot_by_title(path: Path) -> tuple[bool, str]:
    """Best-effort Windows screenshot of the foreground/TPDC/Chrome top-level window.

    V142: the browser PID returned by subprocess may be only a launcher process;
    Chrome's visible top-level window can belong to a different child process.
    Therefore PID-based lookup often fails and leaves a stale screenshot. This
    fallback enumerates visible windows by title and captures the best TPDC/Chrome
    candidate. It never clicks, navigates, closes, or focuses a page.
    """
    if os.name != "nt":
        return False, "not_windows"
    try:
        import ctypes
        from ctypes import wintypes
        try:
            from PIL import ImageGrab  # type: ignore
        except Exception as exc:
            return False, f"pillow_unavailable:{exc}"
        user32 = ctypes.windll.user32
        candidates = []
        foreground = user32.GetForegroundWindow()
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def enum_proc(hwnd, lparam):
            try:
                if not user32.IsWindowVisible(hwnd):
                    return True
                length = user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return True
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value or ""
                tl = title.lower()
                if not any(k in tl for k in ["tpdc", "青藏", "chrome", "google chrome", "国家青藏高原科学数据"]):
                    return True
                rect = wintypes.RECT()
                if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    return True
                w = int(rect.right - rect.left); h = int(rect.bottom - rect.top)
                if w < 300 or h < 200:
                    return True
                score = 0
                if hwnd == foreground:
                    score += 10000
                if "tpdc" in tl or "青藏" in title or "国家青藏高原" in title:
                    score += 5000
                if "chrome" in tl:
                    score += 1000
                score += min(w * h // 100000, 100)
                candidates.append((score, hwnd, title, (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))))
            except Exception:
                pass
            return True
        user32.EnumWindows(EnumWindowsProc(enum_proc), 0)
        if not candidates:
            return False, "no_title_window"
        score, hwnd, title, bbox = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
        img = ImageGrab.grab(bbox=bbox)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        img.save(tmp)
        tmp.replace(path)
        return True, f"window_title_screenshot:{title}"
    except Exception as exc:
        return False, f"window_title_screenshot_failed:{exc}"


def _projection_score(url: str, text: str, index: int) -> int:
    body = f"{url}\n{text}".lower()
    score = index
    # Prefer new data detail / FTP modal tabs over old search result pages.
    if "ftp" in body or "download_" in body or "主机" in body or "端口" in body or "用户名" in body or "密码" in body:
        score += 10000
    if "/data/" in body or "/zh-hans/data/" in body or "数据集摘要" in body or "免费下载" in body:
        score += 5000
    if "alldata" in body or "search" in body or "查询结果" in body:
        score += 1000
    if "home" in body:
        score -= 500
    return score


def _collect_cdp_texts(endpoint: str, max_pages: int = 50, live_screenshot: Path | None = None, browser_pid: str | int | None = None, state_json: Path | None = None) -> tuple[list[str], str, dict]:
    """Read all visible TPDC tabs through CDP and update projection to the best current tab/window.

    V141: TPDC opens dataset details and FTP dialogs in new tabs. Older versions
    kept projecting the original search tab, which made automatic capture look
    broken. This function scans all TPDC tabs and screenshots the best current
    tab; on Windows it first tries to capture the whole Chrome window.
    """
    if not endpoint:
        return [], "no_endpoint", {}
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return [], f"playwright_unavailable:{exc}", {}

    texts: list[str] = []
    note = "ok"
    projection = {"mode": "none", "page_count": 0}
    js = r"""
    () => {
      const out = [];
      const push = (s) => {
        s = (s || '').toString().trim();
        if (s) out.push(s);
      };
      const isVisible = (el) => {
        try {
          const st = window.getComputedStyle(el);
          const r = el.getBoundingClientRect();
          return st && st.display !== 'none' && st.visibility !== 'hidden' && Number(st.opacity || 1) !== 0 && r.width > 0 && r.height > 0;
        } catch(e) { return true; }
      };
      push(location.href);
      push(document.title || '');
      push(document.body ? document.body.innerText : '');
      push(document.documentElement ? document.documentElement.innerText : '');
      // Keep a bounded HTML snapshot because some TPDC copy widgets store values
      // in attributes rather than visible text. This is read-only.
      try { push((document.documentElement ? document.documentElement.outerHTML : '').slice(0, 250000)); } catch(e) {}
      Array.from(document.querySelectorAll('input, textarea')).forEach((el) => { push(el.value); push(el.getAttribute('placeholder')); });
      Array.from(document.querySelectorAll('[title], [aria-label], [data-clipboard-text], [data-copy], [data-value], [value], [data-original-title], [data-content], a[href]')).forEach((el) => {
        ['title','aria-label','data-clipboard-text','data-copy','data-value','value','data-original-title','data-content','href'].forEach(attr => push(el.getAttribute(attr)));
      });
      Array.from(document.querySelectorAll('[role="dialog"], .modal, .el-dialog, .ivu-modal, .ant-modal, .layui-layer, button, a, span, div, p, td, th, li, label')).filter(isVisible).forEach((el) => push(el.innerText || el.textContent));
      return Array.from(new Set(out)).join('\n');
    }
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint)
            pages = []
            for ctx in list(getattr(browser, "contexts", []) or []):
                pages.extend(list(getattr(ctx, "pages", []) or []))
            pages = pages[-max_pages:] if len(pages) > max_pages else pages
            candidates = []
            for idx, page in enumerate(pages):
                try:
                    url = str(getattr(page, "url", ""))
                    if "tpdc.ac.cn" not in url:
                        continue
                    text = page.evaluate(js) or ""
                    texts.append(url)
                    texts.append(text)
                    for fr in list(getattr(page, "frames", []) or []):
                        try:
                            frame_text = fr.evaluate(js) or ""
                            texts.append(frame_text)
                            text += "\n" + frame_text
                        except Exception:
                            continue
                    candidates.append((_projection_score(url, text, idx), idx, page, url))
                except Exception:
                    continue
            projection["page_count"] = len(candidates)
            if live_screenshot:
                ok_win, win_note = _window_screenshot_by_pid(browser_pid, live_screenshot)
                if not ok_win:
                    ok_win, title_note = _window_screenshot_by_title(live_screenshot)
                    win_note = f"{win_note};{title_note}"
                if ok_win:
                    projection.update({"mode": "whole_browser_window", "note": win_note})
                elif candidates:
                    score, idx, best_page, best_url = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
                    try:
                        if _is_truth(os.getenv("DOMESTIC_PROJECTION_BRING_BEST_TAB_TO_FRONT", "1"), "1"):
                            best_page.bring_to_front()
                            time.sleep(0.2)
                    except Exception:
                        pass
                    shot_ok = _atomic_page_screenshot(best_page, live_screenshot)
                    shot_note = "playwright_page_screenshot" if shot_ok else "playwright_page_screenshot_failed"
                    if not shot_ok:
                        shot_ok, shot_note = _cdp_page_screenshot(best_page, live_screenshot)
                    if shot_ok:
                        projection.update({"mode": "best_tpdc_tab", "url": best_url, "score": score, "note": f"{win_note};{shot_note}"})
                    else:
                        # Do not leave stale screenshot in place. The UI will blank the old img.
                        projection.update({"mode": "screenshot_failed", "url": best_url, "score": score, "note": f"{win_note};{shot_note}"})
    except Exception as exc:
        note = f"cdp_read_failed:{exc}"
    if state_json and projection:
        _update_state(state_json, browser_projection=projection, live_screenshot=str(live_screenshot or ""))
    return texts, note, projection


def _write_handoff_files(download_dir: Path, accounts: list[dict]) -> dict:
    download_dir.mkdir(parents=True, exist_ok=True)
    accounts = _dedup_accounts(accounts)
    primary = accounts[0] if accounts else {}
    manifest = {
        "captured_at": _now(),
        "accounts": accounts,
        "primary": primary,
        "policy": "多个主机按页面显示顺序优先第一个；失败后可切换备用主机。",
    }
    accounts_path = download_dir / "ftp_accounts_detected.json"
    _write_json(accounts_path, manifest)
    ticket_files = _write_ftp_ticket_files(download_dir, accounts, source="readonly_cdp_or_clipboard")

    host = primary.get("host", "")
    port = int(primary.get("port") or 21)
    username = primary.get("username", "")
    password = primary.get("password", "")
    note = download_dir / "ftp_download_note.txt"
    note.write_text(
        "已捕捉到 TPDC FTP 账号信息。\n"
        "系统会按 TPDC 主备主机策略执行：先尝试页面显示的第一个主机；只有主主机失败时才切换第二个备用主机。\n\n"
        f"主机: {host}\n端口: {port}\n用户名: {username}\n密码: {password}\n\n"
        "Python 将尝试自动 FTP 下载；若 100G 级大文件中断，请使用生成的 WinSCP 脚本继续。\n",
        encoding="utf-8",
    )

    ftp_url = ""
    if host:
        ftp_url = f"ftp://{quote(username, safe='')}:{quote(password, safe='')}@{host}:{port}/"
    (download_dir / "open_ftp_client.bat").write_text(
        "@echo off\r\nchcp 65001 >nul\r\n"
        f"echo Host: {host}\r\n"
        f"echo Port: {port}\r\n"
        f"echo User: {username}\r\n"
        f"echo Password: {password}\r\n"
        "echo.\r\n"
        f"start \"\" \"{ftp_url}\"\r\n"
        "pause\r\n",
        encoding="utf-8",
    )
    winscp_script = download_dir / "winscp_download_script.txt"
    winscp_script.write_text(
        "option batch continue\n"
        "option confirm off\n"
        "option transfer binary\n"
        f"open ftp://{quote(username, safe='')}:{quote(password, safe='')}@{host}:{port}/\n"
        f"lcd \"{download_dir / 'files'}\"\n"
        "get -resume *\n"
        "exit\n",
        encoding="utf-8",
    )
    winscp_bat = download_dir / "open_winscp_download.bat"
    winscp_bat.write_text(
        "@echo off\r\nchcp 65001 >nul\r\n"
        "where winscp.com >nul 2>nul\r\n"
        "if errorlevel 1 (\r\n"
        "  echo 未检测到 winscp.com，请安装 WinSCP，或使用 FileZilla/FTP Rush 手动连接。\r\n"
        "  pause\r\n"
        "  exit /b 1\r\n"
        ")\r\n"
        f"winscp.com /script=\"{winscp_script}\"\r\n"
        "pause\r\n",
        encoding="utf-8",
    )
    out = {
        "accounts_path": str(accounts_path),
        "note_path": str(note),
        "ftp_url": ftp_url,
        "winscp_script": str(winscp_script),
        "winscp_bat": str(winscp_bat),
    }
    out.update(ticket_files)
    return out


def _connect_ftp(account: dict, timeout: int = 45) -> ftplib.FTP:
    ftp = ftplib.FTP()
    ftp.encoding = os.getenv("DOMESTIC_FTP_ENCODING", "utf-8")
    ftp.connect(str(account["host"]), int(account.get("port") or 21), timeout=timeout)
    ftp.login(str(account.get("username") or "anonymous"), str(account.get("password") or ""))
    ftp.set_pasv(True)
    return ftp


def _ftp_walk(ftp: ftplib.FTP, remote_dir: str | None = "/", max_files: int = 999999) -> list[dict]:
    """Best-effort recursive listing with MLSD first, then NLST fallback.

    `remote_dir=None` means use the FTP server's current working directory. Some
    TPDC temporary accounts reject explicit `/` but allow listing the login
    directory, so the caller should try both current directory and `/`.
    """
    out: list[dict] = []

    def join_child(path: str | None, name: str) -> str:
        if path in {None, "", "."}:
            return name
        return path.rstrip("/") + "/" + name if path != "/" else "/" + name

    def add_file(path: str, size: int | None = None):
        if len(out) < max_files:
            out.append({"path": path, "size": size})

    def walk_mlsd(path: str | None) -> bool:
        try:
            entries = list(ftp.mlsd() if path in {None, "", "."} else ftp.mlsd(path))
        except Exception:
            return False
        for name, facts in entries:
            if len(out) >= max_files:
                break
            if name in {".", ".."}:
                continue
            child = join_child(path, name)
            typ = str(facts.get("type", "")).lower()
            if typ == "dir":
                walk_mlsd(child)
            elif typ == "file" or not typ:
                size = None
                try:
                    size = int(facts.get("size")) if facts.get("size") else None
                except Exception:
                    size = None
                add_file(child, size)
        return True

    def walk_nlst(path: str | None):
        try:
            entries = ftp.nlst() if path in {None, "", "."} else ftp.nlst(path)
        except Exception:
            return
        for entry in entries:
            if len(out) >= max_files:
                break
            name = str(entry).rstrip("/").split("/")[-1]
            if not name or name in {".", ".."}:
                continue
            child = str(entry) if str(entry).startswith("/") else join_child(path, name)
            cur = None
            try:
                cur = ftp.pwd()
                ftp.cwd(child)
                if cur:
                    ftp.cwd(cur)
                walk_nlst(child)
            except Exception:
                try:
                    if cur:
                        ftp.cwd(cur)
                except Exception:
                    pass
                size = None
                try:
                    size = ftp.size(child)
                except Exception:
                    pass
                add_file(child, size)

    if not walk_mlsd(remote_dir):
        walk_nlst(remote_dir)
    return out


def _ftp_probe_session(ftp: ftplib.FTP, log_path: Path, host: str, role: str) -> dict:
    diag = {"host": host, "ftp_host_role": role, "pwd": "", "welcome": "", "syst": "", "root_list_ok": False, "current_list_ok": False, "errors": []}
    try:
        diag["welcome"] = str(ftp.getwelcome() or "")[:500]
    except Exception as exc:
        diag["errors"].append(f"welcome:{exc}")
    try:
        diag["syst"] = str(ftp.sendcmd("SYST") or "")[:500]
    except Exception as exc:
        diag["errors"].append(f"SYST:{exc}")
    try:
        diag["pwd"] = str(ftp.pwd() or "")
    except Exception as exc:
        diag["errors"].append(f"PWD:{exc}")
    for label, path in [("current", None), ("dot", "."), ("root", "/")]:
        try:
            sample = ftp.nlst() if path is None else ftp.nlst(path)
            diag[f"{label}_list_ok"] = True
            diag[f"{label}_list_sample"] = [str(x) for x in (sample or [])[:20]]
        except Exception as exc:
            diag[f"{label}_list_ok"] = False
            diag[f"{label}_list_error"] = str(exc)
            diag["errors"].append(f"NLST {label}:{exc}")
    _append_log(log_path, {"event": "ftp_session_probe", **diag})
    print(f"[TPDC_FTP_DIAG] host={host} role={role} pwd={diag.get('pwd')} current_list_ok={diag.get('current_list_ok')} root_list_ok={diag.get('root_list_ok')} errors={diag.get('errors')[:3]}", flush=True)
    return diag


def _ftp_list_with_diagnostics(ftp: ftplib.FTP, account: dict, log_path: Path, max_files: int = 999999) -> tuple[list[dict], dict]:
    host = str(account.get("host") or "")
    role = str(account.get("host_role") or "")
    diag = _ftp_probe_session(ftp, log_path, host, role)
    candidates = []
    for c in [account.get("remote_dir"), account.get("remote_path"), None, ".", "/"]:
        if c not in candidates:
            candidates.append(c)
    last_err = ""
    for c in candidates:
        try:
            files = _ftp_walk(ftp, c, max_files=max_files)
            _append_log(log_path, {"event": "ftp_list_candidate", "host": host, "ftp_host_role": role, "candidate": c if c is not None else "<current>", "file_count": len(files)})
            if files:
                diag["selected_remote_dir"] = c if c is not None else "<current>"
                diag["selected_file_count"] = len(files)
                return files, diag
        except Exception as exc:
            last_err = str(exc)
            _append_log(log_path, {"event": "ftp_list_candidate_error", "host": host, "ftp_host_role": role, "candidate": c if c is not None else "<current>", "error": last_err})
    msg = "FTP 登录成功，但无法列出可下载目录。"
    if any("421" in str(e) or "Home directory" in str(e) for e in diag.get("errors", [])) or "421" in last_err:
        msg += " 服务端返回 421 Home directory not available，表示该临时账号当前没有可访问的默认目录；通常需要在 TPDC 页面重新生成 FTP 参数，或等待服务端刷新该数据集目录。"
    if diag.get("errors"):
        msg += " 诊断：" + "；".join(str(e) for e in diag.get("errors", [])[:5])
    raise RuntimeError(msg)


def _should_skip_by_year(path: str, requested_year: str, matching_years_exist: bool) -> bool:
    """Conservative year filter. Disabled unless matching files exist.

    If the server contains explicit files for the target year, skip files that
    contain a different 19xx/20xx year. If no target-year file exists, do not skip,
    because many TPDC datasets store many years inside one .dat/.zip file.
    """
    if not requested_year or not matching_years_exist:
        return False
    years = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", path)
    return bool(years) and str(requested_year) not in years



def _rank_and_limit_selected_files(files: list[dict], requested_year: str = "") -> list[dict]:
    """Rank FTP files for the current user request and apply a safe limit.

    This function is intentionally conservative: the year filter is applied
    before it is called, so here we only remove duplicate remote paths, prefer
    scientific data files over logs/checksums/web assets, and cap the number of
    files to avoid accidentally mirroring a whole TPDC FTP tree.  The missing
    function caused V232 downloads to fail with
    ``name '_rank_and_limit_selected_files' is not defined`` immediately after
    FTP connection succeeded.
    """
    unique: list[dict] = []
    seen: set[str] = set()
    for f in files or []:
        if not isinstance(f, dict):
            continue
        remote = str(f.get("path") or f.get("remote") or "").strip()
        if not remote:
            continue
        key = remote.lower()
        if key in seen:
            continue
        seen.add(key)
        item = dict(f)
        item["path"] = remote
        unique.append(item)

    target_year = str(requested_year or "").strip()
    data_ext_rank = {
        ".nc": 80, ".nc4": 80, ".cdf": 76,
        ".tif": 74, ".tiff": 74,
        ".hdf": 70, ".h5": 70, ".hdf5": 70,
        ".zip": 60, ".gz": 52, ".tar": 50, ".7z": 48,
        ".img": 44, ".grd": 42, ".dat": 35,
    }
    bad_ext = {".png", ".jpg", ".jpeg", ".gif", ".html", ".htm", ".xml", ".txt", ".md5", ".sha1", ".sha256", ".json"}

    def score(item: dict) -> tuple[int, int, str]:
        remote = str(item.get("path") or "")
        name = Path(remote).name.lower()
        suffix = Path(name).suffix.lower()
        sc = 0
        if target_year and re.search(rf"(?<!\d){re.escape(target_year)}(?!\d)", remote):
            sc += 200
        sc += data_ext_rank.get(suffix, 10)
        if suffix in bad_ext:
            sc -= 100
        if any(x in name for x in ["readme", "license", "thumb", "preview", "browse", "checksum"]):
            sc -= 60
        size = int(item.get("size") or 0)
        # Prefer larger real data files after extension/year scoring, but avoid
        # making size dominate the selection.
        size_rank = min(size // (1024 * 1024), 999)
        return (sc, size_rank, remote)

    ranked = sorted(unique, key=score, reverse=True)
    try:
        limit = int(os.getenv("DOMESTIC_FTP_SELECTED_FILE_LIMIT", os.getenv("DOMESTIC_FTP_MAX_SELECTED_FILES", "999999")))
    except Exception:
        limit = 999999
    if limit > 0:
        ranked = ranked[:limit]
    return ranked

def _download_one_with_resume(account: dict, remote_path: str, local_path: Path, log_path: Path, retries: int = 5, status_path: Path | None = None, progress_meta: dict | None = None) -> dict:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = ""
    progress_meta = dict(progress_meta or {})
    remote_size = int(progress_meta.get("current_remote_size") or 0)
    completed_before = int(progress_meta.get("completed_bytes_before") or 0)
    total_known = int(progress_meta.get("remote_total_known_bytes") or 0)
    for attempt in range(1, retries + 1):
        ftp = None
        try:
            ftp = _connect_ftp(account, timeout=int(os.getenv("DOMESTIC_FTP_CONNECT_TIMEOUT", "45")))
            existing = local_path.stat().st_size if local_path.exists() else 0
            mode = "ab" if existing > 0 else "wb"
            started = time.time()
            last_tick = started
            bytes_this_session = 0
            with local_path.open(mode + ("" if "b" in mode else "b")) as f:
                def cb(chunk: bytes):
                    nonlocal last_tick, bytes_this_session
                    f.write(chunk)
                    bytes_this_session += len(chunk)
                    now = time.time()
                    if now - last_tick >= 5:
                        last_tick = now
                        try:
                            local_bytes = local_path.stat().st_size if local_path.exists() else existing + bytes_this_session
                        except Exception:
                            local_bytes = existing + bytes_this_session
                        downloaded_total = completed_before + local_bytes
                        progress_percent = (downloaded_total / total_known * 100.0) if total_known else 0.0
                        rec = {
                            "event": "file_progress",
                            "remote": remote_path,
                            "local": str(local_path),
                            "existing_before_session": existing,
                            "bytes_this_session": bytes_this_session,
                            "local_bytes": local_bytes,
                            "attempt": attempt,
                            "downloaded_bytes_total": downloaded_total,
                            "remote_total_known_bytes": total_known,
                            "progress_percent": round(progress_percent, 3) if progress_percent else 0.0,
                        }
                        _append_log(log_path, rec)
                        if status_path:
                            status = dict(progress_meta)
                            status.update({
                                "status": "ftp_downloading",
                                "ftp_download_status": "downloading",
                                "current_remote": remote_path,
                                "current_local": str(local_path),
                                "current_local_bytes": local_bytes,
                                "current_remote_size": remote_size,
                                "completed_bytes_before": completed_before,
                                "downloaded_bytes_total": downloaded_total,
                                "remote_total_known_bytes": total_known,
                                "progress_percent": round(progress_percent, 3) if progress_percent else 0.0,
                                "attempt": attempt,
                                "updated_at": _now(),
                            })
                            _write_json(status_path, status)
                try:
                    ftp.retrbinary(f"RETR {remote_path}", cb, blocksize=1024 * 1024, rest=existing if existing > 0 else None)
                except TypeError:
                    # Older Python fallback: rest may not accept None cleanly.
                    if existing > 0:
                        ftp.sendcmd(f"REST {existing}")
                    ftp.retrbinary(f"RETR {remote_path}", cb, blocksize=1024 * 1024)
            try:
                ftp.quit()
            except Exception:
                pass
            return {"ok": True, "remote": remote_path, "local": str(local_path), "bytes": local_path.stat().st_size, "attempts": attempt}
        except Exception as exc:
            last_err = str(exc)
            _append_log(log_path, {"event": "file_download_error", "remote": remote_path, "attempt": attempt, "error": last_err})
            try:
                if ftp:
                    ftp.close()
            except Exception:
                pass
            time.sleep(min(30, 2 * attempt))
    return {"ok": False, "remote": remote_path, "local": str(local_path), "error": last_err, "attempts": retries}


def _start_python_ftp_download(download_dir: Path, accounts: list[dict], requested_year: str = "") -> dict:
    """Download using ftplib with resume using one fixed TPDC host.

    V232 intentionally uses only ftp2.tpdc.ac.cn:6201 with one captured
    username/password ticket. A directory-level lock prevents duplicate child
    downloaders from simultaneous watcher/capture paths.
    """
    # Normalize captured credentials before any download attempt.  Some DOM
    # scans may contain UI artefacts; only accept TPDC temporary users of the
    # form download_xxx, then rebuild the single fixed FTP host and port.
    valid_cred = None
    for a in accounts or []:
        u = str((a or {}).get("username") or "").strip()
        pw = str((a or {}).get("password") or "").strip()
        if _is_valid_tpdc_username(u) and _is_valid_tpdc_password(pw):
            valid_cred = dict(a)
            valid_cred["username"] = u
            valid_cred["password"] = pw
            break
    if not valid_cred:
        return {"ok": False, "status": "no_valid_tpdc_credentials", "message": "未捕捉到有效TPDC FTP用户名/密码。"}
    primary_host, _backup_host, fixed_port = _tpdc_fixed_ftp_hosts()
    b = dict(valid_cred)
    b.update({"host": primary_host, "port": int(fixed_port or 6201), "priority": 1, "primary": True, "host_role": "primary", "fixed_host_policy": "tpdc_single_primary_fixed_host"})
    accounts = _tag_primary_backup_accounts([b])
    if not accounts:
        return {"ok": False, "status": "no_accounts"}
    status_path = download_dir / "ftp_download_status.json"
    log_path = download_dir / "ftp_python_downloader.log"
    lock_ok, lock_path, lock_info = _try_create_downloader_lock(download_dir)
    if not lock_ok:
        _append_log(log_path, {"event": "duplicate_downloader_suppressed", "lock": str(lock_path), "lock_info": lock_info})
        return {"ok": True, "status": "ftp_download_process_already_running", "lock": str(lock_path), "existing": lock_info}
    max_files = int(os.getenv("DOMESTIC_FTP_MAX_FILES", "999999"))
    retries = int(os.getenv("DOMESTIC_FTP_RETRIES_PER_FILE", "5"))
    results = []
    errors = []

    for idx, account in enumerate(accounts):
        account = dict(account)
        host = str(account.get("host") or "")
        host_role = str(account.get("host_role") or ("primary" if idx == 0 else "backup"))
        host_role_label = "主主机" if host_role == "primary" else "备用主机"
        _debug_print_ftp_accounts([account], source=f"downloader_try_{host_role}_{idx+1}")
        local_root = download_dir / "files"
        _update = {
            "status": "ftp_connecting",
            "ftp_download_status": "connecting",
            "ftp_host": host,
            "ftp_host_role": host_role,
            "ftp_host_role_label": host_role_label,
            "ftp_host_attempt_index": idx + 1,
            "ftp_host_total": len(accounts),
            "ftp_host_strategy": "single_primary_only",
            "ftp_account": {k: v for k, v in account.items() if k != "password"},
            "ftp_download_status_path": str(status_path),
        }
        _write_json(status_path, {**_update, "updated_at": _now(), "note": f"正在连接 TPDC FTP {host_role_label}：{host}。本版本仅使用固定主机 ftp2.tpdc.ac.cn:6201。"})
        try:
            ftp = _connect_ftp(account, timeout=int(os.getenv("DOMESTIC_FTP_CONNECT_TIMEOUT", "45")))
            files, ftp_diag = _ftp_list_with_diagnostics(ftp, account, log_path, max_files=max_files)
            try:
                ftp.quit()
            except Exception:
                pass
            if not files:
                raise RuntimeError("FTP 连接成功，但未列出可下载文件。")
            target_year = str(requested_year or "").strip()
            matching_years_exist = bool(target_year and any(re.search(rf"(?<!\d){re.escape(target_year)}(?!\d)", f.get("path", "")) for f in files))
            selected_files = [f for f in files if not _should_skip_by_year(f.get("path", ""), target_year, matching_years_exist)]
            selected_files = _rank_and_limit_selected_files(selected_files, target_year)
            total_known = sum(int(f.get("size") or 0) for f in selected_files)
            free = shutil.disk_usage(str(download_dir)).free
            space_warning = ""
            if total_known and free < total_known * 1.15:
                space_warning = f"磁盘剩余空间可能不足：已知待下载约 {total_known/1024**3:.2f} GB，剩余 {free/1024**3:.2f} GB。"
            _write_json(status_path, {
                "status": "ftp_downloading",
                "ftp_download_status": "downloading",
                "host": host,
                "ftp_host_role": host_role,
                "ftp_host_role_label": host_role_label,
                "file_count": len(selected_files),
                "remote_total_known_bytes": total_known,
                "local_free_bytes": free,
                "space_warning": space_warning,
                "year_filter": {"requested_year": target_year, "matching_year_files_exist": matching_years_exist},
                "download_root": str(local_root),
                "ftp_diagnostics": ftp_diag,
                "updated_at": _now(),
            })
            ok_count = 0
            completed_bytes_before = 0
            for i, f in enumerate(selected_files, start=1):
                remote = f.get("path") or ""
                # Keep the local download directory shallow for users.  TPDC
                # often exposes deep FTP paths such as /historical/...; store
                # selected files directly under files/ with a safe unique name.
                filename = Path(str(remote).rstrip("/")).name or _safe_name(remote)
                local_path = local_root / _safe_name(filename)
                if local_path.exists() and str(remote).count("/") > 1:
                    # Avoid collisions when different remote folders share a filename.
                    prefix = _safe_name("_".join([x for x in str(remote).split("/")[-3:-1] if x]), 80)
                    if prefix:
                        local_path = local_root / _safe_name(prefix + "_" + filename)
                current_remote_size = int(f.get("size") or 0)
                _write_json(status_path, {
                    "status": "ftp_downloading",
                    "ftp_download_status": "downloading",
                    "host": host,
                    "ftp_host_role": host_role,
                    "ftp_host_role_label": host_role_label,
                    "current_index": i,
                    "file_count": len(selected_files),
                    "current_remote": remote,
                    "current_local": str(local_path),
                    "current_local_bytes": local_path.stat().st_size if local_path.exists() else 0,
                    "current_remote_size": current_remote_size,
                    "completed_bytes_before": completed_bytes_before,
                    "downloaded_bytes_total": completed_bytes_before + (local_path.stat().st_size if local_path.exists() else 0),
                    "remote_total_known_bytes": total_known,
                    "progress_percent": round(((completed_bytes_before + (local_path.stat().st_size if local_path.exists() else 0)) / total_known * 100.0), 3) if total_known else 0.0,
                    "download_root": str(local_root),
                    "updated_at": _now(),
                })
                progress_meta = {
                    "host": host,
                    "ftp_host_role": host_role,
                    "ftp_host_role_label": host_role_label,
                    "current_index": i,
                    "file_count": len(selected_files),
                    "download_root": str(local_root),
                    "ftp_host_role": host_role,
                    "ftp_host_role_label": host_role_label,
                    "current_remote_size": current_remote_size,
                    "completed_bytes_before": completed_bytes_before,
                    "remote_total_known_bytes": total_known,
                }
                r = _download_one_with_resume(account, remote, local_path, log_path, retries=retries, status_path=status_path, progress_meta=progress_meta)
                results.append(r)
                if r.get("ok"):
                    ok_count += 1
                    completed_bytes_before += int(r.get("bytes") or current_remote_size or 0)
                if not r.get("ok"):
                    errors.append(r)
            final = {
                "ok": ok_count > 0,
                "status": "ftp_download_completed" if ok_count == len(selected_files) else "ftp_download_partial" if ok_count else "ftp_download_failed",
                "host": host,
                "ftp_host_role": host_role,
                "ftp_host_role_label": host_role_label,
                "download_root": str(local_root),
                "file_count": len(selected_files),
                "downloaded_ok_count": ok_count,
                "downloaded_bytes_total": completed_bytes_before,
                "remote_total_known_bytes": total_known,
                "progress_percent": round((completed_bytes_before / total_known * 100.0), 3) if total_known else (100.0 if ok_count else 0.0),
                "errors_count": len(errors),
                "results_preview": results[:20],
                "errors_preview": errors[:20],
                "updated_at": _now(),
            }
            _write_json(status_path, final)
            if ok_count > 0 or idx == len(accounts) - 1:
                return final
        except Exception as exc:
            err = {"host": host, "ftp_host_role": host_role, "ftp_host_role_label": host_role_label, "error": str(exc)}
            errors.append(err)
            _append_log(log_path, {"event": "host_failed", **err, "will_try_next_host": idx < len(accounts) - 1})
            if idx < len(accounts) - 1:
                note = f"TPDC FTP {host_role_label} {host} 连接或目录访问失败，正在切换备用主机。"
            else:
                note = f"TPDC FTP {host_role_label} {host} 也失败，所有主备主机均不可用。"
            _write_json(status_path, {"status": "ftp_host_failed", "host": host, "ftp_host_role": host_role, "ftp_host_role_label": host_role_label, "error": str(exc), "will_try_next_host": idx < len(accounts) - 1, "note": note, "updated_at": _now()})
            continue
    return {"ok": False, "status": "ftp_download_failed", "errors": errors[:20], "updated_at": _now()}



def _spawn_python_ftp_downloader(download_dir: Path, accounts_path: Path, requested_year: str = "", state_json: Path | None = None) -> dict:
    """Start the heavy FTP transfer in a separate process.

    Only one downloader is allowed per TPDC download directory.  The watcher may
    see the same ticket through clipboard and CDP text, and playwright can also
    observe the same DOM.  This function claims a parent-side spawn lock before
    launching the child process so the same username/password ticket is not
    downloaded twice.
    """
    claimed, claim_info = _claim_download_spawn(download_dir, source="ftp_clipboard_watcher")
    if not claimed:
        result = {"ok": True, "status": "ftp_download_process_already_running", "existing": claim_info}
        _update_state(state_json, status="ftp_download_process_already_running", ftp_watcher_status="download_already_running", ftp_download_process=result, message="已捕捉到 FTP 账号，但当前任务已有下载进程在运行；不会重复启动第二个下载器。")
        return result
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-download-from-json", str(accounts_path),
            "--download-dir", str(download_dir),
            "--requested-year", str(requested_year or ""),
        ]
        if state_json:
            cmd += ["--state-json", str(state_json)]
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        result = {"ok": True, "status": "ftp_download_process_started", "pid": proc.pid, "command": cmd, "accounts_path": str(accounts_path)}
        _update_state(state_json, status="ftp_download_process_started", ftp_watcher_status="download_started", ftp_download_process=result, message="已捕捉到 FTP 账号，Python FTP 大文件下载进程已启动；进度见 ftp_download_status.json / ftp_python_downloader.log。")
        return result
    except Exception as exc:
        result = {"ok": False, "status": "ftp_download_process_failed_to_start", "error": str(exc)}
        _update_state(state_json, status="ftp_download_process_failed_to_start", ftp_watcher_status="download_start_failed", ftp_download_process=result, message=f"已捕捉 FTP 账号，但启动 Python FTP 下载进程失败：{exc}")
        return result


def _run_download_from_json(accounts_path: Path, download_dir: Path, requested_year: str = "", state_json: Path | None = None) -> int:
    try:
        data = json.loads(accounts_path.read_text(encoding="utf-8") or "{}")
        accounts = data.get("accounts") or ([] if not data.get("primary") else [data.get("primary")])
        if not accounts:
            raise RuntimeError("accounts json does not contain FTP accounts")
        _update_state(state_json, status="ftp_downloading", ftp_watcher_status="downloading", message="Python FTP 下载进程正在连接服务器并列目录。")
        result = _start_python_ftp_download(download_dir, accounts, requested_year=requested_year)
        _update_state(state_json, status=result.get("status") or "ftp_download_processed", ftp_watcher_status="download_processed", ftp_download_result=result, message="FTP 下载进程已更新状态；若中断，可用 open_winscp_download.bat 接管续传。")
        return 0 if result.get("ok") else 4
    except Exception as exc:
        _update_state(state_json, status="ftp_download_failed", ftp_watcher_status="download_failed", message=f"Python FTP 下载进程失败：{exc}")
        return 5

def main() -> int:
    parser = argparse.ArgumentParser(description="Passive TPDC FTP watcher and downloader")
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--cdp-endpoint", default="")
    parser.add_argument("--live-screenshot", default="")
    parser.add_argument("--browser-pid", default="")
    parser.add_argument("--state-json", default="")
    parser.add_argument("--auto-download", default="1")
    parser.add_argument("--requested-year", default="")
    parser.add_argument("--data-name", default="")
    parser.add_argument("--run-download-from-json", default="", help="Internal mode: run FTP download from ftp_accounts_detected.json")
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    state_json = Path(args.state_json) if args.state_json else None
    if args.run_download_from_json:
        return _run_download_from_json(Path(args.run_download_from_json), download_dir, requested_year=args.requested_year, state_json=state_json)
    download_dir.mkdir(parents=True, exist_ok=True)
    log_path = download_dir / "ftp_account_watcher.log"
    live_screenshot = Path(args.live_screenshot) if args.live_screenshot else None
    deadline = time.time() + max(10, int(args.timeout))
    seen = set()
    probe_count = 0
    last_state_update = 0.0
    _append_log(log_path, {
        "event": "watcher_started",
        "mode": "clipboard_plus_readonly_cdp",
        "cdp_endpoint": args.cdp_endpoint,
        "download_dir": str(download_dir),
        "auto_download": args.auto_download,
        "data_name": args.data_name,
        "requested_year": args.requested_year,
    })
    _update_state(state_json, status="ftp_account_waiting", ftp_watcher_status="waiting", message="FTP 捕捉器已启动：用户点击 TPDC 下载/FTP 后，系统会只读捕捉账号并自动下载。")

    while time.time() < deadline:
        texts = []
        sources = []
        # Clipboard is prone to stale TPDC credentials from a previous dataset.
        # When a CDP endpoint is available, prefer the current TPDC page text and
        # ignore clipboard unless explicitly enabled for fallback debugging.
        use_clipboard = _is_truth(os.getenv("DOMESTIC_FTP_ALLOW_CLIPBOARD_WITH_CDP", "0" if args.cdp_endpoint else "1"), "1")
        clip = _clipboard_text() if use_clipboard else ""
        if clip:
            texts.append(clip)
            sources.append("clipboard")
        if args.cdp_endpoint and _is_truth(os.getenv("DOMESTIC_FTP_READONLY_CDP_WATCHER", "1"), "1"):
            cdp_texts, cdp_note, projection = _collect_cdp_texts(args.cdp_endpoint, live_screenshot=live_screenshot, browser_pid=args.browser_pid, state_json=state_json)
            if cdp_texts:
                texts.extend(cdp_texts)
                sources.append("readonly_cdp")
            elif cdp_note and "no_endpoint" not in cdp_note:
                _append_log(log_path, {"event": "cdp_probe_note", "note": cdp_note})
        accounts = _dedup_accounts([a for txt in texts for a in _extract_ftp_accounts_from_text(txt)])
        probe_count += 1
        combined = "\n".join(texts)[:20000]
        ftp_signal_seen = bool(re.search(r"用\s*户\s*名|密\s*码", combined))
        now = time.time()
        if now - last_state_update >= float(os.getenv("DOMESTIC_FTP_PROBE_STATUS_INTERVAL_SECONDS", "5")):
            last_state_update = now
            _update_state(
                state_json,
                status="ftp_account_waiting",
                ftp_watcher_status="waiting",
                ftp_probe_count=probe_count,
                ftp_probe_sources=list(dict.fromkeys(sources)),
                ftp_signal_seen=ftp_signal_seen,
                ftp_last_text_chars=sum(len(t) for t in texts),
                browser_projection=locals().get("projection", {}),
                message=("FTP 捕捉器运行中：已看到中文用户名/密码字段，正在等待字段完整。" if ftp_signal_seen else "FTP 捕捉器运行中：等待 TPDC 弹出中文用户名和密码字段。"),
            )
            if ftp_signal_seen and not accounts:
                try:
                    (download_dir / "ftp_parse_pending_visible_text.txt").write_text(combined, encoding="utf-8")
                except Exception:
                    pass
        if accounts:
            # Collapse all host variants to one TPDC credential ticket. The
            # downloader will internally keep only ftp2.tpdc.ac.cn:6201 from
            # this single username/password; this prevents stale or duplicate
            # host entries from being treated as multiple downloads.
            first = accounts[0]
            primary_host, _backup_host, fixed_port = _tpdc_fixed_ftp_hosts()
            accounts = [
                {**first, "host": primary_host, "port": fixed_port, "primary": True, "priority": 1, "host_role": "primary", "fixed_host_policy": "tpdc_single_primary_fixed_host"},
            ]
            sig = (first.get("username"), first.get("password"))
            if sig not in seen:
                seen.add(sig)
                files = _write_handoff_files(download_dir, accounts)
                _debug_print_ftp_accounts(accounts, source="watcher_capture_single_ticket_primary_only")
                _append_log(log_path, {"event": "ftp_accounts_captured", "sources": sources, "accounts": accounts, "files": files})
                _update_state(state_json, status="ftp_credentials_detected", ftp_watcher_status="captured", ftp_accounts_count=1, ftp_ticket_count=1, ftp_host_count=1, ftp_handoff_files=files, message="已捕捉到 1 组 TPDC FTP 用户名和密码；主机与端口固定为 ftp2.tpdc.ac.cn:6201，本版本不尝试备用主机。")
                if _is_truth(args.auto_download, "1"):
                    accounts_path = Path(files.get("accounts_path") or (download_dir / "ftp_accounts_detected.json"))
                    if _is_truth(os.getenv("DOMESTIC_FTP_DOWNLOAD_BACKGROUND", "1"), "1"):
                        result = _spawn_python_ftp_downloader(download_dir, accounts_path, requested_year=args.requested_year, state_json=state_json)
                    else:
                        result = _start_python_ftp_download(download_dir, accounts, requested_year=args.requested_year)
                        _update_state(state_json, status=result.get("status") or "ftp_download_processed", ftp_watcher_status="download_processed", ftp_download_result=result, message="FTP 下载流程已处理；若下载中断，可使用 open_winscp_download.bat 接管续传。")
                    _append_log(log_path, {"event": "ftp_download_result", "result": result})
                return 0
        time.sleep(max(0.5, float(args.interval)))
    _append_log(log_path, {"event": "watcher_timeout"})
    _update_state(state_json, status="ftp_account_waiting_timeout", ftp_watcher_status="timeout", message="FTP 捕捉器超时；未检测到中文“用户名”和“密码”。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
