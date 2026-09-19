from __future__ import annotations

"""
TPDC browser login/search + passive FTP capture and downloader tool.

作用：
1. 打开真实浏览器，而不是 requests 扫 HTML。
2. 使用 persistent profile 保存登录态/cookie。
3. 你在浏览器中人工登录、搜索数据、点击下载。
4. 脚本捕获浏览器 download 事件，把真实文件保存到 data/manual_domestic_downloads/<platform>/。
5. 主流水线会自动导入该目录中的真实文件，继续做类型识别、标准化、样点抽取、model_ready.csv。

注意：
- 这不是绕过验证码/审核。验证码、人机验证、订单确认仍由用户在浏览器里人工完成。
- 捕获到 HTML/JS/站点资源文件不会被主流水线计为真实原始数据。
"""

import argparse
import ftplib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlparse

PLATFORM_URLS = {
    # V124: hard TPDC-only mode. Other domestic portals are deliberately removed
    # from the executable capture tool so it cannot jump to GSCloud again.
    "tpdc": "https://data.tpdc.ac.cn/",
}

PLATFORM_NAMES = {
    "tpdc": "国家青藏高原科学数据中心",
}


PLATFORM_CREDENTIAL_ENV = {
    "tpdc": ("TPDC_USERNAME", "TPDC_PASSWORD"),
}


def _load_project_env(project_root: Path) -> None:
    """Best-effort .env loading for direct tool runs.

    The Dash app already loads .env, but this helper lets the capture tool work
    when launched independently from a terminal.
    """
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
    except Exception:
        pass


def _clean_secret(value: str | None) -> str:
    v = str(value or "").strip()
    if not v or v.lower().startswith("your_") or "填写" in v or "请" in v:
        return ""
    return v


def _get_platform_credentials(platform: str) -> tuple[str, str, str, str]:
    user_env, pass_env = PLATFORM_CREDENTIAL_ENV.get(platform, ("", ""))
    username = _clean_secret(os.getenv(user_env)) if user_env else ""
    password = _clean_secret(os.getenv(pass_env)) if pass_env else ""
    return user_env, pass_env, username, password


def _mask_username(username: str) -> str:
    u = str(username or "").strip()
    if not u:
        return "未配置"
    if "@" in u:
        head, tail = u.split("@", 1)
        return (head[:2] if len(head) > 2 else head[:1]) + "***@" + tail
    return (u[:2] + "***" + u[-1:]) if len(u) > 3 else u[:1] + "***"


def _is_truth(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _try_click_login_entry(page, platform: str) -> None:
    """Open login form if the platform exposes an obvious login entry.

    This is deliberately conservative: it never bypasses verification and it is
    fine if selectors fail. The user can still click manually.
    """
    candidates = [
        "text=登录", "text=登陆", "text=用户登录", "text=账号登录", "text=Sign in", "text=Login",
        "a:has-text('登录')", "button:has-text('登录')", "a:has-text('登陆')", "button:has-text('登陆')",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=800):
                loc.click(timeout=1200)
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                _wait_for_login_form(page, timeout_ms=12000)
                return
        except Exception:
            continue



def _open_tpdc_login_panel(page, entry_url: str = "https://data.tpdc.ac.cn/") -> dict:
    """Open the visible TPDC login panel/page and wait for the password field.

    The previous builds relied on a loose `text=登录` click. On TPDC this can be
    brittle because the SPA has duplicated text nodes and hidden templates. V124
    deliberately uses visible-element clicking first, then conservative known
    route fallbacks. It never writes credentials unless a visible password box is
    present.
    """
    result = {"attempted": True, "opened": False, "method": "", "note": ""}
    if _wait_for_login_form(page, timeout_ms=1200):
        result.update({"opened": True, "method": "already_visible", "note": "登录表单已可见。"})
        return result

    selectors = [
        "a:has-text('登录')",
        "button:has-text('登录')",
        "span:has-text('登录')",
        "div:has-text('登录')",
        "text=/^\\s*登录\\s*$/",
        "text=/^\\s*登陆\\s*$/",
    ]
    for sel in selectors:
        try:
            locs = page.locator(sel)
            count = min(locs.count(), 12)
            for i in range(count):
                loc = locs.nth(i)
                if not loc.is_visible(timeout=400):
                    continue
                try:
                    box = loc.bounding_box(timeout=500)
                except Exception:
                    box = None
                # The TPDC header login entry is near the top. Avoid clicking
                # arbitrary body text that merely contains the word 登录.
                if box and (box.get("y", 9999) > 320 or box.get("width", 0) > 220):
                    continue
                loc.click(timeout=1800)
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                if _wait_for_login_form(page, timeout_ms=15000):
                    result.update({"opened": True, "method": f"selector:{sel}[{i}]", "note": "已点击 TPDC 登录入口并打开登录表单。"})
                    return result
        except Exception:
            continue

    # JS fallback: click the first visible compact element whose own visible text
    # is exactly 登录/登陆. This avoids Playwright text matching a large parent div.
    try:
        clicked = page.evaluate(
            """
            () => {
              const isVisible = (el) => {
                const st = window.getComputedStyle(el);
                const r = el.getBoundingClientRect();
                return st && st.display !== 'none' && st.visibility !== 'hidden' &&
                       Number(st.opacity || 1) !== 0 && r.width > 5 && r.height > 5 &&
                       r.top >= 0 && r.top < 320 && r.left >= 0 && r.left < window.innerWidth;
              };
              const els = Array.from(document.querySelectorAll('a,button,span,div'))
                .filter(isVisible)
                .filter(el => /^(登录|登陆)$/.test((el.innerText || el.textContent || '').trim()))
                .sort((a,b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);
              if (!els.length) return false;
              els[0].click();
              return true;
            }
            """
        )
        if clicked:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass
            if _wait_for_login_form(page, timeout_ms=15000):
                result.update({"opened": True, "method": "js_exact_visible_login_text", "note": "已通过可见登录文本打开 TPDC 登录表单。"})
                return result
    except Exception:
        pass

    # Route fallback. Keep this last because TPDC is an SPA and the exact route
    # may change; clicking the native header entry is preferred.
    for url in [
        "https://data.tpdc.ac.cn/login",
        "https://data.tpdc.ac.cn/user/login",
        "https://data.tpdc.ac.cn/#/login",
        "https://data.tpdc.ac.cn/#/user/login",
        "https://data.tpdc.ac.cn/#/home?login=1",
    ]:
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            if _wait_for_login_form(page, timeout_ms=8000):
                result.update({"opened": True, "method": f"route:{url}", "note": "已通过 TPDC 登录路由打开登录表单。"})
                return result
        except Exception:
            continue

    result["note"] = "未能自动打开 TPDC 登录表单；请手动点击页面右上角“登录”。"
    return result


def _wait_for_login_completion(page, timeout_seconds: int = 600) -> dict:
    """Wait for the user to finish captcha/manual verification after credentials fill."""
    started = time.time()
    timeout_seconds = max(0, int(timeout_seconds or 0))
    if timeout_seconds <= 0:
        return {"waited": False, "completed": False, "seconds": 0, "note": "未等待用户验证码。"}
    while time.time() - started < timeout_seconds:
        try:
            if not _page_has_visible_password_or_verification(page):
                return {"waited": True, "completed": True, "seconds": int(time.time() - started), "note": "登录/验证码界面已消失，继续自动检索。"}
        except Exception:
            pass
        try:
            page.wait_for_timeout(1000)
        except Exception:
            time.sleep(1)
    return {"waited": True, "completed": False, "seconds": int(time.time() - started), "note": "等待用户输入验证码超时；浏览器保持打开。"}


def _wait_for_login_form(page, timeout_ms: int = 15000) -> bool:
    """Wait until TPDC/login page exposes a visible password field.

    The previous build clicked 登录 and immediately tried to fill. On TPDC the
    login page may still be navigating/rendering, so the fill routine ran too
    early and left the form blank. A visible password field is the hard gate;
    without it we never write credentials into text inputs.
    """
    deadline = time.time() + max(timeout_ms, 1000) / 1000.0
    selectors = [
        "input[type='password']",
        "input[placeholder*='密码']",
        "input[aria-label*='密码']",
    ]
    while time.time() < deadline:
        try:
            page.wait_for_load_state("domcontentloaded", timeout=1200)
        except Exception:
            pass
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if loc.count() and loc.is_visible(timeout=500):
                    return True
            except Exception:
                continue
        page.wait_for_timeout(500)
    return False




def _page_has_visible_password_or_verification(page) -> bool:
    """Return True only when the CURRENT VISIBLE UI is a login/verification form.

    V122 used body.inner_text() and treated hidden SPA templates containing
    “验证码/请输入密码” as a real login page. TPDC keeps login templates in the DOM
    even on the normal home/search pages, so that logic blocked search before it
    started. V124 inspects visible controls and the current URL only.
    """
    # V126: do not treat URL alone as a blocker. TPDC can keep /login in
    # history/route state while the visible login panel is gone, and the old
    # URL-based test prevented the search stage from resuming after captcha login.
    try:
        visible = page.evaluate(
            """
            () => {
              const isVisible = (el) => {
                if (!el) return false;
                const st = window.getComputedStyle(el);
                const r = el.getBoundingClientRect();
                return st && st.display !== 'none' && st.visibility !== 'hidden' &&
                       Number(st.opacity || 1) !== 0 && r.width > 3 && r.height > 3 &&
                       r.bottom >= 0 && r.right >= 0 && r.top <= window.innerHeight + 20 &&
                       r.left <= window.innerWidth + 20;
              };
              const attr = (el) => [el.type, el.name, el.id, el.placeholder, el.getAttribute('aria-label'), el.className]
                .filter(Boolean).join(' ').toLowerCase();
              const inputs = Array.from(document.querySelectorAll('input,textarea')).filter(isVisible);
              const hasPassword = inputs.some(el => (el.type || '').toLowerCase() === 'password' || /密码|password|pass/.test(attr(el)));
              const hasCaptcha = inputs.some(el => /验证码|captcha|短信|sms|code/.test(attr(el)));
              // V126: do NOT block the workflow just because visible text contains
              // 登录/用户登录. TPDC home/search pages keep login-related words in
              // header/menu nodes. Only visible password/captcha input controls mean
              // the current UI is truly waiting for manual verification.
              return Boolean(hasPassword || hasCaptcha);
            }
            """
        )
        return bool(visible)
    except Exception:
        return False

def _fill_first_visible(page, selectors: list[str], value: str, timeout_ms: int = 1500) -> tuple[bool, str]:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=600):
                loc.click(timeout=timeout_ms)
                try:
                    loc.fill("", timeout=timeout_ms)
                except Exception:
                    pass
                loc.fill(value, timeout=timeout_ms)
                return True, sel
        except Exception:
            continue
    return False, ""



def _focus_tpdc_captcha_field(page) -> dict:
    """Focus TPDC captcha input for manual entry; never OCR or auto-solve."""
    result = {"attempted": True, "focused": False, "selector": "", "note": ""}
    selectors = [
        "input[placeholder*='验证码']",
        "input[aria-label*='验证码']",
        "input[name*='captcha' i]",
        "input[id*='captcha' i]",
        "input[name*='code' i]",
        "input[id*='code' i]",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=500):
                loc.click(timeout=1200)
                result.update({"focused": True, "selector": sel, "note": "已聚焦验证码输入框；验证码内容由用户手动输入。"})
                return result
        except Exception:
            continue
    result["note"] = "未找到可聚焦的验证码输入框。"
    return result


def _visible_input_value_len(page, selectors: list[str]) -> int:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=300):
                val = loc.input_value(timeout=800)
                return len(str(val or '').strip())
        except Exception:
            continue
    return 0


def _try_click_tpdc_login_submit(page) -> dict:
    """Click login submit after the user manually enters captcha.

    This is a convenience action, not captcha bypass: the user supplies the code.
    """
    result = {"attempted": True, "clicked": False, "selector": "", "note": ""}
    selectors = [
        "button:has-text('登录')",
        "button:has-text('登 录')",
        "input[type='submit']",
        "a:has-text('登录')",
        "div:has-text('登录')",
    ]
    for sel in selectors:
        try:
            locs = page.locator(sel)
            count = min(locs.count(), 12)
            for i in range(count):
                loc = locs.nth(i)
                if not loc.is_visible(timeout=300):
                    continue
                try:
                    box = loc.bounding_box(timeout=500)
                except Exception:
                    box = None
                # Avoid clicking the top-bar login entry after the form is already open.
                if box and box.get('y', 9999) < 80:
                    continue
                text = ""
                try:
                    text = (loc.inner_text(timeout=500) or '').strip()
                except Exception:
                    pass
                if sel.startswith('div') and text not in {'登录', '登 录'}:
                    continue
                loc.click(timeout=1500)
                result.update({"clicked": True, "selector": f"{sel}[{i}]", "note": "检测到验证码已由用户输入，已点击登录按钮。"})
                return result
        except Exception:
            continue
    result["note"] = "验证码已输入，但未找到安全的登录按钮；请手动点击登录。"
    return result


def _wait_for_tpdc_manual_captcha_and_login(page, timeout_seconds: int = 600) -> dict:
    """Wait for manual captcha, optionally click Login, then wait until login form disappears."""
    started = time.time()
    timeout_seconds = max(0, int(timeout_seconds or 0))
    if timeout_seconds <= 0:
        return {"waited": False, "completed": False, "seconds": 0, "note": "未等待用户验证码。"}
    captcha_selectors = [
        "input[placeholder*='验证码']",
        "input[aria-label*='验证码']",
        "input[name*='captcha' i]",
        "input[id*='captcha' i]",
        "input[name*='code' i]",
        "input[id*='code' i]",
    ]
    focus_result = _focus_tpdc_captcha_field(page)
    submit_result = {"attempted": False, "clicked": False, "note": "尚未检测到用户输入验证码。"}
    auto_click = _is_truth("DOMESTIC_AUTO_CLICK_LOGIN_AFTER_CAPTCHA", "1")
    min_len = int(os.getenv("DOMESTIC_CAPTCHA_MIN_LENGTH", "4") or "4")
    clicked_once = False
    while time.time() - started < timeout_seconds:
        try:
            if not _page_has_visible_password_or_verification(page):
                return {
                    "waited": True,
                    "completed": True,
                    "seconds": int(time.time() - started),
                    "focus_result": focus_result,
                    "submit_result": submit_result,
                    "note": "登录/验证码界面已消失，继续自动检索。",
                }
            if auto_click and not clicked_once and _visible_input_value_len(page, captcha_selectors) >= min_len:
                submit_result = _try_click_tpdc_login_submit(page)
                clicked_once = bool(submit_result.get("clicked"))
        except Exception:
            pass
        try:
            page.wait_for_timeout(800)
        except Exception:
            time.sleep(0.8)
    return {
        "waited": True,
        "completed": False,
        "seconds": int(time.time() - started),
        "focus_result": focus_result,
        "submit_result": submit_result,
        "note": "等待用户输入验证码/登录超时；浏览器保持打开，自动流程会继续轮询登录状态。",
    }


def _force_tpdc_search_route(page, keyword: str) -> dict:
    """Hard-navigate TPDC to the data search route using ONLY the data name."""
    kw = str(keyword or '').strip()
    result = {"attempted": bool(kw), "submitted": False, "search_url": "", "note": "", "method": ""}
    if not kw:
        result["note"] = "无检索关键词。"
        return result
    urls = [
        "https://data.tpdc.ac.cn/allData?searchContent=" + quote(kw, safe=""),
        "https://data.tpdc.ac.cn/product?searchContent=" + quote(kw, safe=""),
    ]
    last_exc = ""
    for url in urls:
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            page.wait_for_timeout(2500)
            current = getattr(page, 'url', '') or ''
            if 'allData' in current or 'product' in current or quote(kw, safe='') in current:
                result.update({"submitted": True, "search_url": url, "method": "hard_route_goto", "note": "已强制进入 TPDC 检索路由。"})
                return result
        except Exception as exc:
            last_exc = str(exc)
            continue
    try:
        url = urls[0]
        page.evaluate("url => { window.location.href = url; }", url)
        page.wait_for_timeout(3500)
        result.update({"submitted": True, "search_url": url, "method": "window_location_href", "note": "已用 window.location.href 进入 TPDC 检索路由。"})
        return result
    except Exception as exc:
        result["note"] = f"TPDC 强制检索路由失败：{last_exc or exc}"
    return result



def _tpdc_page_context(page, keyword: str = "") -> dict:
    """Return coarse TPDC UI context without treating hidden SPA templates as state.

    V128: Previous versions could reach a TPDC result/dataset page, then the
    resume loop re-submitted a new search and the browser appeared to "flash"
    back to the home/initial page. This helper lets the state machine know when
    the current visible page is already a search/result/detail context, so it can
    continue selecting a dataset instead of re-navigating.
    """
    info = {"url": "", "has_keyword": False, "on_home": False, "on_search_or_dataset": False, "has_download_signal": False}
    kw = str(keyword or "").strip().lower()
    try:
        info["url"] = str(getattr(page, "url", "") or "")
    except Exception:
        pass
    url_low = info["url"].lower()
    # V137: NEVER judge page type by matching "data" in the full URL.
    # The TPDC domain itself is data.tpdc.ac.cn, so the old logic classified
    # almost every TPDC page as a dataset/search page and drove the state machine
    # into contradictory branches. Only path/query/fragment are used here.
    try:
        parsed = urlparse(info["url"] or "")
        path_low = (parsed.path or "/").lower()
        route_low = " ".join([path_low, parsed.query.lower(), parsed.fragment.lower()])
    except Exception:
        path_low = ""
        route_low = url_low
    is_home_path = path_low in {"", "/", "/home"} or path_low.endswith("/home")
    route_search_or_detail = any(x in route_low for x in ["alldata", "searchcontent", "product", "dataset", "accessdata", "detail", "downloadfile"])
    info["on_home"] = bool(is_home_path and not route_search_or_detail)
    try:
        body = (page.locator("body").inner_text(timeout=1200) or "")[:15000]
    except Exception:
        body = ""
    body_low = body.lower()
    info["has_keyword"] = bool(kw and (kw in body_low or kw in route_low))
    context_words = ["数据集", "数据资源", "数据文件", "数据下载", "下载", "ftp", "doi", "空间范围", "时间范围", "数据详情", "数据列表", "检索结果"]
    info["has_download_signal"] = any(w.lower() in body_low for w in ["ftp", "下载", "download", "数据文件", "申请下载", "加入订单"])
    info["on_search_or_dataset"] = bool(
        route_search_or_detail or
        (info["has_keyword"] and any(w in body for w in context_words))
    )
    return info


def _try_tpdc_native_search(page, keyword: str) -> dict:
    """Use TPDC's visible search box with ONLY the data name.

    V128: TPDC's router/direct allData URL is not stable enough on the user's
    machine. The correct automation path after login is the same as a human: put
    the parsed data name into the visible TPDC search box and press Enter. Do not
    append year/region/resolution.
    """
    result = {"attempted": False, "filled": False, "submitted": False, "note": "", "method": "tpdc_native_visible_search"}
    kw = str(keyword or "").strip()
    if not kw:
        result["note"] = "无检索关键词。"
        return result
    if _page_has_visible_password_or_verification(page):
        result["note"] = "当前仍是可见登录/验证码界面；不能向页面搜索框写入关键词。"
        return result
    result["attempted"] = True
    # Prefer the header search box. Keep selectors conservative to avoid login
    # fields; the captcha/password guard above is still the hard gate.
    selectors = [
        "input[placeholder*='回车']",
        "input[placeholder*='搜索']",
        "input[placeholder*='检索']",
        "input[placeholder*='关键词']",
        "input[type='search']",
        "input[name*='search' i]",
        "input[id*='search' i]",
        "input[name*='keyword' i]",
        "input[id*='keyword' i]",
    ]
    for sel in selectors:
        try:
            locs = page.locator(sel)
            count = min(locs.count(), 8)
            for i in range(count):
                loc = locs.nth(i)
                if not loc.is_visible(timeout=500):
                    continue
                # Reject controls that are too low inside content forms; TPDC's
                # global search is in the header/top strip.
                try:
                    box = loc.bounding_box(timeout=500)
                except Exception:
                    box = None
                if box and box.get("y", 9999) > 260:
                    continue
                loc.click(timeout=1200)
                try:
                    loc.fill("", timeout=800)
                except Exception:
                    try:
                        loc.press("Control+A", timeout=500)
                        loc.press("Backspace", timeout=500)
                    except Exception:
                        pass
                loc.fill(kw, timeout=1800)
                result["filled"] = True
                try:
                    loc.press("Enter", timeout=1500)
                    result["submitted"] = True
                except Exception:
                    # Fallback click the nearby/search icon if Enter is swallowed.
                    try:
                        page.locator("button:has-text('搜索'),button:has-text('检索'),.search,.search-btn,[class*='search']").first.click(timeout=1200)
                        result["submitted"] = True
                    except Exception:
                        pass
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=8000)
                except Exception:
                    pass
                page.wait_for_timeout(3500)
                result["selector"] = f"{sel}[{i}]"
                result["note"] = "已使用 TPDC 原生搜索框检索数据名称；未追加年份、区域、分辨率。"
                return result
        except Exception:
            continue
    # JS fallback: choose a visible top-area text/search input, excluding login,
    # password and captcha controls.
    try:
        js_result = page.evaluate(
            """
            (kw) => {
              const visible = (el) => {
                const st = window.getComputedStyle(el);
                const r = el.getBoundingClientRect();
                return st && st.display !== 'none' && st.visibility !== 'hidden' &&
                  Number(st.opacity || 1) !== 0 && r.width > 40 && r.height > 12 &&
                  r.top >= 0 && r.top < Math.min(280, window.innerHeight) &&
                  r.left >= 0 && r.left < window.innerWidth;
              };
              const attr = (el) => [el.type, el.name, el.id, el.placeholder, el.getAttribute('aria-label'), el.className]
                .filter(Boolean).join(' ').toLowerCase();
              const bad = (el) => /(密码|password|pass|验证码|captcha|code|sms|短信|邮箱|用户|账号|login|user)/i.test(attr(el));
              const good = (el) => /(回车|搜索|检索|关键词|数据|search|keyword|query)/i.test(attr(el));
              const inputs = Array.from(document.querySelectorAll('input,textarea')).filter(el => visible(el) && !bad(el) && good(el));
              if (!inputs.length) return {filled:false, submitted:false, note:'未找到可见 TPDC 搜索框。'};
              inputs.sort((a,b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);
              const el = inputs[0];
              el.focus();
              el.value = '';
              el.dispatchEvent(new Event('input', {bubbles:true}));
              el.value = kw;
              el.dispatchEvent(new Event('input', {bubbles:true}));
              el.dispatchEvent(new Event('change', {bubbles:true}));
              el.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', code:'Enter', keyCode:13, which:13, bubbles:true}));
              el.dispatchEvent(new KeyboardEvent('keyup', {key:'Enter', code:'Enter', keyCode:13, which:13, bubbles:true}));
              return {filled:true, submitted:true, note:'JS 已向 TPDC 顶部搜索框填入数据名称并触发 Enter。'};
            }
            """,
            kw,
        )
        if isinstance(js_result, dict):
            result.update(js_result)
            if result.get("submitted"):
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=8000)
                except Exception:
                    pass
                page.wait_for_timeout(3500)
                return result
    except Exception as exc:
        result["note"] = f"TPDC 原生搜索失败：{exc}"
        return result
    if not result.get("note"):
        result["note"] = "未找到可安全填入的数据检索框。"
    return result

def _try_fill_tpdc_login_form(page, username: str, password: str) -> dict:
    """TPDC-specific safe credential fill.

    It fills only when a password field is visible. Username selectors are
    restricted to login semantics such as 邮箱/用户名/账号; search/query fields and
    captcha fields are excluded by construction.
    """
    result = {
        "attempted": bool(username and password),
        "username_filled": False,
        "password_filled": False,
        "submitted": False,
        "note": "",
        "method": "tpdc_explicit_selectors",
    }
    if not username or not password:
        result["note"] = "TPDC_USERNAME 或 TPDC_PASSWORD 未配置，无法自动填充。"
        return result
    if not _wait_for_login_form(page, timeout_ms=int(os.getenv("DOMESTIC_LOGIN_FORM_WAIT_MS", "15000"))):
        result["note"] = "已点击登录入口，但 15 秒内未检测到可见密码框；拒绝误填搜索框。"
        return result

    user_selectors = [
        "input[placeholder*='邮箱']",
        "input[placeholder*='用户名']",
        "input[placeholder*='账号']",
        "input[placeholder*='用户']",
        "input[name*='user' i]",
        "input[id*='user' i]",
        "input[name*='email' i]",
        "input[id*='email' i]",
        "input[type='email']",
    ]
    pass_selectors = [
        "input[type='password']",
        "input[placeholder*='密码']",
        "input[name*='pass' i]",
        "input[id*='pass' i]",
    ]
    ok_user, user_sel = _fill_first_visible(page, user_selectors, username)
    ok_pass, pass_sel = _fill_first_visible(page, pass_selectors, password)
    result.update({
        "username_filled": ok_user,
        "password_filled": ok_pass,
        "username_selector": user_sel,
        "password_selector": pass_sel,
    })
    if ok_user and ok_pass:
        result["note"] = "已自动填充 TPDC 账号和密码；验证码仍需用户手动输入。"
    elif ok_pass and not ok_user:
        result["note"] = "已填充密码，但未找到明确的邮箱/用户名输入框；没有触碰搜索框。"
    else:
        result["note"] = "未能用 TPDC 显式选择器填充登录表单。"
    return result

def _try_fill_login_form(page, username: str, password: str) -> dict:
    """Fill login credentials only inside a detected login form.

    V124 keeps the observed TPDC login fix: after clicking 登录 the script must wait
    for the actual login page and fill explicit 邮箱/用户名 + 密码 fields. The
    generic DOM fallback remains guarded by a visible password field and still
    excludes search/captcha inputs.
    """
    result = {"attempted": False, "username_filled": False, "password_filled": False, "submitted": False, "note": ""}
    if not username or not password:
        result["note"] = "平台账号或密码未配置，跳过自动填充。"
        return result
    result["attempted"] = True
    explicit = _try_fill_tpdc_login_form(page, username, password)
    if explicit.get("username_filled") and explicit.get("password_filled"):
        return explicit
    try:
        fill_result = page.evaluate(
            """
            ([username, password]) => {
              const visible = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 1 && rect.height > 1 && !el.disabled && !el.readOnly;
              };
              const attrText = (el) => [el.name, el.id, el.placeholder, el.getAttribute('aria-label'), el.className].filter(Boolean).join(' ').toLowerCase();
              const isSearchOrCaptcha = (el) => /(搜索|检索|查询|关键词|keyword|search|query|验证码|captcha|code|sms|短信)/i.test(attrText(el));
              const isUserCandidate = (el) => {
                const type = (el.getAttribute('type') || 'text').toLowerCase();
                const txt = attrText(el);
                return el.tagName === 'INPUT' && ['text','email','tel','number',''].includes(type) && !isSearchOrCaptcha(el) && !/(password|pass|密码)/i.test(txt);
              };
              const setValue = (el, value) => {
                el.focus();
                el.value = value;
                el.dispatchEvent(new Event('input', {bubbles: true}));
                el.dispatchEvent(new Event('change', {bubbles: true}));
              };
              const passInputs = Array.from(document.querySelectorAll("input[type='password'], input[placeholder*='密码']")).filter(visible);
              if (!passInputs.length) {
                return {has_password_field: false, username_filled: false, password_filled: false, note: '未检测到可见密码框；拒绝向任何普通文本框写入账号，避免误填搜索框。'};
              }
              for (const pass of passInputs) {
                let root = pass.closest("form,[role='dialog'],.modal,.el-dialog,.ant-modal,.login,.login-box,.login-panel,.login-container,.user-login") || pass.parentElement;
                let guard = 0;
                while (root && root !== document.body && root.querySelectorAll('input').length < 2 && guard < 4) {
                  root = root.parentElement;
                  guard += 1;
                }
                root = root || document.body;
                const inputs = Array.from(root.querySelectorAll('input')).filter(visible);
                const user = inputs.find((el) => el !== pass && isUserCandidate(el));
                if (!user) continue;
                setValue(user, username);
                setValue(pass, password);
                return {has_password_field: true, username_filled: true, password_filled: true, note: '已在同一登录容器内填充账号和密码；未触碰搜索框。'};
              }
              return {has_password_field: true, username_filled: false, password_filled: false, note: '检测到密码框，但未找到同一登录容器内的账号框；拒绝误填。'};
            }
            """,
            [username, password],
        )
        if isinstance(fill_result, dict):
            result.update(fill_result)
    except Exception as exc:
        result["note"] = f"登录表单安全填充失败：{exc}"
        return result

    # Do not force-submit when captcha fields are present. The user should verify.
    try:
        body = (page.locator("body").inner_text(timeout=1000) or "")[:5000]
        if any(k in body for k in ["验证码", "人机", "拖动", "实名", "短信"]):
            result["note"] = (result.get("note") or "") + " 页面包含验证码/人机/实名提示，等待用户手动完成。"
            return result
    except Exception:
        pass
    if result.get("username_filled") and result.get("password_filled"):
        for sel in ["button:has-text('登录')", "button:has-text('登陆')", "input[type='submit']", "button:has-text('Sign in')", "button:has-text('Login')"]:
            try:
                loc = page.locator(sel).first
                if loc.count() and loc.is_visible(timeout=500):
                    loc.click(timeout=1500)
                    result["submitted"] = True
                    page.wait_for_timeout(1500)
                    break
            except Exception:
                continue
    if not result.get("note"):
        result["note"] = "已安全尝试平台登录。"
    return result


def _try_search_keyword(page, keyword: str) -> dict:
    """Search TPDC without forcing login.

    V124 uses TPDC's search-result route directly first. This avoids the fragile
    home-page search box and avoids the old failure mode where a hidden login
    template in the SPA made the script believe it was on a captcha page.
    """
    result = {"attempted": False, "filled": False, "submitted": False, "note": "", "method": ""}
    kw = str(keyword or "").strip()
    if not kw:
        result["note"] = "无检索关键词。"
        return result

    # Only a truly visible login page blocks search. Hidden DOM templates no longer count.
    if _page_has_visible_password_or_verification(page):
        result["note"] = "当前可见界面确实是登录/验证码页面；先关闭登录弹窗或返回首页后再检索。"
        return result

    # V128: use TPDC's native visible search first. Direct allData URLs can
    # briefly enter a result route and then bounce back to /home on this site,
    # which is the user's observed "闪一下又回到初始界面".
    if _is_truth("DOMESTIC_TPDC_USE_NATIVE_SEARCH", "1"):
        native_result = _try_tpdc_native_search(page, kw)
        if native_result.get("submitted"):
            result.update(native_result)
            return result
        result["note"] = native_result.get("note") or "TPDC 原生搜索未成功，转入备用策略。"

    # Direct route remains as a disabled fallback for diagnostics only.
    if _is_truth("DOMESTIC_TPDC_DIRECT_SEARCH_URL", "0"):
        route_result = _force_tpdc_search_route(page, kw)
        if route_result.get("submitted"):
            result.update({
                "attempted": True,
                "filled": True,
                "submitted": True,
                "method": route_result.get("method") or "direct_allData_searchContent_url",
                "search_url": route_result.get("search_url", ""),
                "note": "已直接进入 TPDC 数据检索结果页；未触发登录页。",
            })
            return result
        result["note"] = route_result.get("note") or "TPDC 直接检索 URL 未成功，转入页面搜索框 fallback。"

    result["attempted"] = True
    selectors = [
        "input[placeholder*='回车']", "input[placeholder*='搜索']", "input[placeholder*='检索']", "input[placeholder*='关键词']",
        "input[placeholder*='数据']", "input[type='search']", "input[name*='keyword']",
        "input[id*='keyword']", "input[name*='search']", "input[id*='search']",
        "textarea[placeholder*='搜索']", "textarea[placeholder*='检索']",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=900):
                loc.click(timeout=1200)
                try:
                    loc.fill("", timeout=1200)
                except Exception:
                    pass
                loc.fill(kw, timeout=1800)
                result["filled"] = True
                try:
                    loc.press("Enter", timeout=1500)
                    result["submitted"] = True
                except Exception:
                    pass
                page.wait_for_load_state("domcontentloaded", timeout=8000)
                page.wait_for_timeout(2500)
                result["method"] = "visible_search_input"
                result["note"] = "已在 TPDC 搜索框输入关键词并回车检索。"
                return result
        except Exception:
            continue
    # JS fallback: choose the most likely visible search input. This still
    # excludes password/captcha fields and login form fields.
    try:
        js_result = page.evaluate(
            """
            (kw) => {
              const visible = (el) => {
                const st = window.getComputedStyle(el);
                const r = el.getBoundingClientRect();
                return st && st.display !== 'none' && st.visibility !== 'hidden' && r.width > 60 && r.height > 18 && !el.disabled && !el.readOnly;
              };
              const attr = (el) => [el.name, el.id, el.placeholder, el.className, el.getAttribute('aria-label')].filter(Boolean).join(' ');
              const bad = (el) => /(密码|验证码|captcha|code|sms|user|login|账号|邮箱|phone|mobile)/i.test(attr(el)) || (el.type || '').toLowerCase() === 'password';
              const good = (el) => /(回车|搜索|检索|关键词|数据|search|keyword|query)/i.test(attr(el));
              const inputs = Array.from(document.querySelectorAll('input,textarea')).filter(el => visible(el) && !bad(el));
              let cand = inputs.find(good) || inputs.sort((a,b)=>b.getBoundingClientRect().width-a.getBoundingClientRect().width)[0];
              if (!cand) return {filled:false, submitted:false, note:'JS fallback 未找到可用搜索框'};
              cand.focus(); cand.value = kw;
              cand.dispatchEvent(new Event('input', {bubbles:true}));
              cand.dispatchEvent(new Event('change', {bubbles:true}));
              cand.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', code:'Enter', keyCode:13, which:13, bubbles:true}));
              cand.dispatchEvent(new KeyboardEvent('keyup', {key:'Enter', code:'Enter', keyCode:13, which:13, bubbles:true}));
              return {filled:true, submitted:true, note:'JS fallback 已填写搜索框并触发 Enter', placeholder:cand.placeholder || '', id:cand.id || '', name:cand.name || ''};
            }
            """,
            kw,
        )
        if isinstance(js_result, dict) and js_result.get("filled"):
            page.wait_for_load_state("domcontentloaded", timeout=8000)
            page.wait_for_timeout(2500)
            result.update(js_result)
            result["method"] = "js_visible_search_input"
            return result
    except Exception as exc:
        result["note"] = f"搜索框 JS fallback 失败：{exc}"
    result["note"] = result.get("note") or "未找到 TPDC 搜索框；已打开平台页面，但未完成自动检索。"
    return result

def _extract_resolution_meters_from_text(text: str) -> list[float]:
    vals: list[float] = []
    body = str(text or "")
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:km|KM|Km|公里|千米)", body):
        try:
            vals.append(float(m.group(1)) * 1000.0)
        except Exception:
            pass
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:m|M|米)", body):
        try:
            vals.append(float(m.group(1)))
        except Exception:
            pass
    return [v for v in vals if 0 < v <= 100000]


def _whole_ascii_term_hit(text: str, term: str) -> bool:
    if not term:
        return False
    if re.fullmatch(r"[A-Za-z0-9_+.-]+", term):
        return bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", text, flags=re.I))
    return term.lower() in text.lower()


def _temporal_year_match(text: str, requested_year: int | None) -> dict:
    """Accept either an exact year or a multi-year range covering the target year.

    Example: a user asks for 2020, and the page says 1985-2024. That is valid:
    the dataset can be downloaded and later filtered to 2020.
    """
    if not requested_year:
        return {"needed": False, "matched": True, "reason": "no_requested_year"}
    body = str(text or "")
    y = int(requested_year)
    ranges = []
    for m in re.finditer(r"(19\d{2}|20\d{2})\s*(?:-|–|—|至|到|~|－)\s*(19\d{2}|20\d{2})", body):
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = min(a, b), max(a, b)
        ranges.append((lo, hi))
        if lo <= y <= hi:
            return {"needed": True, "matched": True, "reason": f"range_covers_year:{lo}-{hi}", "ranges": ranges}
    years = [int(v) for v in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", body)]
    if y in years:
        return {"needed": True, "matched": True, "reason": f"exact_year:{y}", "years": sorted(set(years))}
    if years or ranges:
        return {"needed": True, "matched": False, "reason": f"year_not_covered:{y}", "years": sorted(set(years)), "ranges": ranges}
    return {"needed": True, "matched": None, "reason": "year_unknown_on_page"}


def _dataset_text_score(text: str, data_name: str, requested_resolution_meters: float | None = None, requested_year: int | None = None) -> dict:
    raw = str(text or "")
    low = raw.lower()
    reject_terms = ["语义分割", "图像分割", "semantic segmentation", "segmentation", "deeplab", "标注样本", "训练样本", "样本库"]
    if any(t.lower() in low for t in reject_terms):
        return {"ok": False, "score": -999, "reason": "rejected_semantic_segmentation_or_training_sample"}
    score = 0
    reasons: list[str] = []
    dn_raw = str(data_name or "").strip()
    dn = dn_raw.upper()
    if dn == "DEM":
        if _whole_ascii_term_hit(raw, "DEM"):
            score += 100; reasons.append("whole_word_dem")
        for term in ["数字高程模型", "数字高程", "高程模型", "SRTM", "ASTER GDEM", "GDEM"]:
            if _whole_ascii_term_hit(raw, term):
                score += 60; reasons.append(f"term:{term}")
        if _whole_ascii_term_hit(raw, "地形"):
            score += 30; reasons.append("term:地形")
        if score == 0 and _whole_ascii_term_hit(raw, "EM"):
            return {"ok": False, "score": -50, "reason": "em_is_not_dem"}
    else:
        # V117: 通用数据类型评分。用户可能要 NDVI、土地利用、土壤水分、气象、遥感影像等，不再只服务 DEM。
        generic_terms = [dn_raw]
        extra_terms_by_name = {
            "NDVI": ["NDVI", "植被指数", "归一化植被指数"],
            "EVI": ["EVI", "增强植被指数"],
            "LAI": ["LAI", "叶面积指数"],
            "FVC": ["FVC", "植被覆盖度", "植被覆盖率"],
            "LULC": ["土地利用", "土地覆盖", "LUCC", "CLCD", "地表覆盖", "LULC"],
            "NPP": ["NPP", "净初级生产力"],
            "降水": ["降水", "降雨", "precipitation", "rainfall"],
            "气温": ["气温", "温度", "temperature", "LST", "地表温度"],
            "气象": ["气象", "气温", "降水", "风速", "湿度", "蒸散"],
            "土壤水分": ["土壤水分", "土壤湿度", "soil moisture"],
            "土壤": ["土壤", "土壤属性", "有机质", "pH", "质地"],
            "水文": ["水文", "径流", "河流", "水系", "湖泊", "冰川", "积雪"],
            "遥感影像": ["遥感影像", "Landsat", "Sentinel", "MODIS", "高分", "影像"],
        }
        generic_terms += extra_terms_by_name.get(dn_raw, [])
        hit_terms = []
        for term in sorted({t for t in generic_terms if t}, key=len, reverse=True):
            if _whole_ascii_term_hit(raw, term):
                hit_terms.append(term)
        if hit_terms:
            score += 80 + min(len(hit_terms), 3) * 20
            reasons.append("term:" + "/".join(hit_terms[:3]))
    if any(w in raw for w in ["数据集", "数据", "产品", "下载", "FTP", "详情", "申请", "订购"]):
        score += 20; reasons.append("dataset_context")
    temporal = _temporal_year_match(raw, requested_year)
    if temporal.get("matched") is True and temporal.get("needed"):
        score += 30; reasons.append(str(temporal.get("reason")))
    elif temporal.get("matched") is False:
        return {"ok": False, "score": -180, "reason": str(temporal.get("reason")), "temporal": temporal}
    elif temporal.get("matched") is None:
        reasons.append("year_unknown_allowed_pending_metadata_check")
    resolutions = _extract_resolution_meters_from_text(raw)
    if requested_resolution_meters:
        if resolutions:
            best = min(resolutions)
            if best <= float(requested_resolution_meters) + 1e-6:
                score += 35; reasons.append(f"resolution_ok:{best:g}m")
            else:
                return {"ok": False, "score": -200, "reason": f"resolution_too_coarse:{best:g}m>{requested_resolution_meters:g}m", "resolutions": resolutions}
        else:
            reasons.append("resolution_unknown")
    return {"ok": score >= 80, "score": score, "reason": "+".join(reasons) or "no_required_data_term", "resolutions": resolutions, "temporal": temporal}



def _try_open_best_dataset_or_download(page, keyword: str, data_name: str = "", requested_resolution_meters: float | None = None, requested_year: int | None = None) -> dict:
    """Open the best matching dataset/product/download entry after search.

    V117 rule: do not click a result merely because it contains "EM" or a loose
    substring. DEM requires a real DEM / digital elevation / terrain signal, and
    requested resolution is a hard upper bound on pixel size.
    """
    result = {"attempted": True, "clicked": False, "selector": "", "note": "", "score": 0}
    if _page_has_visible_password_or_verification(page):
        result["note"] = "当前处于登录/验证码页，不能安全打开数据结果；等待下载阶段登录完成后继续。"
        return result

    best = {"score": -999, "text": "", "index": -1, "tag": "", "reason": ""}
    try:
        candidates = page.locator("a,button,[role='button']")
        count = min(candidates.count(), 200)
        for i in range(count):
            try:
                loc = candidates.nth(i)
                if not loc.is_visible(timeout=250):
                    continue
                txt = (loc.inner_text(timeout=500) or "").strip()
                if not txt or len(txt) > 300:
                    continue
                sc = _dataset_text_score(txt, data_name or keyword, requested_resolution_meters, requested_year)
                if sc.get("score", -999) > best["score"]:
                    best = {"score": int(sc.get("score", 0)), "text": txt[:200], "index": i, "tag": "a,button,[role='button']", "reason": sc.get("reason", "")}
            except Exception:
                continue
        if best["index"] >= 0 and best["score"] >= 80:
            loc = candidates.nth(best["index"])
            loc.scroll_into_view_if_needed(timeout=800)
            loc.click(timeout=1500)
            page.wait_for_load_state("domcontentloaded", timeout=8000)
            page.wait_for_timeout(1200)
            result.update({"clicked": True, "selector": f"{best['tag']} nth={best['index']}", "note": "已按数据类型与分辨率规则进入最相关的数据/下载页面。", "score": best["score"], "matched_text": best["text"], "match_reason": best["reason"]})
            return result
    except Exception as exc:
        result["note"] = f"候选数据结果评分失败：{exc}"
        return result
    # V124 fallback: TPDC result pages often render cards where the meaningful
    # dataset title/metadata are on a parent div, not on the exact <a> text. Score
    # visible card-like containers, then click their inner link/button.
    try:
        card_candidates = page.locator("li,.el-card,.card,.list-item,.data-item,.dataset-item,div[class*='item'],div[class*='card'],div[class*='dataset']")
        card_count = min(card_candidates.count(), 260)
        card_best = {"score": -999, "text": "", "index": -1, "reason": ""}
        for i in range(card_count):
            try:
                loc = card_candidates.nth(i)
                if not loc.is_visible(timeout=200):
                    continue
                txt = (loc.inner_text(timeout=500) or "").strip()
                if not txt or len(txt) < 4 or len(txt) > 1600:
                    continue
                sc = _dataset_text_score(txt, data_name or keyword, requested_resolution_meters, requested_year)
                if sc.get("score", -999) > card_best["score"]:
                    card_best = {"score": int(sc.get("score", 0)), "text": txt[:400], "index": i, "reason": sc.get("reason", "")}
            except Exception:
                continue
        if card_best["index"] >= 0 and card_best["score"] >= 80:
            loc = card_candidates.nth(card_best["index"])
            loc.scroll_into_view_if_needed(timeout=1000)
            inner = loc.locator("a,button,[role='button']").first
            try:
                if inner.count() and inner.is_visible(timeout=500):
                    inner.click(timeout=1800)
                else:
                    loc.click(timeout=1800)
            except Exception:
                loc.click(timeout=1800, force=True)
            page.wait_for_load_state("domcontentloaded", timeout=10000)
            page.wait_for_timeout(1500)
            result.update({
                "clicked": True,
                "selector": f"card_like nth={card_best['index']}",
                "note": "已按卡片全文评分进入最相关数据集页面。",
                "score": card_best["score"],
                "matched_text": card_best["text"],
                "match_reason": card_best["reason"],
            })
            return result
        best["card_best"] = card_best
    except Exception as exc:
        best["card_fallback_error"] = str(exc)

    result.update({"note": "未能可靠识别符合数据类型与分辨率约束的数据结果；不会误点含 EM 的非 DEM 数据。", "best_candidate": best})
    return result


def _tpdc_visible_text(page, limit: int = 16000) -> str:
    """Best-effort visible body text for TPDC state checks."""
    try:
        return (page.locator("body").inner_text(timeout=1200) or "")[:limit]
    except Exception:
        return ""


def _tpdc_is_search_result_page(page, keyword: str = "") -> bool:
    """Detect TPDC search/list page where sorting links such as 下载量 exist.

    V132 deliberately keeps the V130 post-login/search/dataset-opening flow, but
    forbids the old broad download click on search-result lists. The previous
    broad selector could click “下载量” instead of a real download action, causing
    TPDC to route back to the initial page.
    """
    try:
        url = str(getattr(page, "url", "") or "").lower()
    except Exception:
        url = ""
    text = _tpdc_visible_text(page, 20000)
    kw = str(keyword or "").strip()
    has_result_word = any(w in text for w in ["查询结果", "检索结果", "搜索结果", "排序", "下载量", "浏览量", "更新时间"])
    has_kw = bool(kw and kw in text)
    route_like = any(x in url for x in ["searchcontent", "alldata", "product"])
    detail_words = ["FTP 账号", "FTP账号", "数据文件", "文件列表", "下载地址", "数据引用", "引用方式", "数据详情"]
    has_detail_signal = any(w in text for w in detail_words)
    return bool((has_result_word or route_like) and (has_kw or route_like) and not has_detail_signal)


def _try_auto_click_download(page, keyword: str = "") -> dict:
    """Click only real TPDC download/FTP actions, never sorting/navigation links.

    This function fixes the V130 flash-back failure without reintroducing V131's
    overly aggressive freeze. Search and dataset selection are still allowed; the
    only forbidden action is clicking ambiguous result-list controls such as
    下载量/浏览量/更新时间/查看更多/首页.
    """
    result = {"attempted": True, "clicked": False, "selector": "", "note": ""}
    if _page_has_visible_password_or_verification(page):
        result["note"] = "页面包含登录/验证码/实名流程，等待用户手动处理或自动填充后再点击。"
        return result
    if _is_truth("DOMESTIC_TPDC_DOWNLOAD_CLICK_ONLY_ON_DETAIL", "1") and _tpdc_is_search_result_page(page, keyword):
        result["note"] = "当前是 TPDC 检索结果列表；不会点击“下载量/浏览量/更新时间”等排序链接。请先进入具体数据集，或等待系统按结果卡片评分进入数据集。"
        return result
    try:
        body = _tpdc_visible_text(page, 8000)
        if any(k in body for k in ["阅读并同意", "协议确认"]):
            result["note"] = "页面包含协议确认，等待用户手动处理。"
            return result
    except Exception:
        pass

    exact_actions = {"FTP", "FTP下载", "FTP 下载", "下载", "免费下载", "申请下载", "加入订单", "提交订单", "获取数据", "数据下载"}
    reject_patterns = re.compile(r"下载量|浏览量|更新时间|排序|显示|更多|查看更多|首页|产品|新闻|汇交")
    try:
        candidates = page.locator("a,button,[role='button'],input[type='button'],input[type='submit']")
        count = min(candidates.count(), 180)
        normalized_actions = {re.sub(r"\s+", "", x) for x in exact_actions}
        for i in range(count):
            try:
                loc = candidates.nth(i)
                if not loc.is_visible(timeout=350):
                    continue
                txt = ""
                try:
                    txt = (loc.inner_text(timeout=500) or "").strip()
                except Exception:
                    pass
                if not txt:
                    try:
                        txt = (loc.get_attribute("value", timeout=300) or "").strip()
                    except Exception:
                        txt = ""
                norm = re.sub(r"\s+", "", txt)
                spaced = re.sub(r"\s+", " ", txt).strip()
                if not txt:
                    continue
                if reject_patterns.search(norm):
                    continue
                if norm not in normalized_actions and spaced not in exact_actions:
                    continue
                try:
                    box = loc.bounding_box(timeout=500)
                except Exception:
                    box = None
                if box and box.get("y", 0) < 120 and norm not in {"FTP", "FTP下载", "下载"}:
                    continue
                loc.scroll_into_view_if_needed(timeout=800)
                loc.click(timeout=1800)
                page.wait_for_timeout(1800)
                result.update({"clicked": True, "selector": f"exact_action nth={i}", "text": txt, "note": "已点击具体数据集页面中的真实下载/FTP动作按钮；已排除下载量等排序链接。"})
                return result
            except Exception:
                continue
    except Exception as exc:
        result["note"] = f"扫描下载按钮失败：{exc}"
        return result
    result["note"] = "未发现安全的真实下载/FTP按钮；不会点击疑似排序或导航链接。"
    return result

def _page_has_dataset_signal(page, keyword: str, open_result: dict | None = None, click_result: dict | None = None, ftp_accounts: list | None = None, data_name: str = "", requested_resolution_meters: float | None = None, requested_year: int | None = None, platform: str = "") -> dict:
    """Decide whether this candidate platform is worth showing to the user.

    V117 adds strict data-type and resolution checks. For DEM, EM is not DEM;
    semantic-segmentation/training datasets are rejected even if the text happens
    to contain the letters E and M.
    """
    if ftp_accounts:
        return {"matched": True, "reason": "ftp_credentials_detected"}
    if click_result and click_result.get("clicked"):
        return {"matched": True, "reason": "download_or_order_button_clicked"}
    if open_result and open_result.get("clicked"):
        return {"matched": True, "reason": "dataset_or_product_entry_opened", "score": open_result.get("score"), "match_reason": open_result.get("match_reason")}
    try:
        body = (page.locator("body").inner_text(timeout=1200) or "")[:12000]
    except Exception as exc:
        return {"matched": False, "reason": f"body_read_failed:{exc}"}
    body_low = body.lower()
    negative_words = ["暂无数据", "无相关数据", "未找到", "没有找到", "搜索结果为空", "0条", "无结果", "not found", "no result"]
    if any(w.lower() in body_low for w in negative_words):
        return {"matched": False, "reason": "negative_result_text_detected"}
    sc = _dataset_text_score(body, data_name or keyword, requested_resolution_meters, requested_year)
    if sc.get("ok"):
        return {"matched": True, "reason": "strict_dataset_text_detected", "score": sc.get("score"), "match_reason": sc.get("reason"), "resolutions": sc.get("resolutions"), "temporal": sc.get("temporal")}
    # Do not abandon TPDC at the login gate; this is the preferred FTP platform.
    if platform == "tpdc" and any(w in body for w in ["请登录", "登录后", "用户登录", "验证码", "人机", "实名", "账号", "密码"]):
        return {"matched": True, "reason": "tpdc_login_required_first", "score": sc.get("score"), "match_reason": sc.get("reason")}
    return {"matched": False, "reason": sc.get("reason") or "no_reliable_dataset_signal", "score": sc.get("score"), "resolutions": sc.get("resolutions"), "temporal": sc.get("temporal")}


def _safe_live_screenshot(page, live_screenshot: Path) -> bool:
    """Write projection screenshot atomically to avoid half-written PNG flicker."""
    tmp = live_screenshot.with_name(live_screenshot.stem + ".tmp" + live_screenshot.suffix)
    page.screenshot(path=str(tmp), full_page=False)
    tmp.replace(live_screenshot)
    return True




def _tpdc_fixed_ftp_hosts() -> tuple[str, str, int]:
    primary = os.getenv("TPDC_FTP_PRIMARY_HOST", "ftp2.tpdc.ac.cn").strip() or "ftp2.tpdc.ac.cn"
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

def _normalize_tpdc_fixed_host_accounts(accounts: list[dict]) -> list[dict]:
    valid = None
    for a in accounts or []:
        u = str((a or {}).get("username") or "").strip()
        pw = str((a or {}).get("password") or "").strip()
        if _is_valid_tpdc_username(u) and _is_valid_tpdc_password(pw):
            valid = dict(a)
            valid["username"] = u
            valid["password"] = pw
            break
    if not valid:
        return []
    primary_host, _backup_host, fixed_port = _tpdc_fixed_ftp_hosts()
    b = dict(valid)
    b.update({
        "host": primary_host,
        "port": int(fixed_port or 6201),
        "priority": 1,
        "primary": True,
        "host_role": "primary",
        "fixed_host_policy": "tpdc_single_primary_fixed_host",
    })
    return [b]

def _collect_ftp_text_from_page_and_frames(page) -> str:
    """Safely collect FTP credential text from the current TPDC page and frames only.

    This is intentionally narrower than V133: it does not iterate every browser
    tab or switch pages, because that made the login/search state unstable on the
    user's machine. It still captures visible text, input values and copy-button
    attributes from the current active page, which covers TPDC FTP account blocks.
    """
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
          return st && st.display !== 'none' && st.visibility !== 'hidden' &&
                 Number(st.opacity || 1) !== 0 && r.width > 0 && r.height > 0;
        } catch(e) { return true; }
      };
      push(document.body ? document.body.innerText : '');
      push(document.body ? document.body.textContent : '');
      Array.from(document.querySelectorAll('input, textarea')).forEach((el) => {
        push(el.value);
        push(el.getAttribute('placeholder'));
      });
      Array.from(document.querySelectorAll('[title], [aria-label], [data-clipboard-text], [data-copy], [data-value], [value], a[href]')).forEach((el) => {
        ['title','aria-label','data-clipboard-text','data-copy','data-value','value','href'].forEach(attr => push(el.getAttribute(attr)));
      });
      Array.from(document.querySelectorAll('button, a, span, div, p, td, th, li, label')).filter(isVisible).forEach((el) => {
        push(el.innerText || el.textContent);
      });
      return Array.from(new Set(out)).join('\n');
    }
    """
    parts = []
    targets = []
    try:
        targets.append(page)
    except Exception:
        pass
    try:
        targets.extend(list(getattr(page, 'frames', []) or []))
    except Exception:
        pass
    for target in targets:
        try:
            val = target.evaluate(js) or ""
            if val:
                parts.append(str(val))
        except Exception:
            continue
    return "\n".join(parts)


def _extract_ftp_accounts_from_dom(page) -> list[dict]:
    """Extract FTP accounts from current page text, frame text and copy attributes."""
    text = _collect_ftp_text_from_page_and_frames(page)
    return _extract_ftp_accounts_from_text(text)


def _extract_ftp_accounts(page) -> list[dict]:
    texts = []
    try:
        texts.append(page.locator("body").inner_text(timeout=2500) or "")
    except Exception:
        pass
    try:
        texts.append(page.evaluate("() => document.body ? document.body.innerText : ''") or "")
    except Exception:
        pass
    accounts: list[dict] = []
    for text in texts:
        accounts.extend(_extract_ftp_accounts_from_text(text))
    accounts.extend(_extract_ftp_accounts_from_dom(page))
    dedup = []
    seen = set()
    for a in accounts:
        key = (str(a.get("host") or "").lower(), int(a.get("port") or 21), str(a.get("username") or ""), str(a.get("password") or ""))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        dedup.append(a)
    return dedup


def _select_primary_ftp_account(accounts: list[dict]) -> dict:
    """Return the FTP account that should be used first for auto-download.

    TPDC often exposes two hosts in one block. The first visible host is the
    primary choice unless DOMESTIC_FTP_PREFER_FIRST_HOST is explicitly disabled.
    """
    if not accounts:
        return {}
    if str(os.getenv("DOMESTIC_FTP_PREFER_FIRST_HOST", "1")).strip().lower() in {"1", "true", "yes", "on"}:
        return accounts[0]
    return accounts[0]


def _write_ftp_handoff_files(download_dir: Path, accounts: list[dict]) -> dict:
    download_dir.mkdir(parents=True, exist_ok=True)
    accounts = _normalize_tpdc_fixed_host_accounts(accounts)
    # Persist one fixed primary FTP account only.
    for i, a in enumerate(accounts):
        a.setdefault("priority", i + 1)
        if i == 0:
            a.setdefault("primary", True)
    manifest = download_dir / "ftp_accounts_detected.json"
    manifest.write_text(json.dumps(accounts, ensure_ascii=False, indent=2), encoding="utf-8")
    primary = _select_primary_ftp_account(accounts)
    host = primary.get("host", "")
    port = int(primary.get("port") or 21)
    username = primary.get("username", "anonymous")
    password = primary.get("password", "")
    ftp_url = ""
    if host:
        auth = quote(str(username), safe="")
        if password:
            auth += ":" + quote(str(password), safe="")
        ftp_url = f"ftp://{auth}@{host}:{port}/"
    (download_dir / "ftp_download_note.txt").write_text(
        "检测到平台页面提供 FTP 下载账号。\n"
        f"主机: {host}\n端口: {port}\n用户名: {username}\n密码: {password}\n\n"
        "已按页面显示顺序选择第一个 FTP 主机作为自动下载主机。\n"
        "触发时机：页面出现 FTP账号 / 主机 / 端口 / 用户名 / 密码 后自动触发。\n"
        "若 Python 自动 FTP 下载没有开始，可用 FTP Rush / WinSCP / FileZilla 打开以上账号。\n"
        "注意：大文件可能下载数小时到数天。\n",
        encoding="utf-8",
    )
    winscp_script_path = ""
    winscp_bat_path = ""
    if ftp_url:
        bat = download_dir / "open_ftp_client.bat"
        bat.write_text(f"@echo off\r\nstart \"\" \"{ftp_url}\"\r\n", encoding="utf-8")
        winscp_script = download_dir / "winscp_download_script.txt"
        local_root = str(download_dir / "ftp_downloads" / _safe_name(host))
        winscp_script.write_text(
            "option batch abort\n"
            "option confirm off\n"
            f"open ftp://{quote(str(username), safe='')}:{quote(str(password), safe='')}@{host}:{port}/\n"
            f"lcd \"{local_root}\"\n"
            "get * -transfer=binary -resume\n"
            "exit\n",
            encoding="utf-8",
        )
        winscp_bat = download_dir / "open_winscp_download.bat"
        winscp_bat.write_text(
            "@echo off\r\n"
            "where winscp.com >nul 2>nul\r\n"
            "if errorlevel 1 (\r\n"
            "  echo WinSCP not found in PATH. Please install WinSCP or use ftp_download_note.txt manually.\r\n"
            "  pause\r\n"
            "  exit /b 1\r\n"
            ")\r\n"
            f"winscp.com /script=\"{winscp_script}\"\r\n"
            "pause\r\n",
            encoding="utf-8",
        )
        winscp_script_path = str(winscp_script)
        winscp_bat_path = str(winscp_bat)
    return {
        "accounts_path": str(manifest),
        "primary_ftp_url": ftp_url,
        "note_path": str(download_dir / "ftp_download_note.txt"),
        "winscp_script_path": winscp_script_path,
        "winscp_bat_path": winscp_bat_path,
    }


def _ftp_download_tree(account: dict, download_dir: Path, max_files: int = 9999) -> dict:
    """Try to start a real FTP transfer from the account root.

    Many domestic platforms create a temporary FTP account whose root directory
    is already scoped to a selected data/order. In that case recursive download
    from root is acceptable. If the server denies listing or the root is too
    complex, this function records the FTP handoff and leaves the browser open.
    """
    host = str(account.get("host") or "").strip()
    if not host:
        return {"ok": False, "status": "no_host"}
    port = int(account.get("port") or 21)
    username = str(account.get("username") or "anonymous")
    password = str(account.get("password") or "")
    out_root = download_dir / "ftp_downloads" / _safe_name(host)
    out_root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    errors = []
    started = False

    def rec_download(ftp, remote_dir: str, local_dir: Path):
        nonlocal started
        if len(downloaded) >= max_files:
            return
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            entries = ftp.nlst(remote_dir)
        except Exception as exc:
            errors.append({"remote_dir": remote_dir, "error": f"nlst_failed:{exc}"})
            return
        for entry in entries:
            if len(downloaded) >= max_files:
                break
            name = str(entry).rstrip("/").split("/")[-1]
            if not name or name in {".", ".."}:
                continue
            remote_path = entry if str(entry).startswith("/") else (remote_dir.rstrip("/") + "/" + name)
            local_path = local_dir / _safe_name(name, 180)
            try:
                # Try directory first.
                cur = ftp.pwd()
                try:
                    ftp.cwd(remote_path)
                    ftp.cwd(cur)
                    rec_download(ftp, remote_path, local_path)
                    continue
                except Exception:
                    try:
                        ftp.cwd(cur)
                    except Exception:
                        pass
                with local_path.open("wb") as f:
                    started = True
                    ftp.retrbinary(f"RETR {remote_path}", f.write)
                downloaded.append({"remote": remote_path, "local": str(local_path), "bytes": local_path.stat().st_size})
            except Exception as exc:
                errors.append({"remote": remote_path, "error": str(exc)})

    try:
        ftp = ftplib.FTP()
        ftp.connect(host, port, timeout=int(os.getenv("DOMESTIC_FTP_CONNECT_TIMEOUT", "30")))
        ftp.login(username, password)
        ftp.set_pasv(True)
        rec_download(ftp, "/", out_root)
        try:
            ftp.quit()
        except Exception:
            pass
        return {
            "ok": True,
            "status": "ftp_download_started" if started else "ftp_connected_no_file_downloaded",
            "host": host,
            "port": port,
            "username": username,
            "download_root": str(out_root),
            "downloaded_count": len(downloaded),
            "downloaded": downloaded[:50],
            "errors": errors[:50],
        }
    except Exception as exc:
        return {"ok": False, "status": "ftp_connect_or_download_failed", "host": host, "port": port, "username": username, "error": str(exc), "download_root": str(out_root)}

def _safe_name(name: str, max_len: int = 120) -> str:
    import re
    name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", name or "download")
    return (name.strip("._-") or "download")[:max_len]


def _default_root() -> Path:
    # tools/ 在项目根/tools 下。
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_exe_from_command(command: str) -> str:
    c = str(command or "").strip()
    if not c:
        return ""
    m = re.match(r'"([^"]+\.exe)"', c, flags=re.I)
    if m:
        return m.group(1)
    m = re.match(r'([^\s]+\.exe)', c, flags=re.I)
    return m.group(1) if m else ""


def _windows_default_browser_executable() -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg  # type: ignore
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice") as key:
            progid, _ = winreg.QueryValueEx(key, "ProgId")
        for root in [winreg.HKEY_CLASSES_ROOT, winreg.HKEY_CURRENT_USER]:
            try:
                with winreg.OpenKey(root, str(progid) + r"\shell\open\command") as key:
                    cmd, _ = winreg.QueryValueEx(key, None)
                    exe = _extract_exe_from_command(cmd)
                    if exe and Path(exe).exists():
                        return exe
            except Exception:
                continue
    except Exception:
        return ""
    return ""


def _browser_family_from_path(path: str) -> str:
    name = Path(str(path or "")).name.lower()
    if "firefox" in name:
        return "firefox"
    if "msedge" in name or "microsoft\\edge" in str(path or "").lower():
        return "edge"
    return "chromium"


def _find_windows_browser_executable(browser: str = "chrome") -> str:
    """Return the installed browser executable on Windows when possible.

    V124 uses Google Chrome by default. Edge support remains only as an explicit
    developer override, but the packaged .env no longer forces Edge.
    """
    browser = str(browser or "chrome").strip().lower()
    candidates = []
    for key in ["DOMESTIC_BROWSER_EXECUTABLE_PATH", "DOMESTIC_CHROME_EXECUTABLE_PATH", "DOMESTIC_EDGE_EXECUTABLE_PATH"]:
        v = os.getenv(key, "").strip().strip('"')
        if v:
            candidates.append(v)
    if os.name == "nt":
        if browser in {"chrome", "google-chrome", "google_chrome"}:
            for env_key in ["ProgramFiles", "ProgramFiles(x86)", "LocalAppData"]:
                base = os.getenv(env_key, "")
                if not base:
                    continue
                candidates.append(str(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"))
            try:
                import winreg  # type: ignore
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe") as key:
                    exe, _ = winreg.QueryValueEx(key, None)
                    if exe:
                        candidates.insert(0, exe)
            except Exception:
                pass
        elif browser in {"msedge", "edge"}:
            for env_key in ["ProgramFiles(x86)", "ProgramFiles", "LocalAppData"]:
                base = os.getenv(env_key, "")
                if not base:
                    continue
                candidates.append(str(Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"))
            try:
                import winreg  # type: ignore
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe") as key:
                    exe, _ = winreg.QueryValueEx(key, None)
                    if exe:
                        candidates.insert(0, exe)
            except Exception:
                pass
    expected = "chrome.exe" if browser in {"chrome", "google-chrome", "google_chrome"} else "msedge.exe"
    for c in candidates:
        try:
            if c and Path(c).exists() and Path(c).name.lower() == expected:
                return c
        except Exception:
            continue
    return ""


def _find_windows_edge_executable() -> str:
    # Backward-compatible helper used in legacy session metadata only.
    return _find_windows_browser_executable("msedge")


def _find_windows_chrome_executable() -> str:
    return _find_windows_browser_executable("chrome")


def _launch_persistent_browser(playwright, launch_kwargs: dict, browser_channel: str, browser_executable: str) -> tuple[object, str]:
    """Launch the requested real browser.

    V124 default is Google Chrome. It does not force Edge. If Chrome is requested
    and cannot be started via Playwright channel, it tries chrome.exe. It does not
    silently launch Edge when the user asked for Chrome.
    """
    channel = str(browser_channel or os.getenv("DOMESTIC_BROWSER_CHANNEL", "msedge") or "chrome").strip().lower()
    exe = str(browser_executable or "").strip().strip('"')
    if channel in {"", "default"}:
        channel = "chrome"

    # Explicit executable path wins.
    if exe:
        kwargs = dict(launch_kwargs)
        kwargs["executable_path"] = exe
        kwargs.pop("channel", None)
        if _browser_family_from_path(exe) == "firefox":
            kwargs.pop("args", None)
            return playwright.firefox.launch_persistent_context(**kwargs), f"firefox_executable:{exe}"
        fam = _browser_family_from_path(exe)
        label = "chrome_executable" if Path(exe).name.lower() == "chrome.exe" else ("microsoft_edge_executable" if fam == "edge" else "chromium_executable")
        return playwright.chromium.launch_persistent_context(**kwargs), f"{label}:{exe}"

    if channel in {"chrome", "google-chrome", "google_chrome"}:
        kwargs = dict(launch_kwargs)
        # Prefer Playwright's real Chrome channel; if it is not installed in a
        # way Playwright can discover, use the Windows chrome.exe path.
        kwargs["channel"] = "chrome"
        try:
            return playwright.chromium.launch_persistent_context(**kwargs), "google_chrome_channel:chrome"
        except Exception as first_exc:
            chrome_exe = _find_windows_chrome_executable()
            if chrome_exe:
                kwargs = dict(launch_kwargs)
                kwargs["executable_path"] = chrome_exe
                kwargs.pop("channel", None)
                return playwright.chromium.launch_persistent_context(**kwargs), f"chrome_executable:{chrome_exe}"
            raise RuntimeError(f"Google Chrome 启动失败：{first_exc}。请安装 Chrome，或在 .env 的 DOMESTIC_CHROME_EXECUTABLE_PATH 填写 chrome.exe 完整路径。")

    if channel in {"msedge", "edge"}:
        kwargs = dict(launch_kwargs)
        kwargs["channel"] = "msedge"
        try:
            return playwright.chromium.launch_persistent_context(**kwargs), "microsoft_edge_channel:msedge"
        except Exception as first_exc:
            edge_exe = _find_windows_edge_executable()
            if edge_exe:
                kwargs = dict(launch_kwargs)
                kwargs["executable_path"] = edge_exe
                kwargs.pop("channel", None)
                return playwright.chromium.launch_persistent_context(**kwargs), f"microsoft_edge_executable:{edge_exe}"
            raise RuntimeError(f"Microsoft Edge 启动失败：{first_exc}")

    # Developer fallback only when a non-Chrome/Edge channel is explicitly set.
    kwargs = dict(launch_kwargs)
    if channel in {"chromium"}:
        return playwright.chromium.launch_persistent_context(**kwargs), "developer_chromium_fallback"
    kwargs["channel"] = channel
    return playwright.chromium.launch_persistent_context(**kwargs), f"chromium_channel:{channel}"



def _launch_detached_chrome_handoff(profile_dir: Path, url: str, download_dir: Path, timeout_seconds: int = 86400) -> dict:
    """Open an unmanaged Chrome window at the current TPDC URL, then let users operate it.

    This is the V136 escape hatch for TPDC's fragile SPA. Once the user reaches
    the search/data-selection context, the controlled Playwright page is closed
    and the same Chrome profile is reopened as a normal browser process. No
    Playwright loop can then re-search, re-click, or route the page back to TPDC
    home when the user scrolls.
    """
    result = {"attempted": True, "opened": False, "pid": None, "url": str(url or ""), "profile_dir": str(profile_dir), "note": ""}
    chrome = os.getenv("DOMESTIC_CHROME_EXECUTABLE_PATH", "").strip().strip('"') or _find_windows_chrome_executable()
    if not chrome:
        result["note"] = "未找到 chrome.exe；无法进入脱管浏览器模式。"
        return result
    final_url = str(url or "").strip() or "https://data.tpdc.ac.cn/"
    # Start passive clipboard watcher before opening the unmanaged browser. It
    # does not touch the browser; it only reacts when the user copies the TPDC FTP
    # account block.
    watcher_result = {"attempted": False}
    if _is_truth("DOMESTIC_FTP_CLIPBOARD_WATCHER", "1"):
        try:
            watcher = Path(__file__).resolve().parent / "ftp_clipboard_watcher.py"
            if watcher.exists():
                proc_w = subprocess.Popen([
                    sys.executable,
                    str(watcher),
                    "--download-dir", str(download_dir),
                    "--timeout", str(timeout_seconds),
                    "--interval", str(os.getenv("DOMESTIC_FTP_CLIPBOARD_INTERVAL_SECONDS", "2")),
                ], cwd=str(Path(__file__).resolve().parents[1]))
                watcher_result = {"attempted": True, "opened": True, "pid": proc_w.pid, "script": str(watcher)}
            else:
                watcher_result = {"attempted": True, "opened": False, "error": f"watcher_not_found:{watcher}"}
        except Exception as exc:
            watcher_result = {"attempted": True, "opened": False, "error": str(exc)}
    try:
        cmd = [
            chrome,
            f"--user-data-dir={str(profile_dir)}",
                final_url,
        ]
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        result.update({"opened": True, "pid": proc.pid, "command": cmd, "clipboard_watcher": watcher_result, "note": "已打开脱管 Chrome。后续不会有 Playwright 控制页面，滚轮和数据选择不应再被拉回首页。"})
    except Exception as exc:
        result.update({"opened": False, "clipboard_watcher": watcher_result, "note": f"脱管 Chrome 启动失败：{exc}"})
    return result




def _pick_free_port() -> int:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _wait_for_cdp_endpoint(port: int, timeout_seconds: int = 20) -> bool:
    import urllib.request
    deadline = time.time() + max(3, timeout_seconds)
    url = f"http://127.0.0.1:{int(port)}/json/version"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


def _start_ftp_clipboard_watcher(
    download_dir: Path,
    timeout_seconds: int = 86400,
    *,
    cdp_endpoint: str = "",
    state_json: Path | None = None,
    requested_year: str = "",
    data_name: str = "",
    live_screenshot: Path | None = None,
    browser_pid: str | int | None = None,
) -> dict:
    """Start the passive FTP watcher.

    V138: the watcher is still passive with respect to TPDC navigation. It may
    read the current Chrome page through CDP and/or the clipboard, but it must
    never click, fill, press keys, reload, close pages, or close Chrome. Once an
    FTP account block is detected, it starts the Python/WinSCP FTP handoff.
    """
    result = {"attempted": False, "opened": False}
    if not _is_truth("DOMESTIC_FTP_CLIPBOARD_WATCHER", "1"):
        result["note"] = "FTP 捕捉器未启用。"
        return result
    try:
        watcher = Path(__file__).resolve().parent / "ftp_clipboard_watcher.py"
        if not watcher.exists():
            return {"attempted": True, "opened": False, "error": f"watcher_not_found:{watcher}"}
        cmd = [
            sys.executable,
            str(watcher),
            "--download-dir", str(download_dir),
            "--timeout", str(timeout_seconds),
            "--interval", str(os.getenv("DOMESTIC_FTP_CLIPBOARD_INTERVAL_SECONDS", "2")),
            "--auto-download", str(os.getenv("DOMESTIC_FTP_AUTO_DOWNLOAD", "0")),
            "--requested-year", str(requested_year or ""),
            "--data-name", str(data_name or ""),
        ]
        if cdp_endpoint:
            cmd += ["--cdp-endpoint", str(cdp_endpoint)]
        if live_screenshot:
            cmd += ["--live-screenshot", str(live_screenshot)]
        if browser_pid:
            cmd += ["--browser-pid", str(browser_pid)]
        if state_json:
            cmd += ["--state-json", str(state_json)]
        env = os.environ.copy()
        if cdp_endpoint:
            env.setdefault("DOMESTIC_FTP_ALLOW_CLIPBOARD_WITH_CDP", "0")
        proc_w = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]), env=env)
        return {"attempted": True, "opened": True, "pid": proc_w.pid, "script": str(watcher), "mode": "clipboard_plus_readonly_cdp", "cdp_endpoint": cdp_endpoint or "", "live_screenshot": str(live_screenshot or ""), "browser_pid": str(browser_pid or "")}
    except Exception as exc:
        return {"attempted": True, "opened": False, "error": str(exc)}




def _spawn_background_ftp_download_from_accounts(download_dir: Path, accounts: list[dict], *, requested_year: str = "", state_json: Path | None = None) -> dict:
    """Start one robust FTP downloader for one TPDC credential ticket.

    In normal operation the passive ftp_clipboard_watcher is already running and
    is the single owner of FTP download startup.  This fallback is disabled by
    default to avoid two processes downloading the same ticket.
    """
    if not _is_truth(os.getenv("DOMESTIC_FTP_ENABLE_PLAYWRIGHT_DIRECT_DOWNLOAD", "0"), "1"):
        return {"ok": True, "status": "playwright_direct_download_disabled", "message": "FTP下载由被动捕捉器统一启动；playwright不再重复启动下载器。"}
    try:
        download_dir.mkdir(parents=True, exist_ok=True)
        accounts = _normalize_tpdc_fixed_host_accounts([dict(a) for a in (accounts or []) if a.get("username") and a.get("password")])
        if not accounts:
            return {"ok": False, "status": "no_valid_tpdc_credentials", "message": "未捕捉到有效TPDC FTP用户名/密码。"}
        # Parent-side duplicate guard shared with ftp_clipboard_watcher.
        spawn_lock = download_dir / ".ftp_downloader.spawn.lock.json"
        status_path = download_dir / "ftp_download_status.json"
        try:
            status_data = json.loads(status_path.read_text(encoding="utf-8") or "{}") if status_path.exists() else {}
        except Exception:
            status_data = {}
        if spawn_lock.exists() and str(status_data.get("status") or "").lower() not in {"ftp_download_completed", "ftp_download_failed", "ftp_download_partial"}:
            return {"ok": True, "status": "ftp_download_process_already_running", "lock": str(spawn_lock)}
        try:
            spawn_lock.write_text(json.dumps({"pid": os.getpid(), "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "playwright_direct_fallback"}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        for i, a in enumerate(accounts):
            a.setdefault("priority", i + 1)
            a["primary"] = (i == 0)
        accounts_path = download_dir / "ftp_accounts_detected.json"
        accounts_path.write_text(json.dumps({
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "cdp_external_passive_loop",
            "accounts": accounts,
            "primary": accounts[0],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        watcher = Path(__file__).resolve().parent / "ftp_clipboard_watcher.py"
        cmd = [
            sys.executable,
            str(watcher),
            "--run-download-from-json", str(accounts_path),
            "--download-dir", str(download_dir),
            "--requested-year", str(requested_year or ""),
        ]
        if state_json:
            cmd += ["--state-json", str(state_json)]
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        # Public, redacted status for UI / logs.
        redacted = []
        for a in accounts:
            b = dict(a)
            if b.get("password"):
                pw = str(b.get("password"))
                b["password_masked"] = (pw[:1] + "***" + pw[-1:]) if len(pw) > 2 else "***"
                b.pop("password", None)
            redacted.append(b)
        return {
            "ok": True,
            "status": "ftp_download_process_started",
            "pid": proc.pid,
            "accounts_path": str(accounts_path),
            "accounts_count": len(accounts),
            "primary": redacted[0] if redacted else {},
            "command": cmd,
        }
    except Exception as exc:
        return {"ok": False, "status": "ftp_download_process_start_failed", "error": str(exc)}


def _scan_browser_for_tpdc_ftp_accounts(browser) -> tuple[list[dict], dict]:
    """Read all TPDC tabs in the connected Edge/Chrome browser and extract FTP accounts.

    This is strictly read-only: no clicks, no fills, no navigation and no tab
    switching. It fixes the observed case where the detached clipboard watcher
    is running but the user gets no visible reaction after the FTP modal appears.
    """
    accounts: list[dict] = []
    pages_seen = []
    errors = []
    try:
        contexts = list(getattr(browser, "contexts", []) or [])
        for ctx in contexts:
            for pg in list(getattr(ctx, "pages", []) or []):
                try:
                    url = str(getattr(pg, "url", "") or "")
                    title = ""
                    try:
                        title = str(pg.title(timeout=500) or "")
                    except Exception:
                        pass
                    if "tpdc.ac.cn" not in url and "青藏" not in title and "TPDC" not in title.upper():
                        continue
                    pages_seen.append({"url": url, "title": title})
                    found = _extract_ftp_accounts(pg)
                    if found:
                        accounts.extend(found)
                except Exception as exc:
                    errors.append(str(exc)[:300])
    except Exception as exc:
        errors.append(str(exc)[:300])
    dedup = []
    seen = set()
    for a in accounts:
        key = (str(a.get("host") or "").lower(), int(a.get("port") or 21), str(a.get("username") or ""), str(a.get("password") or ""))
        if not key[0] or not key[2] or not key[3] or key in seen:
            continue
        seen.add(key)
        b = dict(a)
        b["priority"] = len(dedup) + 1
        b["primary"] = len(dedup) == 0
        dedup.append(b)
    return dedup, {"pages_seen": pages_seen, "errors": errors, "pages_count": len(pages_seen)}



def _collect_tpdc_ftp_accounts_from_cdp_endpoint(endpoint: str, max_pages: int = 80) -> tuple[list[dict], dict]:
    """Freshly attach to the Edge/Chrome CDP endpoint and scan every TPDC target.

    V173: the persistent Playwright `browser` object can miss tabs that TPDC opens
    after handoff on some Edge setups, especially when DevTools is open. This
    function reconnects to the same CDP endpoint on every probe, enumerates all
    browser targets, and reads visible text/HTML/attributes from every TPDC tab
    without clicking or navigating.
    """
    endpoint = str(endpoint or "").strip().rstrip("/")
    if not endpoint:
        return [], {"source": "fresh_cdp_endpoint", "error": "empty_endpoint", "pages_count": 0, "pages_seen": []}
    accounts: list[dict] = []
    pages_seen: list[dict] = []
    errors: list[str] = []
    target_count = 0
    try:
        import urllib.request
        with urllib.request.urlopen(f"{endpoint}/json/list", timeout=2.5) as resp:
            targets = json.loads(resp.read().decode("utf-8", errors="ignore") or "[]")
            target_count = len(targets) if isinstance(targets, list) else 0
            for t in (targets if isinstance(targets, list) else []):
                url = str(t.get("url") or "")
                title = str(t.get("title") or "")
                typ = str(t.get("type") or "")
                if typ != "page":
                    continue
                if "tpdc.ac.cn" in url or "青藏" in title or "TPDC" in title.upper():
                    pages_seen.append({"url": url, "title": title, "type": typ, "raw_target": True})
    except Exception as exc:
        errors.append(f"json_list_failed:{str(exc)[:240]}")

    js = r"""
    () => {
      const out = [];
      const push = (s) => {
        if (s === undefined || s === null) return;
        s = String(s).trim();
        if (s) out.push(s);
      };
      const isVisible = (el) => {
        try {
          const st = window.getComputedStyle(el);
          const r = el.getBoundingClientRect();
          return st && st.display !== 'none' && st.visibility !== 'hidden' &&
                 Number(st.opacity || 1) !== 0 && r.width > 0 && r.height > 0;
        } catch(e) { return true; }
      };
      push(location.href);
      push(document.title || '');
      push(document.body ? document.body.innerText : '');
      push(document.body ? document.body.textContent : '');
      try { push((document.documentElement ? document.documentElement.outerHTML : '').slice(0, 350000)); } catch(e) {}
      Array.from(document.querySelectorAll('input, textarea')).forEach((el) => {
        push(el.value);
        push(el.getAttribute('placeholder'));
        push(el.getAttribute('title'));
        push(el.getAttribute('aria-label'));
      });
      Array.from(document.querySelectorAll('[title], [aria-label], [data-clipboard-text], [data-copy], [data-value], [value], [data-original-title], [data-content], a[href]')).forEach((el) => {
        ['title','aria-label','data-clipboard-text','data-copy','data-value','value','data-original-title','data-content','href'].forEach(attr => push(el.getAttribute(attr)));
      });
      Array.from(document.querySelectorAll('[role="dialog"], .modal, .el-dialog, .ivu-modal, .ant-modal, .layui-layer, button, a, span, div, p, td, th, li, label'))
        .filter(isVisible)
        .forEach((el) => push(el.innerText || el.textContent));
      return Array.from(new Set(out)).join('\n');
    }
    """
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser2 = p.chromium.connect_over_cdp(endpoint)
            try:
                pages = []
                for ctx in list(getattr(browser2, "contexts", []) or []):
                    pages.extend(list(getattr(ctx, "pages", []) or []))
                if len(pages) > max_pages:
                    pages = pages[-max_pages:]
                for idx, pg in enumerate(pages):
                    try:
                        url = str(getattr(pg, "url", "") or "")
                        title = ""
                        try:
                            title = str(pg.title(timeout=500) or "")
                        except Exception:
                            pass
                        if "tpdc.ac.cn" not in url and "青藏" not in title and "TPDC" not in title.upper():
                            continue
                        page_text = ""
                        try:
                            page_text = pg.evaluate(js) or ""
                        except Exception as exc:
                            errors.append(f"page_eval_failed:{str(exc)[:200]}")
                        texts = [url, title, page_text]
                        try:
                            for fr in list(getattr(pg, "frames", []) or []):
                                try:
                                    texts.append(fr.evaluate(js) or "")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        combined = "\n".join(texts)
                        pages_seen.append({"url": url, "title": title, "idx": idx, "chars": len(combined), "ftp_signal": bool(re.search(r"用\s*户\s*名|密\s*码", combined))})
                        accounts.extend(_extract_ftp_accounts_from_text(combined))
                    except Exception as exc:
                        errors.append(f"page_scan_failed:{str(exc)[:240]}")
            finally:
                # Disconnect only this CDP client; never close the user's browser.
                try:
                    browser2.close()
                except Exception:
                    pass
    except Exception as exc:
        errors.append(f"fresh_cdp_failed:{str(exc)[:300]}")

    dedup: list[dict] = []
    seen: set[tuple] = set()
    for a in accounts:
        key = (str(a.get("host") or "").lower(), int(a.get("port") or 21), str(a.get("username") or ""), str(a.get("password") or ""))
        if not key[0] or not key[2] or not key[3] or key in seen:
            continue
        seen.add(key)
        b = dict(a)
        b.setdefault("source", "fresh_cdp_endpoint")
        b["priority"] = len(dedup) + 1
        b["primary"] = len(dedup) == 0
        dedup.append(b)
    return dedup, {"source": "fresh_cdp_endpoint", "endpoint": endpoint, "target_count": target_count, "pages_count": len(pages_seen), "pages_seen": pages_seen[-12:], "errors": errors[-12:]}


def _run_cdp_external_passive_ftp_loop(browser, download_dir: Path, *, state_json: Path | None, live_screenshot: Path | None, requested_year: str = "", timeout_seconds: int = 86400, initial_state: dict | None = None, cdp_endpoint: str = "") -> int:
    """Keep the CDP connection alive and passively capture TPDC FTP credentials.

    The earlier implementation returned immediately after search handoff and
    relied only on a detached watcher. On some Windows/Edge setups the detached
    watcher did not visibly react. This loop is deliberately simple and runs in
    the same process that already printed CDP logs, so PyCharm and the UI state
    both receive clear feedback when the FTP modal appears.
    """
    deadline = time.time() + max(30, int(timeout_seconds or 86400))
    seen_sig = set()
    probe_count = 0
    last_print = 0.0
    last_state = 0.0
    print("[CDP][FTP-WATCH] 被动FTP捕捉循环已启动；请在TPDC详情页点击下载，弹出FTP账号后系统将自动捕捉。", flush=True)
    while time.time() < deadline:
        probe_count += 1
        # First scan the existing Playwright browser object, then do a fresh
        # CDP endpoint attach. The fresh attach is the important V173 fix: TPDC
        # often opens the selected dataset in a new tab after the original CDP
        # connection was created, and some Edge setups do not surface that tab
        # through the old object. Both scans are read-only.
        try:
            accounts_a, meta_a = _scan_browser_for_tpdc_ftp_accounts(browser)
        except Exception as exc:
            accounts_a, meta_a = [], {"errors": [str(exc)[:500]], "pages_seen": [], "pages_count": 0, "source": "attached_browser_object"}
        try:
            accounts_b, meta_b = _collect_tpdc_ftp_accounts_from_cdp_endpoint(cdp_endpoint) if cdp_endpoint else ([], {"source": "fresh_cdp_endpoint", "error": "no_endpoint", "pages_seen": [], "pages_count": 0})
        except Exception as exc:
            accounts_b, meta_b = [], {"errors": [str(exc)[:500]], "pages_seen": [], "pages_count": 0, "source": "fresh_cdp_endpoint"}
        # Collapse all captures to one TPDC ticket.  The UI exposes Chinese
        # “用户名/密码”; this version uses one fixed FTP host only:
        # ftp2.tpdc.ac.cn:6201.
        raw_accounts = []
        _seen_ticket = set()
        for _acc in list(accounts_a or []) + list(accounts_b or []):
            _key = (str(_acc.get("username") or ""), str(_acc.get("password") or ""))
            if not _key[0] or not _key[1] or _key in _seen_ticket:
                continue
            _seen_ticket.add(_key)
            raw_accounts.append(dict(_acc))
        accounts = _normalize_tpdc_fixed_host_accounts(raw_accounts[:1])
        meta = {"attached_browser_object": meta_a, "fresh_cdp_endpoint": meta_b, "pages_count": int(meta_a.get("pages_count") or 0) + int(meta_b.get("pages_count") or 0)}
        now = time.time()
        if now - last_state >= 5:
            last_state = now
            try:
                base = dict(initial_state or {})
                base.update({
                    "status": "ftp_account_waiting",
                    "ftp_watcher_status": "cdp_passive_waiting",
                    "ftp_probe_count": probe_count,
                    "message": "FTP捕捉器运行中：等待TPDC弹出中文用户名和密码字段。",
                    "ftp_probe_meta": meta,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "live_screenshot": str(live_screenshot or ""),
                    "download_dir": str(download_dir),
                })
                if state_json:
                    _write_json(state_json, base)
            except Exception:
                pass
        if now - last_print >= 10:
            last_print = now
            print(f"[CDP][FTP-WATCH] probe={probe_count} tpdc_pages={meta.get('pages_count')} accounts={len(accounts)}", flush=True)
        if accounts:
            sig = tuple((a.get("username"), a.get("password")) for a in accounts[:1])
            if sig in seen_sig:
                time.sleep(1.0)
                continue
            seen_sig.add(sig)
            files = _write_ftp_handoff_files(download_dir, accounts)
            dl = _spawn_background_ftp_download_from_accounts(download_dir, accounts, requested_year=requested_year, state_json=state_json)
            safe_primary = {k: v for k, v in accounts[0].items() if k != "password"} if accounts else {}
            print(f"[CDP][FTP-WATCH] 已捕捉FTP用户名/密码 1 组，主主机={safe_primary.get('host')}，主备主机数={len(accounts)}，下载进程={dl.get('pid')}", flush=True)
            try:
                state = dict(initial_state or {})
                state.update({
                    "status": "ftp_credentials_detected",
                    "ftp_watcher_status": "captured",
                    "ftp_accounts_count": 1,
                    "ftp_ticket_count": 1,
                    "ftp_host_count": len(accounts),
                    "ftp_primary_account": safe_primary,
                    "ftp_handoff_files": files,
                    "ftp_download_process": dl,
                    "message": "已捕捉到 1 组 TPDC FTP 用户名和密码，正在按固定主备主机策略启动 Python FTP 下载。",
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "download_dir": str(download_dir),
                })
                if state_json:
                    _write_json(state_json, state)
            except Exception:
                pass
            return 0
        time.sleep(max(0.8, float(os.getenv("DOMESTIC_FTP_CDP_PASSIVE_INTERVAL_SECONDS", "2"))))
    print("[CDP][FTP-WATCH] 超时：未捕捉到FTP账号。", flush=True)
    return 0

def _launch_external_browser_with_cdp(profile_dir: Path, entry_url: str) -> dict:
    """Launch or reuse a real browser process with a CDP port.

    V160+ default: prefer Microsoft Edge, because the user normally operates the
    Dash app in Edge and does not want a separate Google Chrome window.
    Security note: an ordinary already-open Edge tab cannot be scripted by a
    local Dash app unless that browser was launched with a remote-debugging port.
    Therefore this helper uses the same browser family (Edge by default) with a
    stable automation profile and CDP, instead of opening an in-app handoff modal.
    """
    profile_dir.mkdir(parents=True, exist_ok=True)
    preferred = str(os.getenv("DOMESTIC_BROWSER_CHANNEL", "msedge") or "msedge").strip().lower()
    explicit = os.getenv("DOMESTIC_BROWSER_EXECUTABLE_PATH", "").strip().strip('"')
    if not explicit:
        if preferred in {"msedge", "edge"}:
            explicit = os.getenv("DOMESTIC_EDGE_EXECUTABLE_PATH", "").strip().strip('"') or _find_windows_edge_executable()
        else:
            explicit = os.getenv("DOMESTIC_CHROME_EXECUTABLE_PATH", "").strip().strip('"') or _find_windows_chrome_executable()
    if not explicit:
        return {"ok": False, "error": "未找到可用浏览器。请在 .env 中填写 DOMESTIC_BROWSER_EXECUTABLE_PATH 或 DOMESTIC_EDGE_EXECUTABLE_PATH。", "preferred_browser": preferred}
    browser_path = explicit
    browser_name = "Microsoft Edge" if "edge" in Path(browser_path).name.lower() or "msedge" in Path(browser_path).name.lower() else "Google Chrome"
    port_file = profile_dir / "cdp_port.txt"
    existing_port = None
    try:
        if port_file.exists():
            existing_port = int((port_file.read_text(encoding="utf-8") or "").strip())
            if _wait_for_cdp_endpoint(existing_port, timeout_seconds=2):
                return {"ok": True, "reused": True, "pid": None, "port": existing_port, "endpoint": f"http://127.0.0.1:{existing_port}", "profile_dir": str(profile_dir), "browser": browser_path, "browser_name": browser_name}
    except Exception:
        existing_port = None
    port = _pick_free_port()
    cmd = [
        browser_path,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={str(profile_dir)}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-first-run-ui",
        "--disable-features=Translate,MediaRouter,OptimizationHints",
        entry_url or "https://data.tpdc.ac.cn/",
    ]
    try:
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        port_file.write_text(str(port), encoding="utf-8")
        if not _wait_for_cdp_endpoint(port, timeout_seconds=int(os.getenv("DOMESTIC_CDP_CONNECT_TIMEOUT_SECONDS", "25"))):
            return {"ok": False, "error": "浏览器已启动但 CDP 端口未就绪。", "pid": proc.pid, "port": port, "command": cmd}
        return {"ok": True, "reused": False, "pid": proc.pid, "port": port, "endpoint": f"http://127.0.0.1:{port}", "profile_dir": str(profile_dir), "browser": browser_path, "browser_name": browser_name, "command": cmd}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": cmd}


def _append_action_log(log_path: Path, action: str, **kwargs) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action}
        rec.update(kwargs)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _main_cdp_external(args, project_root: Path) -> int:
    """V137 simplified TPDC flow: external Chrome + temporary CDP connection.

    Workflow:
    1) Start real Chrome with remote debugging and persistent TPDC profile.
    2) Connect through CDP only long enough to fill credentials and search once.
    3) Start passive clipboard FTP watcher.
    4) Return without closing Chrome, contexts or pages.
    """
    platform = "tpdc"
    entry_url = "https://data.tpdc.ac.cn/"
    started_at = time.strftime("%Y%m%d_%H%M%S")
    browser_profile_suffix = "tpdc_edge_cdp" if str(os.getenv("DOMESTIC_BROWSER_CHANNEL", "msedge")).strip().lower() in {"msedge", "edge"} else "tpdc_chrome_cdp"
    if args.profile_dir:
        profile_dir = Path(args.profile_dir)
    elif args.session_dir:
        profile_dir = Path(args.session_dir) / browser_profile_suffix
    else:
        profile_dir = Path(os.getenv("DOMESTIC_PLATFORM_SESSION_DIR", str(project_root / "data" / "playwright_profiles"))) / browser_profile_suffix
    # V137: do not create a fresh profile per task; that causes Chrome first-run
    # screens and loses TPDC login persistence. Keep one stable TPDC CDP profile.
    download_dir = Path(args.download_root) if args.download_root else project_root / "data" / "manual_domestic_downloads" / platform / started_at
    download_dir.mkdir(parents=True, exist_ok=True)
    live_screenshot = Path(args.live_screenshot) if args.live_screenshot else download_dir / "_browser_live" / "live.png"
    state_json = Path(args.state_json) if args.state_json else download_dir / "_browser_live" / "state.json"
    live_screenshot.parent.mkdir(parents=True, exist_ok=True)
    state_json.parent.mkdir(parents=True, exist_ok=True)
    action_log = download_dir / "tpdc_cdp_action_log.jsonl"

    def update_state(**kwargs):
        base = {
            "platform": platform,
            "platform_name": PLATFORM_NAMES.get(platform, platform),
            "entry_url": entry_url,
            "download_dir": str(download_dir),
            "live_screenshot": str(live_screenshot),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "browser_control_mode": "cdp_external",
        }
        base.update(kwargs)
        _write_json(state_json, base)

    user_env, pass_env, username, password = _get_platform_credentials(platform)
    disable_auto_login = (os.getenv("DOMESTIC_PLATFORM_AUTO_LOGIN", "1").strip() == "0") or (os.getenv("DOMESTIC_CAPTURE_AUTO_LOGIN", "1").strip() == "0")
    use_env_login = bool((args.auto_login or (username and password)) and not disable_auto_login)
    keyword = str(args.keyword or args.data_name or "").strip() or "DEM"

    launch_result = _launch_external_browser_with_cdp(profile_dir, entry_url)
    watcher_result = {"attempted": False, "opened": False, "note": "浏览器未启动前不启动 FTP 捕捉器。"}
    if launch_result.get("ok"):
        watcher_result = _start_ftp_clipboard_watcher(
            download_dir,
            timeout_seconds=args.timeout,
            cdp_endpoint=str(launch_result.get("endpoint") or ""),
            state_json=state_json,
            requested_year=str(args.requested_year or ""),
            data_name=str(args.data_name or keyword or ""),
            live_screenshot=live_screenshot,
            browser_pid=str(launch_result.get("pid") or ""),
        )
    update_state(
        status="external_browser_started" if launch_result.get("ok") else "external_browser_failed",
        message="使用受控 Edge/浏览器 + CDP 只读捕捉 FTP；用户选择数据集和点击下载后，后台自动读取 FTP 弹窗并启动下载。",
        launch_result=launch_result,
        clipboard_watcher=watcher_result,
        keyword=keyword,
        credential_username_env=user_env,
        credential_username_masked=_mask_username(username),
    )
    _append_action_log(action_log, "launch_external_browser", result=launch_result, watcher=watcher_result)
    if not launch_result.get("ok"):
        print(f"[CDP][ERROR] {launch_result}", file=sys.stderr)
        return 3

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("[ERROR] 缺少 playwright。请先执行 pip install playwright", file=sys.stderr)
        return 2

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(str(launch_result["endpoint"]))
        try:
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.pages[0] if context.pages else context.new_page()
            # Ensure TPDC tab is foregrounded and at the expected entry page.
            before_url = getattr(page, "url", "")
            if "data.tpdc.ac.cn" not in str(before_url):
                _append_action_log(action_log, "goto_entry", before_url=before_url, reason="active_page_not_tpdc")
                page.goto(entry_url, wait_until="domcontentloaded", timeout=60000)
            try:
                _safe_live_screenshot(page, live_screenshot)
            except Exception:
                pass

            login_result = {"attempted": False, "note": "未配置或未启用 TPDC 自动登录。"}
            login_open_result = {"attempted": False, "opened": False, "note": "未请求预登录。"}
            login_wait_result = {"waited": False, "completed": False, "note": "未等待验证码。"}
            if _is_truth("DOMESTIC_TPDC_PRELOGIN", "1") and use_env_login:
                update_state(status="auto_login_opening", message="正在打开 TPDC 登录界面并填充账号密码；验证码由用户输入。", launch_result=launch_result, clipboard_watcher=watcher_result)
                _append_action_log(action_log, "open_login_panel", before_url=getattr(page, "url", ""))
                login_open_result = _open_tpdc_login_panel(page, entry_url=entry_url)
                _append_action_log(action_log, "open_login_panel_result", result=login_open_result, after_url=getattr(page, "url", ""))
                if login_open_result.get("opened"):
                    login_result = _try_fill_login_form(page, username, password)
                    _append_action_log(action_log, "fill_login_result", result=login_result, after_url=getattr(page, "url", ""))
                    if login_result.get("username_filled") or login_result.get("password_filled"):
                        update_state(status="waiting_login_verification", message="TPDC 账号密码已填好。请手动输入验证码并点击登录；登录成功后会自动搜索一次数据名称。", login_open_result=login_open_result, login_result=login_result, launch_result=launch_result, clipboard_watcher=watcher_result)
                        login_wait_result = _wait_for_tpdc_manual_captcha_and_login(page, timeout_seconds=int(os.getenv("DOMESTIC_LOGIN_USER_VERIFY_WAIT_SECONDS", "600")))
                        _append_action_log(action_log, "login_wait_result", result=login_wait_result, after_url=getattr(page, "url", ""))
                else:
                    login_result = {"attempted": True, "username_filled": False, "password_filled": False, "note": login_open_result.get("note") or "未能打开 TPDC 登录表单。"}

            # Search exactly once, using data name only. If a login panel is still
            # visible, stop automation and let the user continue manually.
            search_result = {"attempted": False, "filled": False, "submitted": False, "note": "未执行检索。"}
            if not _page_has_visible_password_or_verification(page):
                update_state(status="searching_once", message=f"正在 TPDC 搜索框检索数据名称：{keyword}。之后立即断开控制，不再重搜、不再点击、不再改 URL。", login_open_result=login_open_result, login_result=login_result, login_wait_result=login_wait_result, launch_result=launch_result, clipboard_watcher=watcher_result)
                _append_action_log(action_log, "search_once_start", before_url=getattr(page, "url", ""), keyword=keyword)
                search_result = _try_search_keyword(page, keyword)
                _append_action_log(action_log, "search_once_result", result=search_result, after_url=getattr(page, "url", ""))
                try:
                    _safe_live_screenshot(page, live_screenshot)
                except Exception:
                    pass
            else:
                search_result = {"attempted": False, "submitted": False, "note": "仍在可见登录/验证码界面；本轮不搜索，避免误填。"}
                _append_action_log(action_log, "search_skipped_login_visible", url=getattr(page, "url", ""))

            # Final handoff. Do not close browser/context/page. Exiting this process
            # only drops the CDP connection; the external Chrome process remains.
            final_url = getattr(page, "url", "")
            update_state(
                status="manual_selection_mode",
                message="已断开主动页面控制。请在当前浏览器窗口中滚动、选择数据集、点击下载/FTP。系统后台会只读捕捉当前页面和剪贴板中的 FTP 账号块，捕捉成功后自动生成脚本并尝试 Python FTP 下载。",
                login_open_result=login_open_result,
                login_result=login_result,
                login_wait_result=login_wait_result,
                search_result=search_result,
                final_url=final_url,
                action_log=str(action_log),
                launch_result=launch_result,
                clipboard_watcher=watcher_result,
                instruction="看到 FTP账号/主机/端口/用户名/密码 后，系统会自动只读捕捉；若未捕捉，请点击页面复制按钮或框选文字 Ctrl+C。多个主机按主备策略处理：先用第一个主主机，失败后才切换第二个备用主机。",
            )
            _append_action_log(action_log, "handoff_manual_selection", final_url=final_url)
            print("[CDP] 已进入手动选择模式；不会关闭或重启浏览器。", flush=True)
            if _is_truth("DOMESTIC_CDP_KEEP_PASSIVE_FTP_LOOP", "1"):
                return _run_cdp_external_passive_ftp_loop(
                    browser,
                    download_dir,
                    state_json=state_json,
                    live_screenshot=live_screenshot,
                    requested_year=str(args.requested_year or ""),
                    timeout_seconds=int(args.timeout or 86400),
                    initial_state={
                        "platform": platform,
                        "platform_name": PLATFORM_NAMES.get(platform, platform),
                        "entry_url": entry_url,
                        "browser_control_mode": "cdp_external_passive_ftp_loop",
                        "launch_result": launch_result,
                        "clipboard_watcher": watcher_result,
                        "final_url": final_url,
                        "action_log": str(action_log),
                    },
                    cdp_endpoint=str(launch_result.get("endpoint") or ""),
                )
            return 0
        finally:
            # Intentionally do nothing. Do NOT call browser.close/context.close/page.close.
            pass

def main() -> int:
    project_root = _default_root()
    _load_project_env(project_root)
    parser = argparse.ArgumentParser(description="Playwright 国内平台真实下载捕获工具")
    parser.add_argument("--platform", choices=list(PLATFORM_URLS) + ["all"], default="tpdc")
    parser.add_argument("--platform-sequence", default="", help="逗号分隔的平台尝试顺序；为空则使用 --platform")
    parser.add_argument("--per-platform-timeout", type=int, default=int(os.getenv("DOMESTIC_PER_PLATFORM_TIMEOUT", "900")), help="多平台模式下每个平台最多保持时间，默认15分钟")
    parser.add_argument("--url", default="", help="覆盖默认入口URL")
    parser.add_argument("--timeout", type=int, default=1800, help="浏览器保持打开秒数，默认30分钟")
    parser.add_argument("--headed", action="store_true", default=True, help="显示浏览器窗口，默认显示")
    parser.add_argument("--profile-dir", default="", help="浏览器登录态目录；默认 data/playwright_profiles/<platform>")
    parser.add_argument("--download-root", default="", help="下载保存根目录；默认 data/manual_domestic_downloads")
    parser.add_argument("--keyword", default="MODIS NDVI DEM 土地利用", help="只作为提示，不强制自动搜索")
    parser.add_argument("--auto-login", action="store_true", help="使用 .env 中配置的平台账号尝试自动填充登录表单")
    parser.add_argument("--session-dir", default="", help="浏览器登录态根目录；默认读取 DOMESTIC_PLATFORM_SESSION_DIR 或 data/playwright_profiles")
    parser.add_argument("--browser-channel", default=os.getenv("DOMESTIC_BROWSER_CHANNEL", "msedge"), help="浏览器通道；默认 msedge，优先使用 Microsoft Edge")
    parser.add_argument("--browser-executable", default=os.getenv("DOMESTIC_BROWSER_EXECUTABLE_PATH", "") or os.getenv("DOMESTIC_CHROME_EXECUTABLE_PATH", "") or os.getenv("DOMESTIC_EDGE_EXECUTABLE_PATH", ""), help="浏览器可执行文件路径，可选；Chrome 模式下可填写 chrome.exe 完整路径")
    parser.add_argument("--edge-executable", default="", help="兼容旧参数；请改用 --browser-executable")
    parser.add_argument("--data-name", default="", help="用户真实请求的数据类型，如 DEM/NDVI/LULC")
    parser.add_argument("--requested-resolution-meters", default="", help="用户请求分辨率，单位米；实际数据必须等于或更精细")
    parser.add_argument("--requested-year", default="", help="用户请求年份；多年数据集只要覆盖该年份即可，后续只保留目标年份")
    parser.add_argument("--live-screenshot", default="", help="实时投射截图输出路径，可选")
    parser.add_argument("--state-json", default="", help="实时状态 JSON 输出路径，可选")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("[ERROR] 缺少 playwright。请先执行：", file=sys.stderr)
        print("  pip install playwright", file=sys.stderr)
        print("  python -m playwright install chromium", file=sys.stderr)
        return 2

    try:
        requested_resolution_meters = float(args.requested_resolution_meters) if str(args.requested_resolution_meters or "").strip() else None
    except Exception:
        requested_resolution_meters = None
    try:
        requested_year = int(args.requested_year) if str(args.requested_year or "").strip() else None
    except Exception:
        requested_year = None
    if args.edge_executable and not args.browser_executable:
        args.browser_executable = args.edge_executable
    # V160: force TPDC only, and prefer Microsoft Edge by default.
    platforms = ["tpdc"]
    if not str(args.browser_channel or "").strip():
        args.browser_channel = os.getenv("DOMESTIC_BROWSER_CHANNEL", "msedge")
    os.environ.setdefault("DOMESTIC_BROWSER_CHANNEL", "msedge")
    os.environ.setdefault("DOMESTIC_EDGE_ONLY", "1")
    os.environ.setdefault("DOMESTIC_BROWSER_STRICT_EDGE", "1")
    os.environ.setdefault("DOMESTIC_CHROME_ONLY", "0")
    args.platform = "tpdc"
    args.url = "https://data.tpdc.ac.cn/"

    # Default path: external browser + temporary CDP control. This bypasses
    # the old persistent Playwright lifecycle that could close/reopen the browser.
    if str(os.getenv("DOMESTIC_BROWSER_CONTROL_MODE", "cdp_external")).strip().lower() in {"cdp", "cdp_external", "external_chrome"}:
        return _main_cdp_external(args, project_root)

    started_at = time.strftime("%Y%m%d_%H%M%S")

    for platform in platforms:
        entry_url = args.url if args.url and len(platforms) == 1 else PLATFORM_URLS[platform]
        if args.profile_dir:
            profile_dir = Path(args.profile_dir)
        elif args.session_dir:
            profile_dir = Path(args.session_dir) / platform
        else:
            profile_dir = Path(os.getenv("DOMESTIC_PLATFORM_SESSION_DIR", str(project_root / "data" / "playwright_profiles"))) / platform
        # V124: stale persistent profiles were reopening TPDC login pages and
        # preserving odd zoom/layout settings. Use a fresh automation profile by
        # default while still keeping everything under the configured session dir.
        if _is_truth("DOMESTIC_RESET_BROWSER_PROFILE_ON_START", "1"):
            profile_dir = profile_dir / ("fresh_" + started_at)
        download_dir = Path(args.download_root) if args.download_root else project_root / "data" / "manual_domestic_downloads" / platform / started_at
        download_dir.mkdir(parents=True, exist_ok=True)
        profile_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = download_dir / "playwright_capture_manifest.jsonl"
        live_screenshot = Path(args.live_screenshot) if args.live_screenshot else download_dir / "_browser_live" / "live.png"
        state_json = Path(args.state_json) if args.state_json else download_dir / "_browser_live" / "state.json"
        live_screenshot.parent.mkdir(parents=True, exist_ok=True)
        state_json.parent.mkdir(parents=True, exist_ok=True)

        def update_state(**kwargs):
            base = {
                "platform": platform,
                "platform_name": PLATFORM_NAMES.get(platform, platform),
                "entry_url": entry_url,
                "download_dir": str(download_dir),
                "live_screenshot": str(live_screenshot),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            base.update(kwargs)
            try:
                _write_json(state_json, base)
            except Exception:
                pass

        print("=" * 80)
        print(f"[PLAYWRIGHT] 平台：{platform}")
        print(f"[PLAYWRIGHT] 入口：{entry_url}")
        print(f"[PLAYWRIGHT] 登录态目录：{profile_dir}")
        print(f"[PLAYWRIGHT] 下载目录：{download_dir}")
        print(f"[PLAYWRIGHT] 浏览器策略：优先 Microsoft Edge / channel={args.browser_channel} / executable={args.browser_executable or os.getenv('DOMESTIC_BROWSER_EXECUTABLE_PATH', '') or os.getenv('DOMESTIC_EDGE_EXECUTABLE_PATH', '') or 'auto'}")
        print(f"[PLAYWRIGHT] 操作提示：先打开 TPDC 登录界面并填好账号密码；用户只输入验证码。登录成功后自动按“数据名称”检索，并且必须继续到具体数据集选择/下载入口；FTP 入口优先。")
        user_env, pass_env, username, password = _get_platform_credentials(platform)
        # V124: in TPDC-only mode, available TPDC_USERNAME/TPDC_PASSWORD means
        # auto-fill must run by default. DOMESTIC_PLATFORM_AUTO_LOGIN=0 or
        # DOMESTIC_CAPTURE_AUTO_LOGIN=0 can explicitly disable it.
        disable_auto_login = (os.getenv("DOMESTIC_PLATFORM_AUTO_LOGIN", "1").strip() == "0") or (os.getenv("DOMESTIC_CAPTURE_AUTO_LOGIN", "1").strip() == "0")
        use_env_login = bool((args.auto_login or (username and password)) and not disable_auto_login)
        print(f"[PLAYWRIGHT] 关键词参考：{args.keyword}")
        print(f"[PLAYWRIGHT] 自动登录：{use_env_login} | 账号变量：{user_env or 'N/A'} | 账号：{_mask_username(username)}")
        print("[PLAYWRIGHT] 捕获到真实浏览器下载后会自动保存。完成后关闭浏览器或等 timeout。")
        print("[PLAYWRIGHT] 若出现手机验证码、人机验证、实名确认或协议确认，请用户在浏览器中手动完成。")

        with sync_playwright() as p:
            launch_kwargs = dict(
                user_data_dir=str(profile_dir),
                headless=False,
                accept_downloads=True,
                # V127: do not set a fixed pixel viewport/window-size/device scale
                # factor/browser zoom. Chrome may start maximized without explicit
                # dimensions so TPDC uses the browser's native responsive layout.
                # no_viewport=True lets Chrome use its native window size.
                no_viewport=True,
                # V127: do not force --start-maximized or any window geometry.
                # The user requested the platform to keep its native/default layout.
                args=["--disable-blink-features=AutomationControlled"],
            )
            try:
                context, launched_browser = _launch_persistent_browser(p, launch_kwargs, args.browser_channel, args.browser_executable)
                print(f"[PLAYWRIGHT] 实际启动：{launched_browser}")
            except Exception as exc:
                update_state(status="browser_launch_failed", message=f"浏览器启动失败：{exc}。请检查 Microsoft Edge 是否安装，或在 .env 的 DOMESTIC_EDGE_EXECUTABLE_PATH 填写 msedge.exe 完整路径。")
                print(f"[PLAYWRIGHT][ERROR] 浏览器启动失败，已停止：{exc}", file=sys.stderr)
                return 3
            page = context.pages[0] if context.pages else context.new_page()
            captured = []

            def save_download(download):
                try:
                    suggested = _safe_name(download.suggested_filename or Path(urlparse(download.url).path).name or "download.bin")
                    dest = download_dir / suggested
                    # 避免重名覆盖。
                    if dest.exists():
                        stem, suffix = dest.stem, dest.suffix
                        dest = download_dir / f"{stem}_{int(time.time())}{suffix}"
                    download.save_as(str(dest))
                    item = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "platform": platform,
                        "url": download.url,
                        "suggested_filename": download.suggested_filename,
                        "path": str(dest),
                        "bytes": dest.stat().st_size if dest.exists() else None,
                    }
                    captured.append(item)
                    with manifest_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    update_state(status="download_captured", message="已捕获真实浏览器下载任务。", captured_count=len(captured), last_download=item)
                    print(f"[PLAYWRIGHT][DOWNLOAD] 已保存：{dest} | {item['bytes']} bytes")
                except Exception as exc:
                    print(f"[PLAYWRIGHT][ERROR] 保存下载失败：{exc}", file=sys.stderr)

            def log_response(resp):
                try:
                    headers = resp.headers or {}
                    cd = headers.get("content-disposition", "")
                    ct = headers.get("content-type", "")
                    url = resp.url
                    if "attachment" in cd.lower() or any(url.lower().split("?")[0].endswith(ext) for ext in [".zip", ".tif", ".tiff", ".hdf", ".h5", ".nc", ".csv", ".xlsx"]):
                        rec = {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "platform": platform,
                            "kind": "response_candidate",
                            "url": url,
                            "status": resp.status,
                            "content_type": ct,
                            "content_disposition": cd,
                        }
                        with (download_dir / "network_response_candidates.jsonl").open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        print(f"[PLAYWRIGHT][RESPONSE] 候选响应：{resp.status} {ct} {url[:140]}")
                except Exception:
                    pass

            page.on("download", save_download)
            page.on("response", log_response)
            multi_platform = False
            update_state(
                status="opening_tpdc",
                message="V135 换用两段式流程：只负责登录后搜索/进入数据选择一次；一旦到达数据上下文立即交给用户，后台只做被动 FTP 捕捉，不再重搜、不再点下载、不再拉回首页。",
                captured_count=0,
            )
            page.goto(entry_url, wait_until="domcontentloaded", timeout=60000)
            # V127: do not send Ctrl+0 or any zoom/layout command. Use TPDC/Chrome native layout.
            # User explicitly wants the browser to open the TPDC login UI,
            # fill TPDC_USERNAME/TPDC_PASSWORD first, and leave only the captcha /
            # manual verification to the user. Search starts after the login form
            # disappears, or after the configured wait expires.
            login_result = {"attempted": False, "note": "未配置或未启用 TPDC 自动登录。"}
            login_open_result = {"attempted": False, "opened": False, "note": "未请求预登录。"}
            login_wait_result = {"waited": False, "completed": False, "note": "未等待验证码。"}
            if _is_truth("DOMESTIC_TPDC_PRELOGIN", "1") and use_env_login:
                update_state(status="auto_login_opening", message="正在打开 TPDC 登录界面，并自动填充账号密码；验证码需要用户手动输入。", captured_count=0)
                login_open_result = _open_tpdc_login_panel(page, entry_url=entry_url)
                print(f"[PLAYWRIGHT][LOGIN_OPEN] {json.dumps(login_open_result, ensure_ascii=False)}")
                if login_open_result.get("opened"):
                    update_state(status="auto_login_filling", message="已打开 TPDC 登录界面，正在填写账号密码；请只手动输入验证码。", login_open_result=login_open_result, captured_count=0)
                    login_result = _try_fill_login_form(page, username, password)
                    print(f"[PLAYWRIGHT][LOGIN] {json.dumps(login_result, ensure_ascii=False)}")
                    if login_result.get("username_filled") or login_result.get("password_filled"):
                        update_state(status="waiting_login_verification", message="TPDC 账号密码已填好。请在浏览器中输入验证码并点击登录；登录成功后系统会继续检索数据。", login_open_result=login_open_result, login_result=login_result, captured_count=0)
                        login_wait_result = _wait_for_tpdc_manual_captcha_and_login(page, timeout_seconds=int(os.getenv("DOMESTIC_LOGIN_USER_VERIFY_WAIT_SECONDS", "600")))
                        print(f"[PLAYWRIGHT][LOGIN_WAIT] {json.dumps(login_wait_result, ensure_ascii=False)}")
                        if not login_wait_result.get("completed"):
                            update_state(status="waiting_user_login", message="仍在 TPDC 登录/验证码界面。请先完成验证码登录，之后可在浏览器中继续；本次自动检索暂不抢占页面。", login_open_result=login_open_result, login_result=login_result, login_wait_result=login_wait_result, captured_count=0)
                    # V126: do not return to the TPDC home page after login.
                    # The next stage will force-navigate directly to the allData search
                    # route using only the parsed data name. Returning home was the main
                    # reason previous builds appeared to stop after successful login.
                else:
                    login_result = {"attempted": True, "username_filled": False, "password_filled": False, "note": login_open_result.get("note") or "未能打开 TPDC 登录表单。"}
            search_result = {"attempted": False, "filled": False, "submitted": False, "note": "尚未执行检索。"}
            open_result = {"attempted": False, "clicked": False, "note": "尚未打开数据集。"}
            click_result = {"attempted": False, "clicked": False, "note": "尚未点击下载。"}
            ftp_result = {"detected": False, "accounts_count": 0, "note": "尚未提取 FTP。"}
            dataset_signal = {"matched": False, "reason": "not_checked_yet"}
            auto_sequence_done = False
            # V135: two-stage handoff. After the first successful data-name search
            # or dataset-context detection, the script must stop controlling TPDC
            # navigation completely. The user's observed failure was caused by the
            # resume loop repeatedly re-searching/re-clicking after the page had
            # already reached the data-selection interface; scrolling then appeared
            # as a flash back to the initial page. From this point onward the tool
            # acts only as a passive FTP-account observer.
            manual_takeover_lock = False
            manual_takeover_reason = ""
            detached_handoff_result = {"attempted": False, "opened": False}
            detached_handoff_done = False

            def run_tpdc_auto_sequence(phase: str) -> bool:
                """Search TPDC by data name and proceed to dataset/download.

                V127 makes this resumable and does not treat mere search submission as completion. If the user finishes captcha/login after
                the first attempt, the waiting loop calls this function again instead
                of leaving the browser on TPDC home forever.
                """
                nonlocal search_result, open_result, click_result, ftp_result, dataset_signal, auto_sequence_done, login_result, manual_takeover_lock, manual_takeover_reason, detached_handoff_done, detached_handoff_result
                if manual_takeover_lock:
                    # Passive mode: do not search, do not click, do not change URL.
                    current_page_ctx = _tpdc_page_context(page, args.keyword)
                    dataset_signal = _page_has_dataset_signal(page, args.keyword, open_result=open_result, click_result=click_result, ftp_accounts=[], data_name=args.data_name, requested_resolution_meters=requested_resolution_meters, requested_year=requested_year, platform=platform)
                    update_state(
                        status="manual_takeover_passive_ftp_watch",
                        message=f"已到达 TPDC 数据选择/数据集上下文，自动导航已冻结：{manual_takeover_reason}。现在不会再重搜、不会再点页面、不会再回首页；请在浏览器中滚动/选择数据，系统只被动捕捉 FTP 账号。",
                        phase=phase,
                        page_context=current_page_ctx,
                        dataset_signal=dataset_signal,
                        search_result=search_result,
                        open_result=open_result,
                        click_result=click_result,
                        ftp_result=ftp_result,
                        captured_count=len(captured),
                        current_url=getattr(page, "url", ""),
                    )
                    auto_sequence_done = True
                    return True

                if _page_has_visible_password_or_verification(page):
                    update_state(
                        status="waiting_login_before_search",
                        message="当前仍可见登录/验证码界面。请完成验证码并点击登录；登录界面消失后系统会自动继续检索数据名称。",
                        phase=phase,
                        login_open_result=login_open_result,
                        login_result=login_result,
                        login_wait_result=login_wait_result,
                        captured_count=len(captured),
                        current_url=getattr(page, "url", ""),
                    )
                    return False

                page_ctx = _tpdc_page_context(page, args.keyword)
                already_on_result_or_dataset = bool(
                    search_result.get("submitted") and
                    page_ctx.get("has_keyword") and
                    page_ctx.get("on_search_or_dataset") and
                    not page_ctx.get("on_home")
                )
                update_state(
                    status="selecting_dataset" if already_on_result_or_dataset else "searching",
                    message=(
                        f"TPDC 已在检索/数据集上下文中，继续选择具体数据集：{args.keyword}；不再重复提交搜索，避免页面闪回首页。"
                        if already_on_result_or_dataset else
                        f"正在 TPDC 原生搜索框检索数据名称：{args.keyword}。搜索框只使用数据名称，不追加年份、区域、分辨率。"
                    ),
                    phase=phase,
                    login_open_result=login_open_result,
                    login_result=login_result,
                    login_wait_result=login_wait_result,
                    page_context=page_ctx,
                    captured_count=len(captured),
                    current_url=getattr(page, "url", ""),
                )
                if already_on_result_or_dataset:
                    search_result = {**search_result, "attempted": True, "submitted": True, "note": "当前页面已是 TPDC 检索/数据集上下文，本轮不重新提交搜索。", "method": search_result.get("method") or "reuse_current_result_page"}
                else:
                    search_result = _try_search_keyword(page, args.keyword)
                print(f"[PLAYWRIGHT][SEARCH][{phase}] {json.dumps(search_result, ensure_ascii=False)}")

                # V136 hard handoff: TPDC is fragile under continued browser
                # automation. The system now performs login + exactly one data-name
                # search, then stops all Playwright navigation/click control. The
                # user selects data manually in a detached Chrome window. This is
                # intentionally less ambitious but prevents the scroll-triggered
                # flash-back to the initial page.
                if _is_truth("DOMESTIC_TPDC_DETACH_AFTER_SEARCH", "1"):
                    open_result = {"attempted": False, "clicked": False, "note": "V136 脱管模式：不再自动点击结果卡片；搜索结果/数据选择交给用户，避免 TPDC 被脚本拉回首页。"}
                    click_result = {"attempted": False, "clicked": False, "note": "V136 脱管模式：不自动点击下载/排序/订单按钮；用户手动选择数据和 FTP 入口。"}
                else:
                    open_result = _try_open_best_dataset_or_download(page, args.keyword, data_name=args.data_name, requested_resolution_meters=requested_resolution_meters, requested_year=requested_year)
                    print(f"[PLAYWRIGHT][OPEN][{phase}] {json.dumps(open_result, ensure_ascii=False)}")
                    if _is_truth("DOMESTIC_TPDC_DISABLE_AUTO_DOWNLOAD_CLICK", "1"):
                        click_result = {"attempted": False, "clicked": False, "note": "V135 两段式策略默认不自动点击下载/排序/订单按钮；进入数据选择后交给用户，后台只被动捕捉 FTP 账号。"}
                    else:
                        click_result = _try_auto_click_download(page, args.keyword)
                print(f"[PLAYWRIGHT][OPEN][{phase}] {json.dumps(open_result, ensure_ascii=False)}")
                print(f"[PLAYWRIGHT][CLICK][{phase}] {json.dumps(click_result, ensure_ascii=False)}")

                # If the download/order click leads to a login gate, fill credentials
                # there too. The verification itself remains manual.
                post_download_login_result = {"attempted": False, "note": "下载阶段未出现可填充登录表单。"}
                if use_env_login:
                    try:
                        if _wait_for_login_form(page, timeout_ms=5000):
                            update_state(status="download_login", message="下载需要登录，正在自动填充 TPDC 账号密码；验证码仍需人工输入。", captured_count=len(captured), phase=phase)
                            post_download_login_result = _try_fill_login_form(page, username, password)
                            print(f"[PLAYWRIGHT][DOWNLOAD_LOGIN][{phase}] {json.dumps(post_download_login_result, ensure_ascii=False)}")
                    except Exception as exc:
                        post_download_login_result = {"attempted": True, "note": f"下载阶段登录检测失败：{exc}"}
                if post_download_login_result.get("attempted"):
                    login_result = post_download_login_result

                ftp_accounts = _extract_ftp_accounts(page)
                ftp_result = {"detected": bool(ftp_accounts), "accounts_count": len(ftp_accounts)}
                if ftp_accounts:
                    try:
                        print("[TPDC_FTP_DEBUG][playwright] accounts=" + json.dumps(ftp_accounts, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
                    ftp_files = _write_ftp_handoff_files(download_dir, ftp_accounts)
                    ftp_result.update(ftp_files)
                    update_state(status="ftp_credentials_detected", message="已识别页面中的 TPDC FTP 用户名和密码。系统只使用 ftp2.tpdc.ac.cn:6201 进行下载，不尝试备用主机。", ftp_result=ftp_result, captured_count=len(captured), phase=phase)
                    if str(os.getenv("DOMESTIC_FTP_AUTO_DOWNLOAD", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                        ftp_errors = []
                        ftp_download = {}
                        for _i, _acc in enumerate((ftp_accounts or [])[:1]):
                            _role = "primary"
                            _role_label = "主主机"
                            try:
                                print(f"[TPDC_FTP_DEBUG][playwright_try_{_role}_{_i+1}] " + json.dumps(_acc, ensure_ascii=False), flush=True)
                            except Exception:
                                pass
                            ftp_download = _ftp_download_tree(_acc, download_dir, max_files=int(os.getenv("DOMESTIC_FTP_MAX_FILES", "9999")))
                            ftp_download["ftp_host_role"] = _role
                            ftp_download["ftp_host_role_label"] = _role_label
                            if ftp_download.get("ok"):
                                break
                            ftp_errors.append({"account_index": _i + 1, "ftp_host_role": _role, "ftp_host_role_label": _role_label, "host": _acc.get("host"), "username": _acc.get("username"), "error": ftp_download.get("error") or ftp_download.get("status")})
                        ftp_result["auto_download"] = ftp_download
                        ftp_result["auto_download_attempt_errors"] = ftp_errors
                        update_state(status=ftp_download.get("status") or "ftp_processed", message="已处理 FTP 下载；若未自动下载，请使用生成的 FTP 说明文件或在浏览器中操作。", ftp_result=ftp_result, captured_count=len(captured), phase=phase)
                print(f"[PLAYWRIGHT][FTP][{phase}] {json.dumps(ftp_result, ensure_ascii=False)}")

                dataset_signal = _page_has_dataset_signal(page, args.keyword, open_result=open_result, click_result=click_result, ftp_accounts=ftp_accounts, data_name=args.data_name, requested_resolution_meters=requested_resolution_meters, requested_year=requested_year, platform=platform)
                print(f"[PLAYWRIGHT][DATASET_SIGNAL][{phase}] {json.dumps(dataset_signal, ensure_ascii=False)}")
                # V127: submitting a TPDC search is NOT completion.
                # Completion means the agent has reached/selected a concrete dataset
                # or download/FTP signal. Otherwise keep retrying after result cards
                # finish rendering.
                current_page_ctx = _tpdc_page_context(page, args.keyword)
                # V128: reaching a TPDC dataset/search selection context with a
                # valid keyword is enough to stop re-submitting searches. The browser
                # must stay where the user can select/download data instead of being
                # yanked back to the home page by the resume loop.
                auto_sequence_done = bool(
                    open_result.get("clicked")
                    or click_result.get("clicked")
                    or ftp_result.get("detected")
                    or dataset_signal.get("matched")
                    or (current_page_ctx.get("on_search_or_dataset") and current_page_ctx.get("has_keyword") and not current_page_ctx.get("on_home"))
                    or search_result.get("submitted")
                )
                if auto_sequence_done and _is_truth("DOMESTIC_TPDC_MANUAL_TAKEOVER_AFTER_FIRST_NAV", "1"):
                    if not manual_takeover_lock:
                        manual_takeover_reason = (
                            "已提交数据名称检索" if search_result.get("submitted") and not open_result.get("clicked") else
                            "已进入/点击数据集候选" if open_result.get("clicked") else
                            "已检测到数据集或 FTP 信号" if (dataset_signal.get("matched") or ftp_result.get("detected")) else
                            "已进入 TPDC 数据上下文"
                        )
                    manual_takeover_lock = True
                    auto_sequence_done = True

                # V136: hard detach after the first successful search/data-context
                # transition. This deliberately ends Playwright control and reopens
                # the same profile in a normal Chrome window. It is the hard fix for
                # the user's reproducible failure: scrolling inside TPDC caused the
                # controlled page to jump back to the initial interface.
                if auto_sequence_done and _is_truth("DOMESTIC_TPDC_DETACH_AFTER_SEARCH", "1") and not detached_handoff_done:
                    handoff_url = str(getattr(page, "url", "") or "https://data.tpdc.ac.cn/")
                    update_state(
                        status="detaching_to_unmanaged_chrome",
                        message="已完成 TPDC 登录/检索，正在关闭受控浏览器并改用普通 Chrome 打开当前数据选择页。后续不会再有脚本重搜、重定向或点击页面。",
                        phase=phase,
                        search_result=search_result,
                        open_result=open_result,
                        click_result=click_result,
                        ftp_result=ftp_result,
                        dataset_signal=dataset_signal,
                        current_url=handoff_url,
                    )
                    try:
                        context.close()
                    except Exception:
                        pass
                    detached_handoff_result = _launch_detached_chrome_handoff(profile_dir, handoff_url, download_dir, timeout_seconds=args.timeout)
                    detached_handoff_done = True
                    update_state(
                        status="unmanaged_chrome_handoff",
                        message="V136 已切换到普通 Chrome 脱管模式。请在新打开的 Chrome 窗口中滚动和选择数据；系统不会再控制页面。若出现 FTP 账号块，请点击平台复制按钮或手动复制该块，剪贴板监视器会生成 FTP 下载脚本。",
                        phase=phase,
                        detached_handoff=detached_handoff_result,
                        search_result=search_result,
                        open_result=open_result,
                        click_result=click_result,
                        ftp_result=ftp_result,
                        dataset_signal=dataset_signal,
                        current_url=handoff_url,
                    )
                    return True
                try:
                    _safe_live_screenshot(page, live_screenshot)
                except Exception:
                    pass
                update_state(
                    status="waiting_user_or_download" if auto_sequence_done else "search_not_confirmed",
                    message=("V135 已完成一次数据名称检索/数据上下文进入，并冻结自动导航。请在浏览器中滚动、选择数据或点击下载；系统只被动捕捉 FTP 账号。" if auto_sequence_done else "尚未完成数据名称检索；登录界面消失后会再尝试一次。"),
                    phase=phase,
                    login_result=login_result,
                    search_result=search_result,
                    open_result=open_result,
                    click_result=click_result,
                    ftp_result=ftp_result,
                    dataset_signal=dataset_signal,
                    page_context=current_page_ctx,
                    captured_count=len(captured),
                    current_url=getattr(page, "url", ""),
                )
                return auto_sequence_done

            # First attempt. If captcha/login is still visible, this returns False
            # and the waiting loop below resumes automatically after the user logs in.
            try:
                run_tpdc_auto_sequence("initial_or_after_prelogin")
            except Exception as exc:
                update_state(status="initial_sequence_failed_but_browser_kept", message=f"初次自动检索失败，浏览器保持打开；登录后会继续重试：{exc}", captured_count=len(captured), current_url=getattr(page, 'url', ''))
            if detached_handoff_done:
                print("[PLAYWRIGHT] V136 已进入普通 Chrome 脱管模式，受控进程退出。")
                return 0
            if multi_platform and not dataset_signal.get("matched"):
                update_state(status="platform_rejected", message="当前候选平台未可靠确认对应数据，智能体将转入下一个平台。", dataset_signal=dataset_signal, captured_count=len(captured))
                try:
                    context.close()
                except Exception:
                    pass
                continue
            try:
                _safe_live_screenshot(page, live_screenshot)
            except Exception:
                pass
            update_state(status="waiting_user_or_download" if auto_sequence_done else "waiting_dataset_selection", message=("已进入 TPDC 数据集/下载选择流程。若平台要求验证码、确认或最终下载按钮，请在浏览器中完成。" if auto_sequence_done else "已登录并提交数据名称检索，正在等待结果卡片渲染并继续自动选择具体数据集。"), login_result=login_result, search_result=search_result, open_result=open_result, click_result=click_result, ftp_result=ftp_result, dataset_signal=dataset_signal, captured_count=0)
            _write_json(download_dir / "capture_session.json", {
                "platform": platform,
                "entry_url": entry_url,
                "profile_dir": str(profile_dir),
                "download_dir": str(download_dir),
                "started_at": started_at,
                "keyword_hint": args.keyword,
                "auto_login": use_env_login,
                "credential_username_env": user_env,
                "credential_username_masked": _mask_username(username),
                "login_result": login_result,
                "search_result": search_result,
                "open_result": open_result,
                "click_result": click_result,
                "ftp_result": ftp_result,
                "live_screenshot": str(live_screenshot),
                "state_json": str(state_json),
                "browser_channel": str(args.browser_channel or "chrome"),
                "browser_executable": args.browser_executable or _find_windows_chrome_executable(),
                "launched_browser": launched_browser,
                "note": "V135 两段式接管：登录后只执行一次搜索/进入数据选择；随后冻结所有浏览器导航动作，仅保留当前页/iframe 的被动 FTP 捕捉，避免滚轮或页面渲染后被脚本拉回首页。",
            })

            platform_timeout = args.per_platform_timeout if len(platforms) > 1 else args.timeout
            deadline = time.time() + max(platform_timeout, 30)
            last_shot = 0.0
            last_resume_attempt = 0.0
            last_ftp_probe = 0.0
            ftp_processed_signature = set()
            try:
                while time.time() < deadline:
                    # 如果用户关闭浏览器，则退出。
                    if not context.pages:
                        break
                    now = time.time()

                    # V127: resumable automation. In V124, once the script entered
                    # the captcha/login waiting state, it could remain on TPDC home
                    # after the user logged in. Here we actively resume the
                    # data-name search as soon as visible login controls disappear.
                    if (not auto_sequence_done) and (not manual_takeover_lock) and now - last_resume_attempt >= 3.0:
                        last_resume_attempt = now
                        try:
                            if not _page_has_visible_password_or_verification(page):
                                run_tpdc_auto_sequence("post_login_resume_loop")
                                if detached_handoff_done:
                                    print("[PLAYWRIGHT] V136 已进入普通 Chrome 脱管模式，受控进程退出。")
                                    return 0
                        except Exception as exc:
                            update_state(status="resume_search_failed", message=f"登录后自动恢复检索失败：{exc}", captured_count=len(captured), current_url=getattr(page, 'url', ''))

                    if _is_truth("DOMESTIC_FTP_TRIGGER_ON_ACCOUNT_BLOCK", "1") and now - last_ftp_probe >= 5.0:
                        last_ftp_probe = now
                        try:
                            visible_ftp_accounts = _extract_ftp_accounts(page)
                            if visible_ftp_accounts:
                                sig = tuple((a.get("host"), a.get("port"), a.get("username"), a.get("password")) for a in visible_ftp_accounts)
                                if sig not in ftp_processed_signature:
                                    ftp_processed_signature.add(sig)
                                    ftp_files = _write_ftp_handoff_files(download_dir, visible_ftp_accounts)
                                    ftp_result = {"detected": True, "accounts_count": len(visible_ftp_accounts), **ftp_files}
                                    update_state(status="ftp_credentials_detected", message="已在 TPDC 页面捕捉到 FTP 主机/端口/用户名/密码，正在写出连接文件并尝试自动 FTP 下载。", ftp_result=ftp_result, captured_count=len(captured), current_url=getattr(page, 'url', ''))
                                    if str(os.getenv("DOMESTIC_FTP_AUTO_DOWNLOAD", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                                        ftp_download = _ftp_download_tree(_select_primary_ftp_account(visible_ftp_accounts), download_dir, max_files=int(os.getenv("DOMESTIC_FTP_MAX_FILES", "9999")))
                                        ftp_result["auto_download"] = ftp_download
                                        update_state(status=ftp_download.get("status") or "ftp_processed", message="FTP 账号已处理。若 Python FTP 未成功，请使用生成的 open_ftp_client.bat / open_winscp_download.bat 接管。", ftp_result=ftp_result, captured_count=len(captured), current_url=getattr(page, 'url', ''))
                        except Exception as exc:
                            update_state(status="ftp_probe_failed", message=f"FTP 账号捕捉失败：{exc}", captured_count=len(captured), current_url=getattr(page, 'url', ''))

                    if now - last_shot >= 20.0:
                        try:
                            _safe_live_screenshot(page, live_screenshot)
                            update_state(
                                status="waiting_user_or_download" if auto_sequence_done else "waiting_login_or_resume",
                                message=("V135 被动接管中：浏览器已交给用户操作；系统不会再滚动、重搜、点击或改变 URL，只做低频截图与 FTP 账号捕捉。" if manual_takeover_lock else "TPDC 页面低频投射中；登录界面消失后只尝试一次数据名称检索。"),
                                captured_count=len(captured),
                                current_url=page.url,
                                search_result=search_result,
                                open_result=open_result,
                                click_result=click_result,
                                ftp_result=ftp_result,
                                dataset_signal=dataset_signal,
                            )
                        except Exception as exc:
                            update_state(status="screenshot_failed", message=f"页面投射截图失败：{exc}", captured_count=len(captured), current_url=getattr(page, 'url', ''))
                        last_shot = now
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n[PLAYWRIGHT] 收到 Ctrl+C，准备关闭浏览器。")
            finally:
                # V134: do not close the browser by default. The user reported
                # that after captcha login the automation window could disappear
                # before reaching the data-selection page. Keep the browser open
                # unless explicitly disabled, so manual takeover remains possible.
                if not _is_truth("DOMESTIC_KEEP_BROWSER_OPEN_AFTER_CAPTURE", "1"):
                    try:
                        context.close()
                    except Exception:
                        pass
            print(f"[PLAYWRIGHT] 平台 {platform} 捕获完成，下载数量：{len(captured)}，目录：{download_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
