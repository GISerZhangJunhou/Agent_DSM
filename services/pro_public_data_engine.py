from __future__ import annotations

"""
PRO 版公开数据自动获取处理引擎。

设计边界：
1. 不爬取、不猜测、不购买土壤有机质样点；SOM 样点由用户上传。
2. 只访问公开页面和公开下载链接；登录/验证码/审批留给人工。
3. 下载数据写入任务临时目录，并生成 provenance/availability 报告。
4. 本模块是“能否爬取并使用公开协变量/统计数据”的当前版本，不是最终生产爬虫。
"""

import csv
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import requests

from utils.pro_console import pro_console_log
from services.sample_reader import validate_uploaded_file

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36 ProDataTest/0.1"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
}

DOWNLOAD_EXTS = (".zip", ".rar", ".7z", ".xls", ".xlsx", ".csv", ".tif", ".tiff", ".nc", ".json", ".geojson")
HTML_EXTS = (".htm", ".html", ".shtml", ".jsp", "")


@dataclass
class ProDataTaskSpec:
    request_text: str
    year: int | None = None
    region: str = "四川省"
    target: str = "SOM"
    resolution: str = "250m"
    sample_path: str | None = None
    keep_raw: bool = True
    allow_download: bool = True
    max_download_mb: int = 500


@dataclass
class DataRecord:
    name: str
    source: str
    category: str
    status: str
    url: str | None = None
    local_path: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProDataRunResult:
    ok: bool
    status: str
    work_dir: str
    manifest: str
    report_md: str
    records: list[dict[str, Any]]
    sample_check: dict[str, Any]
    usable_outputs: dict[str, Any]
    warnings: list[str]


class PublicDataEngine:
    def __init__(self, base_dir: Path, timeout: int = 30):
        self.base_dir = Path(base_dir)
        self.timeout = int(os.getenv("PRO_PUBLIC_HTTP_TIMEOUT", str(timeout)))
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.records: list[DataRecord] = []
        self.warnings: list[str] = []

    def _emit(self, stage: str, message: str, payload: Any | None = None) -> None:
        pro_console_log(stage, message, payload=payload)

    def run(self, spec: ProDataTaskSpec) -> ProDataRunResult:
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_region = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", spec.region or "region")[:32]
        year_label = str(spec.year or "unknown_year")
        work_dir = self.base_dir / f"pro_public_test_{safe_region}_{year_label}_{ts}"
        raw_dir = work_dir / "raw_tmp"
        processed_dir = work_dir / "processed_tmp"
        meta_dir = work_dir / "metadata"
        for p in [raw_dir, processed_dir, meta_dir]:
            p.mkdir(parents=True, exist_ok=True)

        self._emit("PRO_ENGINE", "公开数据获取启动", {
            "request_text": spec.request_text,
            "year": spec.year,
            "region": spec.region,
            "target": spec.target,
            "sample_path": spec.sample_path,
            "work_dir": str(work_dir),
            "allow_download": spec.allow_download,
            "max_download_mb": spec.max_download_mb,
        })

        sample_check = self._validate_sample(spec.sample_path, meta_dir)
        if not sample_check.get("ok"):
            self.warnings.append(sample_check.get("message") or "未提供或未通过样点检查；当前版本仍会继续下载公开协变量/统计数据。")
            self._emit("SAMPLE", "样点读取/检查未通过或未提供", sample_check)
        else:
            self._emit("SAMPLE", "样点读取/检查成功", sample_check)

        # 1) 四川统计年鉴：优先验证真实可下载/可解析的农业统计代理变量。
        if "四川" in (spec.region or ""):
            self._crawl_sichuan_yearbook(spec, raw_dir, processed_dir)
        else:
            self.records.append(DataRecord(
                name="四川统计年鉴适配器",
                source="四川省统计局",
                category="management_proxy/statistical_yearbook",
                status="skipped",
                message="当前当前版本只对四川统计年鉴做了定向适配；其他省份后续扩展。",
            ))

        # 2) NODA：只做公开检索页面探测，不自动绕过登录/验证码。
        self._probe_noda(spec, raw_dir)

        # 3) 农业农村部：做公开页面可访问性探测，后续再接具体接口/表格。
        self._probe_moa(spec, raw_dir)

        usable_outputs = self._collect_usable_outputs(processed_dir)
        manifest_path = meta_dir / "pro_public_data_manifest.json"
        report_path = meta_dir / "data_availability_report.md"

        payload = {
            "created_at": int(time.time()),
            "created_at_text": time.strftime("%Y-%m-%d %H:%M:%S"),
            "spec": asdict(spec),
            "policy": {
                "sample_strategy": "user_upload_only",
                "captcha_strategy": "email_or_sms_forward_code_via_qq_mail; graphical_or_slider_captcha_manual",
                "verification_strategy": "QQ邮箱IMAP读取平台邮件验证码或短信转发验证码；失败后人工输入",
                "login_strategy": "configured_account_or_manual_browser_takeover",
                "storage_strategy": "task_temp_cache; keep outputs and provenance",
            },
            "sample_check": sample_check,
            "records": [asdict(r) for r in self.records],
            "usable_outputs": usable_outputs,
            "warnings": self.warnings,
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(self._build_report(payload), encoding="utf-8")

        ok = any(r.status in {"downloaded", "parsed", "saved", "available"} for r in self.records)
        self._emit("PRO_ENGINE", "公开数据获取结束", {
            "ok": ok,
            "status": "done" if ok else "no_public_data_downloaded",
            "record_count": len(self.records),
            "manifest": str(manifest_path),
            "report_md": str(report_path),
            "processed_file_count": usable_outputs.get("processed_file_count"),
            "warnings": self.warnings,
        })
        return ProDataRunResult(
            ok=ok,
            status="done" if ok else "no_public_data_downloaded",
            work_dir=str(work_dir),
            manifest=str(manifest_path),
            report_md=str(report_path),
            records=[asdict(r) for r in self.records],
            sample_check=sample_check,
            usable_outputs=usable_outputs,
            warnings=self.warnings,
        )

    # ------------------------- 样点检查 -------------------------
    def _validate_sample(self, sample_path: str | None, meta_dir: Path) -> dict[str, Any]:
        """复用正式上传层校验能力，避免诊断层和生产层逻辑不一致。"""
        if not sample_path:
            return {
                "ok": False,
                "message": "未提供样点文件。PRO 当前版本不会联网爬取 SOM 样点；正式制图前必须上传样点 CSV/Excel/TIF。",
                "required_fields": ["lon", "lat", "som"],
            }
        out = validate_uploaded_file(sample_path)
        try:
            (meta_dir / "sample_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return out

    # ------------------------- 四川统计年鉴 -------------------------
    def _crawl_sichuan_yearbook(self, spec: ProDataTaskSpec, raw_dir: Path, processed_dir: Path) -> None:
        year = spec.year or 2018
        source = "四川省统计局统计年鉴"
        self._emit("NETWORK", "开始访问四川统计年鉴平台", {"year": year, "region": spec.region})
        index_url = "https://tjj.sc.gov.cn/scstjj/c105855/nj.shtml"
        year_dir = raw_dir / "sichuan_yearbook"
        year_dir.mkdir(parents=True, exist_ok=True)

        self._emit("NETWORK", "请求网页", {"platform": "四川省统计局统计年鉴", "url": index_url})
        html, err = self._get_text(index_url)
        if err:
            self.records.append(DataRecord(
                name="四川统计年鉴索引页",
                source=source,
                category="management_proxy/statistical_yearbook",
                status="error",
                url=index_url,
                message=err,
            ))
            return
        index_path = year_dir / "sichuan_yearbook_index.html"
        index_path.write_text(html, encoding="utf-8", errors="ignore")
        self.records.append(DataRecord(
            name="四川统计年鉴索引页",
            source=source,
            category="management_proxy/statistical_yearbook",
            status="saved",
            url=index_url,
            local_path=str(index_path),
            message="已保存年鉴索引页。",
        ))

        year_links = self._extract_links(html, index_url)
        target_links = [(text, url) for text, url in year_links if f"{year}年" in text or str(year) in text]
        self._emit("NETWORK", "四川统计年鉴年份链接识别完成", {"year": year, "candidate_count": len(target_links), "candidates": target_links[:5]})
        if not target_links:
            self.records.append(DataRecord(
                name=f"四川统计年鉴 {year}",
                source=source,
                category="management_proxy/statistical_yearbook",
                status="not_found",
                url=index_url,
                message="索引页未找到对应年份链接。",
            ))
            return

        # 只取最相关的第一个年份链接。部分年份直接指向 rar/zip，部分年份指向网页。
        _, target_url = target_links[0]
        if self._is_download_url(target_url):
            rec = self._download_url(target_url, year_dir, f"sichuan_yearbook_{year}", source, "management_proxy/statistical_yearbook", spec)
            self.records.append(rec)
            self._try_unpack_or_index(rec, processed_dir / "sichuan_yearbook")
            return

        page_html, page_err = self._get_text(target_url)
        if page_err:
            self.records.append(DataRecord(
                name=f"四川统计年鉴 {year} 页面",
                source=source,
                category="management_proxy/statistical_yearbook",
                status="error",
                url=target_url,
                message=page_err,
            ))
            return

        page_path = year_dir / f"sichuan_yearbook_{year}_page.html"
        page_path.write_text(page_html, encoding="utf-8", errors="ignore")
        self.records.append(DataRecord(
            name=f"四川统计年鉴 {year} 页面",
            source=source,
            category="management_proxy/statistical_yearbook",
            status="saved",
            url=target_url,
            local_path=str(page_path),
            message="已保存年份页面。",
        ))

        links = self._extract_links(page_html, target_url)
        file_links = [(text, url) for text, url in links if self._is_download_url(url)]
        if file_links:
            for text, file_url in file_links[:5]:
                rec = self._download_url(file_url, year_dir, f"sichuan_yearbook_{year}_{self._slug(text)[:24]}", source, "management_proxy/statistical_yearbook", spec)
                self.records.append(rec)
                self._try_unpack_or_index(rec, processed_dir / "sichuan_yearbook")

        # HTML 年鉴通常还有 zk/html/lefte.htm 和章节表格；递归抓取农业章节候选页。
        self._crawl_sichuan_yearbook_html_tables(year, target_url, page_html, year_dir, processed_dir / "sichuan_yearbook")

    def _crawl_sichuan_yearbook_html_tables(self, year: int, target_url: str, page_html: str, raw_dir: Path, processed_dir: Path) -> None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        source = "四川省统计局统计年鉴HTML"
        links = self._extract_links(page_html, target_url)
        # 有些年鉴入口直接是 rar；也尝试固定 HTML 目录。
        candidate_urls = [url for text, url in links if any(k in text for k in ["农业", "十三", "目录", "HTML", "网页"])]
        fixed_candidates = [
            f"https://tjj.sc.gov.cn/tjnj/cs/{year}/lefte.htm",
            f"https://tjj.sc.gov.cn/tjnj/cs/{year}/zk/html/lefte.htm",
            f"https://tjj.sc.gov.cn/tjnj/cs/{year}/zk/html/note.htm",
        ]
        for u in fixed_candidates:
            if u not in candidate_urls:
                candidate_urls.append(u)

        visited = set()
        html_files: list[Path] = []
        for url in candidate_urls[:8]:
            if url in visited:
                continue
            visited.add(url)
            self._emit("NETWORK", "访问四川统计年鉴 HTML 候选页", {"year": year, "url": url})
            html, err = self._get_text(url)
            if err:
                continue
            p = raw_dir / (self._slug(Path(urlparse(url).path).name or "page") + ".html")
            p.write_text(html, encoding="utf-8", errors="ignore")
            html_files.append(p)
            self.records.append(DataRecord(
                name=f"四川统计年鉴HTML候选页 {year}",
                source=source,
                category="management_proxy/statistical_yearbook_html",
                status="saved",
                url=url,
                local_path=str(p),
                message="已保存 HTML 候选页。",
            ))
            for text, child in self._extract_links(html, url):
                if child in visited:
                    continue
                # 农业章节通常包含 13；同时保留含“农业”的链接。
                if "13" in child or "农业" in text:
                    child_html, child_err = self._get_text(child)
                    if child_err:
                        continue
                    visited.add(child)
                    cp = raw_dir / ("yearbook_agri_" + self._slug(Path(urlparse(child).path).name or str(len(html_files))) + ".html")
                    cp.write_text(child_html, encoding="utf-8", errors="ignore")
                    html_files.append(cp)
                    self.records.append(DataRecord(
                        name=f"四川统计年鉴农业章节候选页 {year}",
                        source=source,
                        category="management_proxy/agriculture_statistics",
                        status="saved",
                        url=child,
                        local_path=str(cp),
                        message="已保存农业章节候选 HTML。",
                    ))
                    if len(html_files) >= 20:
                        break
            if len(html_files) >= 20:
                break

        # 尝试将 HTML 表格转 CSV。没有表格也不报错。
        if pd is None:
            return
        table_count = 0
        for html_path in html_files:
            try:
                tables = pd.read_html(str(html_path))
            except Exception:
                continue
            for i, df in enumerate(tables):
                if df.empty:
                    continue
                out = processed_dir / f"{html_path.stem}_table_{i+1}.csv"
                df.to_csv(out, index=False, encoding="utf-8-sig")
                table_count += 1
                self.records.append(DataRecord(
                    name="四川统计年鉴HTML表格转CSV",
                    source=source,
                    category="management_proxy/agriculture_statistics",
                    status="parsed",
                    local_path=str(out),
                    size_bytes=out.stat().st_size,
                    message=f"已从 HTML 提取表格：{html_path.name} / table {i+1}",
                    metadata={"rows": int(len(df)), "columns": [str(c) for c in df.columns]},
                ))
        if table_count == 0:
            self.records.append(DataRecord(
                name="四川统计年鉴HTML表格转CSV",
                source=source,
                category="management_proxy/agriculture_statistics",
                status="no_table_parsed",
                message="已尝试解析 HTML 表格，但未得到可用表格；可能需要改进年份页面适配规则。",
            ))

    # ------------------------- NODA/MOA 探测 -------------------------
    def _probe_noda(self, spec: ProDataTaskSpec, raw_dir: Path) -> None:
        keyword = f"{spec.region} {spec.year or ''} 遥感 NDVI 土地覆盖".strip()
        url = "https://www.noda.ac.cn/portal/indexSearch?keyword=" + quote(keyword)
        self._emit("NETWORK", "开始访问 NODA 检索页", {"keyword": keyword, "url": url})
        html, err = self._get_text(url)
        out_dir = raw_dir / "noda_probe"
        out_dir.mkdir(parents=True, exist_ok=True)
        if err:
            self.records.append(DataRecord(
                name="NODA公开检索页探测",
                source="国家综合地球观测数据共享平台 NODA",
                category="remote_sensing/search_probe",
                status="error",
                url=url,
                message=err,
            ))
            return
        p = out_dir / "noda_search.html"
        p.write_text(html, encoding="utf-8", errors="ignore")
        links = self._extract_links(html, url)
        self.records.append(DataRecord(
            name="NODA公开检索页探测",
            source="国家综合地球观测数据共享平台 NODA",
            category="remote_sensing/search_probe",
            status="saved",
            url=url,
            local_path=str(p),
            size_bytes=p.stat().st_size,
            message="已保存 NODA 检索页；若下载需要登录、审核或验证码，当前版本只记录人工介入状态，不绕过。",
            metadata={"keyword": keyword, "link_count": len(links)},
        ))

    def _probe_moa(self, spec: ProDataTaskSpec, raw_dir: Path) -> None:
        # 农业农村部网站入口稳定，但具体数据频道可能改版。当前版本先记录入口可访问性和链接。
        url = "https://www.moa.gov.cn/"
        self._emit("NETWORK", "开始访问农业农村部公开页面", {"url": url})
        html, err = self._get_text(url)
        out_dir = raw_dir / "moa_probe"
        out_dir.mkdir(parents=True, exist_ok=True)
        if err:
            self.records.append(DataRecord(
                name="农业农村部公开页面探测",
                source="农业农村部",
                category="agriculture_statistics/search_probe",
                status="error",
                url=url,
                message=err,
            ))
            return
        p = out_dir / "moa_home.html"
        p.write_text(html, encoding="utf-8", errors="ignore")
        links = self._extract_links(html, url)
        data_links = [(t, u) for t, u in links if any(k in (t + u) for k in ["数据", "统计", "种植", "农业", "下载"])]
        (out_dir / "moa_candidate_links.json").write_text(json.dumps(data_links[:100], ensure_ascii=False, indent=2), encoding="utf-8")
        self.records.append(DataRecord(
            name="农业农村部公开页面探测",
            source="农业农村部",
            category="agriculture_statistics/search_probe",
            status="saved",
            url=url,
            local_path=str(p),
            size_bytes=p.stat().st_size,
            message="已保存农业农村部首页并提取数据/统计候选链接。",
            metadata={"candidate_link_count": len(data_links)},
        ))

    # ------------------------- 文件处理与工具 -------------------------
    def _get_text(self, url: str) -> tuple[str, str | None]:
        try:
            self._emit("HTTP", "GET 文本页面", {"url": url})
            r = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            r.raise_for_status()
            if not r.encoding or r.encoding.lower() in {"iso-8859-1", "ascii"}:
                r.encoding = r.apparent_encoding or "utf-8"
            self._emit("HTTP", "GET 成功", {"url": url, "status_code": r.status_code, "encoding": r.encoding, "chars": len(r.text or "")})
            return r.text, None
        except Exception as exc:
            self._emit("HTTP", "GET 失败", {"url": url, "error": str(exc)})
            return "", str(exc)

    def _download_url(self, url: str, out_dir: Path, name_hint: str, source: str, category: str, spec: ProDataTaskSpec) -> DataRecord:
        out_dir.mkdir(parents=True, exist_ok=True)
        parsed_name = Path(urlparse(url).path).name
        suffix = Path(parsed_name).suffix.lower()
        if not suffix:
            suffix = ".bin"
        filename = self._slug(name_hint) + suffix
        dst = out_dir / filename
        if not spec.allow_download:
            return DataRecord(name=name_hint, source=source, category=category, status="planned", url=url, message="allow_download=False，未实际下载。")
        try:
            self._emit("DOWNLOAD", "开始下载文件", {"url": url, "name_hint": name_hint, "out_dir": str(out_dir), "max_download_mb": spec.max_download_mb})
            with self.session.get(url, timeout=self.timeout, stream=True, allow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                max_bytes = max(1, spec.max_download_mb) * 1024 * 1024
                if total > max_bytes:
                    return DataRecord(name=name_hint, source=source, category=category, status="skipped_too_large", url=url, size_bytes=total, message=f"文件超过限制 {spec.max_download_mb} MB，已跳过。")
                h = hashlib.sha256()
                size = 0
                with dst.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > max_bytes:
                            f.close()
                            try:
                                dst.unlink()
                            except Exception:
                                pass
                            return DataRecord(name=name_hint, source=source, category=category, status="skipped_too_large", url=url, size_bytes=size, message=f"下载中超过限制 {spec.max_download_mb} MB，已停止。")
                        h.update(chunk)
                        f.write(chunk)
            meta = self._inspect_file(dst)
            self._emit("DOWNLOAD", "文件下载成功", {"url": url, "local_path": str(dst), "size_bytes": dst.stat().st_size, "metadata": meta})
            return DataRecord(
                name=name_hint,
                source=source,
                category=category,
                status="downloaded",
                url=url,
                local_path=str(dst),
                size_bytes=dst.stat().st_size,
                sha256=h.hexdigest(),
                message="文件下载成功并完成基础检查。",
                metadata=meta,
            )
        except Exception as exc:
            self._emit("DOWNLOAD", "文件下载失败", {"url": url, "local_path": str(dst), "error": str(exc)})
            return DataRecord(name=name_hint, source=source, category=category, status="error", url=url, local_path=str(dst), message=str(exc))

    def _try_unpack_or_index(self, rec: DataRecord, processed_dir: Path) -> None:
        if not rec.local_path or rec.status != "downloaded":
            return
        p = Path(rec.local_path)
        processed_dir.mkdir(parents=True, exist_ok=True)
        suffix = p.suffix.lower()
        if suffix == ".zip":
            try:
                with zipfile.ZipFile(p, "r") as zf:
                    names = zf.namelist()
                    index_path = processed_dir / (p.stem + "_zip_index.json")
                    index_path.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
                    # 只自动解压小型表格文件，避免原始数据失控膨胀。
                    for name in names[:80]:
                        if Path(name).suffix.lower() in {".csv", ".xls", ".xlsx", ".html", ".htm"}:
                            zf.extract(name, processed_dir)
                self.records.append(DataRecord(
                    name="ZIP文件索引/轻量解压",
                    source=rec.source,
                    category=rec.category,
                    status="parsed",
                    local_path=str(processed_dir),
                    message="已生成 ZIP 文件索引，并尝试解压轻量表格/HTML 文件。",
                    metadata={"source_file": str(p)},
                ))
            except Exception as exc:
                self.records.append(DataRecord(name="ZIP文件索引", source=rec.source, category=rec.category, status="parse_error", local_path=str(p), message=str(exc)))
        elif suffix in {".rar", ".7z"}:
            # Python 标准库不能可靠解压 rar/7z；当前版本先记录索引需求，避免强依赖 unrar/7zip。
            self.records.append(DataRecord(
                name="压缩包待人工/外部工具解压",
                source=rec.source,
                category=rec.category,
                status="available",
                local_path=str(p),
                message="已下载压缩包。RAR/7Z 解压需要本机安装 7-Zip/unrar；当前版本不强制解压。",
                metadata={"suggested_tool": "7-Zip 或 WinRAR"},
            ))
        elif suffix in {".csv", ".xls", ".xlsx", ".html", ".htm"}:
            try:
                shutil.copy2(p, processed_dir / p.name)
            except Exception:
                pass

    def _inspect_file(self, path: Path) -> dict[str, Any]:
        meta: dict[str, Any] = {"suffix": path.suffix.lower()}
        try:
            meta["size_bytes"] = path.stat().st_size
        except Exception:
            pass
        suffix = path.suffix.lower()
        if suffix == ".zip":
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    meta["zip_file_count"] = len(zf.namelist())
                    meta["zip_first_files"] = zf.namelist()[:20]
            except Exception as exc:
                meta["zip_error"] = str(exc)
        elif suffix in {".csv", ".xls", ".xlsx"} and pd is not None:
            try:
                df = pd.read_csv(path, nrows=5) if suffix == ".csv" else pd.read_excel(path, nrows=5)
                meta["columns"] = [str(c) for c in df.columns]
                meta["preview_rows"] = int(len(df))
            except Exception as exc:
                meta["table_error"] = str(exc)
        elif suffix in {".tif", ".tiff"} and rasterio is not None:
            try:
                with rasterio.open(path) as src:
                    meta.update({
                        "width": src.width,
                        "height": src.height,
                        "count": src.count,
                        "crs": str(src.crs),
                        "bounds": list(src.bounds),
                        "res": list(src.res),
                    })
            except Exception as exc:
                meta["raster_error"] = str(exc)
        return meta

    def _collect_usable_outputs(self, processed_dir: Path) -> dict[str, Any]:
        files = []
        if processed_dir.exists():
            for p in processed_dir.rglob("*"):
                if p.is_file():
                    files.append({"path": str(p), "suffix": p.suffix.lower(), "size_bytes": p.stat().st_size})
        return {
            "processed_file_count": len(files),
            "processed_files": files[:200],
            "can_start_modeling": False,
            "modeling_blocker": "当前版本只验证公开数据获取与解析；正式 RFK 仍需要用户样点 + 栅格协变量堆栈完成后再启动。",
        }

    def _build_report(self, payload: dict[str, Any]) -> str:
        spec = payload.get("spec", {})
        lines = [
            "# PRO公开数据自动获取报告",
            "",
            f"- 请求：{spec.get('request_text', '')}",
            f"- 区域：{spec.get('region', '')}",
            f"- 年份：{spec.get('year', '')}",
            f"- 目标变量：{spec.get('target', '')}",
            "- 样点策略：用户上传；系统不联网爬取 SOM 样点。",
            "- 验证码策略：邮箱验证码/短信转发验证码可通过 QQ 邮箱 IMAP 辅助读取；图形验证码、滑块验证码、人机验证仍由用户人工完成。",
            "",
            "## 样点检查",
            "",
            "```json",
            json.dumps(payload.get("sample_check", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## 数据记录",
            "",
            "| 状态 | 类别 | 名称 | 来源 | 本地路径/说明 |",
            "|---|---|---|---|---|",
        ]
        for r in payload.get("records", []):
            msg = r.get("local_path") or r.get("message") or ""
            msg = str(msg).replace("|", "｜")[:260]
            lines.append(f"| {r.get('status')} | {r.get('category')} | {r.get('name')} | {r.get('source')} | {msg} |")
        lines += ["", "## 警告/限制", ""]
        warnings = payload.get("warnings", []) or ["无"]
        for w in warnings:
            lines.append(f"- {w}")
        lines += ["", "## 可用输出", "", "```json", json.dumps(payload.get("usable_outputs", {}), ensure_ascii=False, indent=2), "```", ""]
        return "\n".join(lines)

    def _extract_links(self, html: str, base_url: str) -> list[tuple[str, str]]:
        links: list[tuple[str, str]] = []
        # 简单正则足够用于政府静态页；动态页面只做探测保存。
        pattern = re.compile(r"<a\b[^>]*?href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.I | re.S)
        for href, text_html in pattern.findall(html or ""):
            if href.startswith("javascript:") or href.startswith("#"):
                continue
            text = re.sub(r"<.*?>", "", text_html)
            text = re.sub(r"\s+", " ", text).strip()
            links.append((text, urljoin(base_url, href)))
        return links

    def _is_download_url(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        return path.endswith(DOWNLOAD_EXTS)

    def _slug(self, s: str) -> str:
        s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(s or "file"))
        return s.strip("._-")[:80] or "file"


def parse_request_to_spec(request_text: str, sample_path: str | None = None) -> ProDataTaskSpec:
    text = request_text or ""
    year_match = re.search(r"(19\d{2}|20\d{2})", text)
    year = int(year_match.group(1)) if year_match else None
    region = "四川省"
    for cand in ["四川省", "四川", "成都市", "成都", "重庆市", "重庆", "云南省", "云南", "贵州省", "贵州"]:
        if cand in text:
            region = cand if cand.endswith(("省", "市")) else (cand + "省" if cand not in ["成都", "重庆"] else ("成都市" if cand == "成都" else "重庆市"))
            break
    target = "SOM" if any(k in text.lower() for k in ["som", "有机质", "土壤有机质"]) else "soil_mapping"
    spec = ProDataTaskSpec(
        request_text=text,
        year=year,
        region=region,
        target=target,
        sample_path=sample_path,
        keep_raw=os.getenv("PRO_KEEP_RAW_DATA", "1") == "1",
        allow_download=os.getenv("PRO_PUBLIC_ALLOW_DOWNLOAD", "1") == "1",
        max_download_mb=int(os.getenv("PRO_PUBLIC_MAX_DOWNLOAD_MB", "500")),
    )
    pro_console_log("INTENT", "用户提问解析完成", {
        "request_text": text,
        "parsed_year": spec.year,
        "parsed_region": spec.region,
        "target": spec.target,
        "sample_path": spec.sample_path,
    })
    return spec


def run_public_data_test(base_dir: Path, request_text: str, sample_path: str | None = None) -> ProDataRunResult:
    spec = parse_request_to_spec(request_text, sample_path=sample_path or os.getenv("PRO_TEST_SAMPLE_PATH") or None)
    engine = PublicDataEngine(base_dir=base_dir)
    return engine.run(spec)
