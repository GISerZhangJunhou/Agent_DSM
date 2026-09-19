import base64
import re
import zipfile
from pathlib import Path

from config.settings import UPLOAD_DIR
from utils.pro_console import pro_console_log
from services.sample_reader import validate_uploaded_file, build_upload_feedback

_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_ALLOWED_ZIP_SUFFIXES = {".csv", ".txt", ".xls", ".xlsx", ".tif", ".tiff", ".nc", ".hdf", ".h5", ".hdf5", ".shp", ".dbf", ".shx", ".prj", ".cpg", ".geojson", ".gpkg"}


def _safe_filename(filename: str | None, fallback: str = "upload.bin") -> str:
    """Return a Windows-safe basename for uploads.

    This fixes OSError(22, 'Invalid argument') caused by browser fake paths,
    colons, slashes, control characters, Windows reserved names and very long names.
    """
    raw = str(filename or "").strip().replace("\\", "/")
    raw = raw.split("/")[-1].strip() or fallback
    raw = re.sub(r"[\x00-\x1f\x7f]", "_", raw)
    raw = re.sub(r'[<>:"/\\|?*]+', "_", raw)
    raw = raw.strip(" .") or fallback
    stem = Path(raw).stem.strip(" .") or "upload"
    suffix = Path(raw).suffix
    if stem.upper() in _WINDOWS_RESERVED_NAMES:
        stem = f"_{stem}"
    max_stem = max(16, 120 - len(suffix))
    if len(stem) > max_stem:
        stem = stem[:max_stem]
    return stem + suffix


def _dedup_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 1000):
        cand = path.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
    return path.with_name(f"{stem}_{int(path.stat().st_mtime_ns)}{suffix}")


def _normalize_upload_lists(contents_list, filename_list):
    if isinstance(contents_list, str):
        contents_list = [contents_list]
    if isinstance(filename_list, str):
        filename_list = [filename_list]
    return list(contents_list or []), list(filename_list or [])


def _guess_upload_role(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    lower = str(filename or "").lower()
    if suffix in {".csv", ".txt", ".xls", ".xlsx"}:
        return "tabular_sample_or_attribute_table"
    if suffix in {".tif", ".tiff"}:
        if any(k in lower for k in ["mask", "掩膜", "boundary", "admin", "crop", "cropland"]):
            return "raster_mask_or_boundary"
        return "raster_sample_or_covariate"
    if suffix in {".nc", ".hdf", ".h5", ".hdf5"}:
        return "multidimensional_covariate_need_conversion"
    if suffix == ".zip":
        return "archive_package"
    if suffix in {".shp", ".gpkg", ".geojson", ".json"}:
        return "vector_boundary_or_sample"
    return "unknown"


def _safe_extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                continue
            if Path(name).suffix.lower() not in _ALLOWED_ZIP_SUFFIXES:
                continue
            safe_parts = [_safe_filename(part, fallback="part") for part in Path(name).parts]
            target = _dedup_path(out_dir.joinpath(*safe_parts))
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return extracted


def _validate_existing_file(path: Path, original_name: str | None = None, parent_zip: str | None = None) -> dict:
    info = {
        "name": path.name,
        "original_name": original_name or path.name,
        "path": str(path),
        "size": path.stat().st_size,
        "role_guess": _guess_upload_role(path.name),
    }
    if parent_zip:
        info["parent_archive"] = parent_zip
    try:
        validation = validate_uploaded_file(path)
        info["validation"] = validation
        info["usable_for_modeling"] = bool(validation.get("ok"))
        info["user_feedback"] = build_upload_feedback(info)
    except Exception as exc:
        # Do not crash upload because a non-sample covariate cannot pass sample CSV checks.
        info["validation"] = {"ok": False, "stage": "role_only", "message": f"文件已保存；样点校验不适用或读取异常：{exc}"}
        info["usable_for_modeling"] = False
        info["user_feedback"] = f"文件已保存：{path.name}\n系统角色猜测：{info['role_guess']}\n说明：该文件暂未作为训练样点校验通过。"
    return info


def save_dash_upload(contents: str, filename: str, session_id: str):
    safe_name = _safe_filename(filename)
    pro_console_log("UPLOAD", "开始保存用户上传文件", {"filename": filename, "safe_filename": safe_name, "session_id": session_id})
    if "," not in str(contents):
        raise ValueError("上传内容格式异常：未收到 Dash base64 数据。")
    _, b64data = str(contents).split(",", 1)
    data = base64.b64decode(b64data.encode("utf-8"))
    session_dir = UPLOAD_DIR / _safe_filename(session_id, fallback="session")
    session_dir.mkdir(parents=True, exist_ok=True)
    out = _dedup_path(session_dir / safe_name)
    out.write_bytes(data)
    info = _validate_existing_file(out, original_name=filename)
    pro_console_log("UPLOAD", "用户上传文件保存完成", {
        "name": filename,
        "safe_name": safe_name,
        "path": str(out),
        "size": out.stat().st_size,
        "role_guess": info.get("role_guess"),
        "validation_ok": (info.get("validation") or {}).get("ok"),
        "validation_stage": (info.get("validation") or {}).get("stage"),
    })
    return info


def save_multiple_uploads(contents_list, filename_list, session_id: str):
    contents_list, filename_list = _normalize_upload_lists(contents_list, filename_list)
    saved = []
    if not contents_list or not filename_list:
        pro_console_log("UPLOAD", "未检测到上传内容或文件名", {"session_id": session_id})
        return saved
    pro_console_log("UPLOAD", "收到上传请求", {"session_id": session_id, "file_count": len(filename_list), "filenames": filename_list})
    for c, n in zip(contents_list, filename_list):
        try:
            item = save_dash_upload(c, n, session_id)
            saved.append(item)
        except Exception as exc:
            err_item = {
                "name": str(n or "未知文件"),
                "original_name": str(n or "未知文件"),
                "path": "",
                "size": 0,
                "upload_failed": True,
                "upload_error": str(exc),
                "role_guess": _guess_upload_role(str(n or "")),
                "user_feedback": f"上传失败：{n or '未知文件'}\n原因：{exc}",
            }
            saved.append(err_item)
            pro_console_log("UPLOAD", "单个文件上传失败，已继续处理后续文件", {"filename": n, "error": str(exc), "session_id": session_id})
            continue
        try:
            path = Path(item.get("path") or "")
            if path.exists() and path.suffix.lower() == ".zip":
                extract_dir = path.parent / (path.stem + "_extracted")
                for child in _safe_extract_zip(path, extract_dir):
                    saved.append(_validate_existing_file(child, parent_zip=path.name))
        except Exception as exc:
            saved.append({
                "name": f"{item.get('name') or n} 内部文件",
                "original_name": f"{item.get('original_name') or n} 内部文件",
                "path": "",
                "size": 0,
                "upload_failed": True,
                "upload_error": f"压缩包展开失败：{exc}",
                "role_guess": "archive_package",
                "user_feedback": f"压缩包展开失败：{item.get('name') or n}\n原因：{exc}",
            })
            pro_console_log("UPLOAD", "压缩包展开失败", {"filename": n, "error": str(exc), "session_id": session_id})
    pro_console_log("UPLOAD", "上传批次处理完成", {"session_id": session_id, "saved_count": len(saved), "saved_files": saved})
    return saved
