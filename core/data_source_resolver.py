from core.query_normalizer import normalize_query


def resolve_data_source(user_text: str, uploaded_files: list[dict], current_data_source: str = "default") -> str:
    text = normalize_query(user_text)
    explicit_user = ["用我的数据", "用我上传的", "按我这个", "不要默认数据", "使用我的数据", "用这个csv", "用这个tif"]
    if any(k in text for k in explicit_user):
        return "user"
    if current_data_source == "user":
        return "user"
    return "default"


def resolve_user_data_mode(user_text: str, uploaded_files: list[dict]) -> str | None:
    text = normalize_query(user_text)
    if any(k in text for k in ["显示这个tif", "看看我上传的栅格", "打开我这个图", "显示我上传的图"]):
        return "display_only"
    if any(k in text for k in ["重新建模", "重新跑", "根据我的样本", "做预测", "做不确定性分析", "用我的数据制图"]):
        return "modeling"
    return None


def pick_latest_uploaded_tif(uploaded_files: list[dict]) -> dict | None:
    tifs = [f for f in uploaded_files if str(f.get("name", "")).lower().endswith((".tif", ".tiff"))]
    return tifs[-1] if tifs else None
