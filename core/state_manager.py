from dataclasses import dataclass, field
from typing import Any, Dict, List
import copy
import time
import uuid

from config.settings import CHAT_HISTORY_LIMIT
from services.gcp_terminology_service import sanitize_gcp_terminology


DEFAULT_CARTOGRAPHY = {
    "graph_height": 760,
    "map_x": 0.08,
    "map_y": 0.08,
    "map_w": 0.78,
    "map_h": 0.82,
    "legend_x": 1.03,
    "legend_y": 0.50,
    "legend_len": 0.72,
    "legend_thickness": 18,
    "north_x": 0.90,
    "north_y": 0.81,
    "north_size": 0.060,
    "scale_x": 0.12,
    "scale_y": 0.095,
    "scale_width": 0.18,
    "scale_height": 0.022,
    # 图层式布局：相对画布百分比
    "map_left_pct": 0.05,
    "map_top_pct": 0.06,
    "map_width_pct": 0.76,
    "map_height_pct": 0.80,
    "legend_left_pct": 0.84,
    "legend_top_pct": 0.16,
    "legend_width_pct": 0.12,
    "legend_height_pct": 0.48,
    "north_left_pct": 0.82,
    "north_top_pct": 0.03,
    "north_width_pct": 0.13,
    "north_height_pct": 0.17,
    "scale_left_pct": 0.09,
    "scale_top_pct": 0.87,
    "scale_width_pct": 0.28,
    "scale_height_pct": 0.10,
}


def default_cartography() -> Dict[str, float]:
    return copy.deepcopy(DEFAULT_CARTOGRAPHY)


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    has_requested_map: bool = False
    som_map_shown: bool = False
    gcp_shown: bool = False
    model_explained: bool = False
    active_data_source: str = "default"
    latest_rfk_task_id: str | None = None
    latest_gcp_task_id: str | None = None
    latest_result_kind: str | None = None
    latest_uploaded_files: List[Dict[str, Any]] = field(default_factory=list)
    session_title: str | None = None
    conversation_summary: str = ""
    current_project: Dict[str, Any] = field(default_factory=dict)
    task_memory: Dict[str, Any] = field(default_factory=dict)
    latest_download_paths: Dict[str, str] = field(default_factory=dict)
    latest_mapping_output_dir: str | None = None
    latest_uncertainty_output_dir: str | None = None
    latest_upload_feedback: str | None = None
    data_role_report: Dict[str, Any] = field(default_factory=dict)
    covariate_readiness_plan: Dict[str, Any] = field(default_factory=dict)
    ai_covariate_review: str | None = None
    upload_analysis_paths: Dict[str, str] = field(default_factory=dict)
    upload_review_inventory_sig: str | None = None
    preprocess_report: Dict[str, Any] = field(default_factory=dict)
    preprocess_paths: Dict[str, str] = field(default_factory=dict)
    rfk_result_paths: Dict[str, str] = field(default_factory=dict)
    gcp_result_paths: Dict[str, str] = field(default_factory=dict)
    uploaded_result_paths: Dict[str, str] = field(default_factory=dict)
    last_result_paths: Dict[str, str] = field(default_factory=dict)
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    last_task_type: str | None = None
    last_route_decision: Dict[str, Any] = field(default_factory=dict)
    pending_pro_request_text: str | None = None
    pending_pro_target: Dict[str, Any] = field(default_factory=dict)
    # V166: when mapping is requested before enough prediction rasters are present,
    # remember the request and resume automatically after later uploads/imports.
    pending_mapping_request_text: str | None = None
    pending_mapping_reason: str | None = None
    som_palette: str = "green"
    gcp_palette: str = "green"
    map_style_prefs: Dict[str, Any] = field(default_factory=dict)
    som_cartography: Dict[str, float] = field(default_factory=default_cartography)
    gcp_cartography: Dict[str, float] = field(default_factory=default_cartography)
    active_layout_key: str | None = None
    uploaded_role_selected_idx: int = 0
    ui_notification: Dict[str, Any] = field(default_factory=dict)
    ui_notification_seq: int = 0
    created_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str) -> None:
        """Append one chat message with conservative assistant-side de-duplication.

        Polling callbacks can fire repeatedly while the browser store is being
        updated.  Without a final guard, the same completion card or download
        milestone can be appended more than once.  User messages are never
        suppressed; assistant messages with identical normalized content within
        the recent window are skipped.
        """
        role = str(role or "assistant")
        content = sanitize_gcp_terminology(content)
        if role == "assistant" and content.strip():
            norm = " ".join(content.split())
            for item in reversed(self.chat_history[-10:]):
                if item.get("role") != "assistant":
                    continue
                old_norm = " ".join(str(item.get("content") or "").split())
                if old_norm == norm:
                    return
        self.chat_history.append({"role": role, "content": content})
        if len(self.chat_history) > CHAT_HISTORY_LIMIT:
            self.chat_history = self.chat_history[-CHAT_HISTORY_LIMIT:]
        self._refresh_conversation_summary()

    def _refresh_conversation_summary(self) -> None:
        try:
            tail = self.chat_history[-12:]
            parts = []
            for item in tail:
                role = "用户" if item.get("role") == "user" else "助手"
                content = str(item.get("content", "")).strip().replace("\n", " ")
                if content:
                    parts.append(f"{role}: {content[:120]}")
            self.conversation_summary = " | ".join(parts)[-1800:]
        except Exception:
            pass

    def update_project_memory(self, **kwargs) -> None:
        data = dict(self.current_project or {})
        for k, v in kwargs.items():
            if v is not None:
                data[k] = v
        self.current_project = data

    def remember_task(self, key: str, value: Any) -> None:
        data = dict(self.task_memory or {})
        data[str(key)] = value
        self.task_memory = data

    def recent_history(self, limit: int = 8) -> List[Dict[str, str]]:
        return self.chat_history[-limit:]

    @property
    def has_rfk_result(self) -> bool:
        return bool(self.som_map_shown and self.rfk_result_paths.get("pred_tif"))

    @property
    def has_gcp_result(self) -> bool:
        return bool(self.gcp_shown and self.gcp_result_paths.get("width_tif"))

    def current_result_context(self) -> Dict[str, Any]:
        return {
            "rfk_result_paths": self.rfk_result_paths,
            "gcp_result_paths": self.gcp_result_paths,
            "uploaded_result_paths": self.uploaded_result_paths,
            "last_result_paths": self.last_result_paths,
        }

    def routing_context(self) -> Dict[str, Any]:
        return {
            "has_rfk_result": self.has_rfk_result,
            "has_gcp_result": self.has_gcp_result,
            "active_data_source": self.active_data_source,
            "last_task_type": self.last_task_type,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "has_requested_map": self.has_requested_map,
            "som_map_shown": self.som_map_shown,
            "gcp_shown": self.gcp_shown,
            "model_explained": self.model_explained,
            "active_data_source": self.active_data_source,
            "latest_rfk_task_id": self.latest_rfk_task_id,
            "latest_gcp_task_id": self.latest_gcp_task_id,
            "latest_result_kind": self.latest_result_kind,
            "latest_uploaded_files": self.latest_uploaded_files,
            "session_title": self.session_title,
            "conversation_summary": self.conversation_summary,
            "current_project": self.current_project,
            "task_memory": self.task_memory,
            "latest_download_paths": self.latest_download_paths,
            "latest_mapping_output_dir": self.latest_mapping_output_dir,
            "latest_uncertainty_output_dir": self.latest_uncertainty_output_dir,
            "latest_upload_feedback": self.latest_upload_feedback,
            "data_role_report": self.data_role_report,
            "covariate_readiness_plan": self.covariate_readiness_plan,
            "ai_covariate_review": self.ai_covariate_review,
            "upload_analysis_paths": self.upload_analysis_paths,
            "upload_review_inventory_sig": self.upload_review_inventory_sig,
            "preprocess_report": self.preprocess_report,
            "preprocess_paths": self.preprocess_paths,
            "rfk_result_paths": self.rfk_result_paths,
            "gcp_result_paths": self.gcp_result_paths,
            "uploaded_result_paths": self.uploaded_result_paths,
            "last_result_paths": self.last_result_paths,
            "chat_history": self.chat_history,
            "last_task_type": self.last_task_type,
            "last_route_decision": self.last_route_decision,
            "pending_pro_request_text": self.pending_pro_request_text,
            "pending_pro_target": self.pending_pro_target,
            "pending_mapping_request_text": self.pending_mapping_request_text,
            "pending_mapping_reason": self.pending_mapping_reason,
            "som_palette": self.som_palette,
            "gcp_palette": self.gcp_palette,
            "map_style_prefs": self.map_style_prefs,
            "som_cartography": self.som_cartography,
            "gcp_cartography": self.gcp_cartography,
            "active_layout_key": self.active_layout_key,
            "uploaded_role_selected_idx": self.uploaded_role_selected_idx,
            "ui_notification": self.ui_notification,
            "ui_notification_seq": self.ui_notification_seq,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        s = cls(session_id=data.get("session_id", str(uuid.uuid4())))
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, v)
        if not getattr(s, "som_cartography", None):
            s.som_cartography = default_cartography()
        else:
            merged = default_cartography()
            merged.update(s.som_cartography)
            s.som_cartography = merged
        if not getattr(s, "gcp_cartography", None):
            s.gcp_cartography = default_cartography()
        else:
            merged = default_cartography()
            merged.update(s.gcp_cartography)
            s.gcp_cartography = merged
        return s
