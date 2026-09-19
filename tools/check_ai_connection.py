from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.llm_service import compose_agent_reply, get_llm_health  # noqa: E402


def main() -> int:
    print("[AI-CHECK] 开始检查智能体大模型连通性。")
    health = get_llm_health()
    print("[AI-CHECK] LLM 配置：", health)
    reply = compose_agent_reply(
        user_text="请用一句话回答：成都市耕地土壤有机质制图需要准备哪些主要数据？",
        extra_context="这是智能体正式运行前的连通性自检。",
        chat_history=[],
    )
    if not reply:
        print("[AI-CHECK] 未获得大模型回复。请检查 .env 中的 API Key、base_url 和模型名称。")
        print("[AI-CHECK] 最新状态：", get_llm_health())
        return 1
    print("[AI-CHECK] 大模型回复：")
    print(reply)
    print("[AI-CHECK] 检查完成：AI 对话主干可用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
