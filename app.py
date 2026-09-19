from __future__ import annotations

import os
import logging
from pathlib import Path

from config.settings import ENV_PATH, BASE_DIR
from ui.dashboard import create_app


class QuietDashFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        noisy = (
            'POST /__heartbeat HTTP/1.1" 204',
            'POST /__shutdown_intent HTTP/1.1" 204',
            'POST /_dash-update-component HTTP/1.1" 204',
        )
        return not any(p in msg for p in noisy)



def main():
    print(f"[BOOT] app.py path = {Path(__file__).resolve()}")
    print(f"[BOOT] cwd = {Path.cwd()}")
    print(f"[BOOT] project_root = {BASE_DIR}")
    print(f"[BOOT] env_path = {ENV_PATH}")
    print(f"[BOOT] env_exists = {ENV_PATH.exists()}")
    print(f"[BOOT] API key exists = {bool(os.getenv('DASHSCOPE_API_KEY') or os.getenv('QWEN_API_KEY') or os.getenv('OPENAI_API_KEY'))}")

    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.addFilter(QuietDashFilter())
    werkzeug_logger.setLevel(logging.ERROR)
    app = create_app()
    app.run(debug=False, host="127.0.0.1", port=7050)


if __name__ == "__main__":
    main()
