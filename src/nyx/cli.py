"""Production gateway entry point."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def main():
    key_file = os.getenv("NYX_API_KEY_FILE")
    if key_file:
        key = Path(key_file).read_text().strip()
        if not key:
            raise SystemExit("NYX_API_KEY_FILE is empty")
        os.environ["NYX_HOSTED_API_KEY_SHA256"] = hashlib.sha256(key.encode()).hexdigest()
    elif not os.getenv("NYX_HOSTED_API_KEY_SHA256"):
        raise SystemExit("Set NYX_API_KEY_FILE or NYX_HOSTED_API_KEY_SHA256")

    import uvicorn

    uvicorn.run(
        "nyx.hosted_api:create_app",
        factory=True,
        host=os.getenv("NYX_HOST", "127.0.0.1"),
        port=int(os.getenv("NYX_PORT", "8000")),
        access_log=False,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
