"""Generate a local Nyx bearer key without printing it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".nyx-key"))
    args = parser.parse_args()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(args.output, flags, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(secrets.token_urlsafe(48))
    print(f"Created {args.output} with mode 0600")


if __name__ == "__main__":
    main()
