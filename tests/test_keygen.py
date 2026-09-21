import os
import sys

from nyx.keygen import main


def test_keygen_creates_private_nonempty_file(tmp_path, monkeypatch):
    target = tmp_path / "key"
    monkeypatch.setattr(sys, "argv", ["nyx-keygen", "--output", str(target)])
    main()
    assert len(target.read_text()) >= 48
    assert os.stat(target).st_mode & 0o777 == 0o600
