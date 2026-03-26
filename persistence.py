import json
import os
import tempfile


def atomic_write_json(path: str, data):
    dir_path = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_path, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=dir_path, suffix=".tmp", delete=False
        ) as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            tmp_path = f.name
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

