import json
import os


class Checkpoint:
    """JSON progress file so a crashed run resumes without repeating API calls.

    Only small, derived results are stored (model responses, page/panel
    indices); images are re-extracted from the PDF deterministically on resume.
    """

    def __init__(self, path, fingerprint, fresh=False):
        self.path = path
        self.data = {"fingerprint": fingerprint}
        if fresh or not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("fingerprint") == fingerprint:
            self.data = saved
            print(f"Resuming from checkpoint {path}")
        else:
            print(f"Checkpoint {path} is for a different input; starting fresh.")

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        # Write then rename, so a crash mid-write never corrupts the checkpoint
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f)
        os.replace(tmp_path, self.path)
