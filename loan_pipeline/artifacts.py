import hashlib
import json
import time
import traceback
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


class Run:
    def __init__(self, root):
        from uuid import uuid4
        self.id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
        self.path = Path(root).resolve() / self.id
        self.path.mkdir(parents=True)
        self.stages = []

    @contextmanager
    def stage(self, name):
        folder = self.path / name
        folder.mkdir()
        record = {'stage': name, 'status': 'running'}
        self.stages.append(record)
        started = time.perf_counter()
        print(f'[{name}] started', flush=True)
        report = {}
        try:
            yield folder, report
            record['status'] = 'passed'
        except Exception as exc:
            record.update(status='failed', error=str(exc))
            report['traceback'] = traceback.format_exc()
            raise
        finally:
            record['seconds'] = round(time.perf_counter() - started, 3)
            write_json(folder / 'report.json', {**record, **report})
            write_json(self.path / 'stages.json', self.stages)
            print(f'[{name}] {record["status"]}', flush=True)

    def manifest(self):
        write_json(self.path / 'manifest.json', {
            str(p.relative_to(self.path)): {'sha256': digest(p), 'bytes': p.stat().st_size}
            for p in sorted(self.path.rglob('*')) if p.is_file() and p.name != 'manifest.json'
        })
