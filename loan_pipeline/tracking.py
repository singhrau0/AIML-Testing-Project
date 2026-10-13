import importlib.metadata
import platform
import subprocess
from .artifacts import write_json


def track(run, config, folder, report):
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    write_json(folder / 'environment.json', {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages})
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    report.update(run_id=run.id, config=config.to_dict(), git_commit=commit,
                  tracking='Immutable per-run local artifacts, dataset/source hashes and environment versions')
