from pathlib import Path
import shutil
import pandas as pd
from .artifacts import digest


def ingest(path, folder, report):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path}')
    shutil.copy2(path, folder / 'source.csv')
    data = pd.read_csv(folder / 'source.csv')
    report.update(source=str(path), sha256=digest(folder / 'source.csv'), rows=len(data),
                  columns=list(data.columns), missing=data.isna().sum().to_dict())
    return data
