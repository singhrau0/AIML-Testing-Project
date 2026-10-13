import html
import json
from .cleaning_analysis import render_analysis


def summarize(run, folder, report):
    records = []
    for stage in run.stages:
        path = run.path / stage['stage'] / 'report.json'
        if path.exists():
            records.append((stage['stage'], json.loads(path.read_text())))
    sections = ''.join(f'<section><h2>{html.escape(name)}</h2>'
                      + (render_analysis(value) if name == '04_cleaning' else '')
                      + f'<details><summary>Full process report</summary><pre>{html.escape(json.dumps(value, indent=2))}</pre></details></section>'
                       for name, value in records)
    (folder / 'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Loan pipeline report</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#f5f7fa;color:#182434}'
        'section{background:white;padding:20px;margin:16px 0;border-radius:10px;overflow:auto}'
        'table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd;text-align:left}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        f'</head><body><h1>Loan MLOps run</h1><p>{html.escape(run.id)}</p>'
        '<p>Predicts historical approval labels. Candidate models require review before production use.</p>'
        + sections + '</body></html>')
    report.update(reported_stages=len(records), report='index.html')
