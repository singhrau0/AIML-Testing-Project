from .config import TARGET


def profile(data, folder, report):
    data.describe(include='all').to_csv(folder / 'descriptive_statistics.csv')
    data.isna().sum().rename('missing_count').to_csv(folder / 'missing_values.csv')
    report.update(rows=len(data), target_counts=data[TARGET].value_counts().to_dict(),
                  missing_fraction=data.isna().mean().to_dict(),
                  limitation='Source collection date, sampling design and label provenance are not supplied.')
