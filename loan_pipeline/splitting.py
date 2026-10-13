from sklearn.model_selection import train_test_split
from .config import TARGET


def split(data, config, folder, report):
    train, test = train_test_split(data, test_size=config.test_size, random_state=config.seed,
                                   stratify=data[TARGET])
    if train[TARGET].value_counts().min() < config.cv_folds:
        raise ValueError('Too few training examples per class for configured CV folds')
    assert set(train.Loan_ID).isdisjoint(test.Loan_ID)
    train.to_csv(folder / 'train.csv', index=False)
    test.to_csv(folder / 'test.csv', index=False)
    report.update(train_rows=len(train), test_rows=len(test), seed=config.seed,
                  train_classes=train[TARGET].value_counts().to_dict(),
                  test_classes=test[TARGET].value_counts().to_dict(), disjoint_ids=True,
                  strategy='Stratified holdout; CV and all learned transforms restricted to training data.')
    return train, test
