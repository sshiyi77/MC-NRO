import algorithms
import datasets
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / 'data'
OUTPUT_PATH = Path(__file__).parent / 'oversampled_data'


def prepare(name, partition, fold, mode='OVA'):
    output_path = OUTPUT_PATH / "".join(name)
    output_path.mkdir(parents=True, exist_ok=True)

    (X_train, y_train), (X_test, y_test) = datasets.load(name, partition, fold)

    header = pd.read_csv(
        DATA_PATH / 'folds' / "".join(name) / ('%s.%d.%d.train.csv' % ("".join(name), partition, fold))
    ).columns

    if mode == 'OVA':

        X_train, y_train = algorithms.MultiClassNRO().fit_sample(X_train, y_train)

        csv_path = output_path / ('%s.%d.%d.train.oversampled.csv' % ("".join(name), partition, fold))

        pd.DataFrame(np.c_[X_train, y_train]).to_csv(csv_path, index=False, header=header)

    elif mode == 'OVO':
        classes = np.unique(np.concatenate([y_train, y_test]))

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                indices = ((y_train == classes[i]) | (y_train == classes[j]))

                X, y = X_train[indices].copy(), y_train[indices].copy()

                X, y = algorithms.NRO().fit_sample(X, y)

                csv_path = output_path / ('%s.%d.%d.train.oversampled.%dv%d.csv' %
                                          ("".join(name), partition, fold, classes[i], classes[j]))

                pd.DataFrame(np.c_[X, y]).to_csv(csv_path, index=False, header=header)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    partition = [1, 2, 3, 4, 5]
    fold = [1, 2]

    for name in datasets.names():
        for i in partition:
            for j in fold:
                prepare(name, partition=i, fold=j)
