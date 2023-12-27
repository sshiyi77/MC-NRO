from math import log

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


ORIGINAL_PATH = Path(__file__).parent / 'data' / 'original'
FOLD_PATH = Path(__file__).parent / 'data' / 'folds'


def partition(name, scale=True, noise_type=None, noise_level=0.0):
    fold_path = FOLD_PATH / name

    if not fold_path.exists():
        fold_path.mkdir(parents=True, exist_ok=True)

    original_path = ORIGINAL_PATH / ('%s-full.dat' % name)

    skiprows = 0

    with open(original_path) as f:
        for line in f:
            if line.startswith('@'):
                skiprows += 1
                if line.startswith('@input'):
                    inputs = [l.strip() for l in line[8:].split(',')]
                elif line.startswith('@output'):
                    outputs = [l.strip() for l in line[8:].split(',')]
            else:
                break

    df = pd.read_csv(original_path, skiprows=skiprows, header=None)

    df.columns = inputs + outputs

    categorical_columns = []
    numeric_columns = []

    for column in df.columns[:-1]:
        try:
            df[column].astype(float)
            numeric_columns.append(column)
        except ValueError:
            categorical_columns.append(column)

    df_encoded = pd.concat([pd.get_dummies(df[inputs]), df[outputs]], axis=1)

    N = len(df)
    idf_values = {}
    for column in categorical_columns:
        df_freq = df[column].value_counts()
        idf = df_freq.apply(lambda x: log((N + 1) / (x + 1)) + 1)  # 平滑处理
        idf_values[column] = idf

    for column in categorical_columns:
        for value in idf_values[column].index:
            encoded_col_name = f"{column}_{value}"
            df_encoded[encoded_col_name] *= idf_values[column][value]

    matrix = df_encoded.values
    X, y = matrix[:, :-1], matrix[:, -1]
    y = LabelEncoder().fit_transform(y)

    for i in range(5):
        skf = StratifiedKFold(n_splits=2, shuffle=True)
        skf.get_n_splits(X, y)
        splits = list(skf.split(X, y))

        for j in range(len(splits)):
            train_index, test_index = splits[j]
            train_set = [X[train_index].copy(), y[train_index]]
            test_set = [X[test_index].copy(), y[test_index]]

            if scale:
                scaler = MinMaxScaler().fit(X[train_index])
                train_set[0] = scaler.transform(train_set[0])
                test_set[0] = scaler.transform(test_set[0])

            if noise_type == 'class' and noise_level > 0.0:
                classes = np.unique(y)
                sizes = [sum(y == c) for c in classes]
                minority_class = classes[np.argmin(sizes)]
                majority_class = classes[np.argmax(sizes)]

                assert minority_class != majority_class

                for k in range(len(train_set[1])):
                    if train_set[1][k] != minority_class and np.random.rand() < noise_level:
                        train_set[1][k] = minority_class

            if noise_type == 'attribute' and noise_level > 0.0:
                maximum = np.max(train_set[0], axis=0)
                minimum = np.min(train_set[0], axis=0)

                for k in range(train_set[0].shape[1]):
                    train_set[0][:, k] += np.random.normal(loc=0.0, scale=noise_level * (maximum[k] - minimum[k]) / 5.0,
                                                           size=train_set[0].shape[0])

            train_set = pd.DataFrame(np.c_[train_set[0], train_set[1]])
            test_set = pd.DataFrame(np.c_[scaler.transform(X[test_index]), y[test_index]])

            dfs = {'train': train_set, 'test': test_set}

            for partition_type in ['train', 'test']:
                file_name = name + '.' + str(i + 1) + '.' + str(j + 1) + '.' + partition_type + '.csv'
                path = FOLD_PATH / name / file_name
                dfs[partition_type].to_csv(path, index=False, header=df_encoded.columns)


def load(name, partition, fold):
    partitions = []

    for partition_type in ['train', 'test']:
        path = FOLD_PATH / "".join(name) / ('%s.%d.%d.%s.csv' % ("".join(name), partition, fold, partition_type))
        df = pd.read_csv(path)
        matrix = df.values
        X, y = matrix[:, :-1], matrix[:, -1]
        partitions.append([X, y])

    return partitions


def names():
    return [path.name.replace('-full.dat', '') for path in ORIGINAL_PATH.iterdir()]


if __name__ == '__main__':
    print('Partitioning...')

    for name in names():
        print('Partitioning %s...' % name)
        partition(name, scale=True, noise_type=None, noise_level=0.0)