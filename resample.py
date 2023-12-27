import argparse
import algorithm
import datasets
import logging
import numpy as np
import pandas as pd

from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).parent / 'data'
OUTPUT_PATH = Path(__file__).parent / 'oversampled_data'


def prepare(dataset, partition, fold, mode='OVE', output_path=OUTPUT_PATH, k1=1, k2=1,
            cleaning_strategy='translate and remove', selection_strategy='proportional', p_norm=2.0, gamma=0.1,
            method='sampling'):
    logging.info('Processing fold %dx%d of dataset "%s"...' % (partition, fold, dataset))

    output_path = Path(output_path) / "".join(dataset)
    output_path.mkdir(parents=True, exist_ok=True)

    (X_train, y_train), (X_test, y_test) = datasets.load(dataset, partition, fold)

    header = pd.read_csv(
        DATA_PATH / 'folds' / "".join(dataset) / ('%s.%d.%d.train.csv' % ("".join(dataset), partition, fold))
    ).columns

    if mode == 'OVE':
        logging.info('Training distribution before resampling: %s.' % Counter(y_train))

        X_train, y_train = algorithm.MultiClassNRO(k1=k1, k2=k2,
                                                   cleaning_strategy=cleaning_strategy,
                                                   selection_strategy=selection_strategy,
                                                   gamma=gamma, p_norm=p_norm, method=method
                                                   ).fit_sample(X_train, y_train)

        logging.info('Training distribution after resampling: %s.' % Counter(y_train))

        csv_path = output_path / ('%s.%d.%d.train.oversampled.csv' % ("".join(dataset), partition, fold))

        pd.DataFrame(np.c_[X_train, y_train]).to_csv(csv_path, index=False, header=header)

    elif mode == 'OVO':
        classes = np.unique(np.concatenate([y_train, y_test]))

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                logging.info('Resampling class %s vs. class %s.' % (classes[i], classes[j]))

                indices = ((y_train == classes[i]) | (y_train == classes[j]))

                X, y = X_train[indices].copy(), y_train[indices].copy()

                logging.info('Training distribution before resampling: %s.' % Counter(y))

                X, y = algorithm.NRO(k1=k1, k2=k2,
                                     cleaning_strategy=cleaning_strategy, selection_strategy=selection_strategy,
                                     gamma=gamma, p_norm=p_norm
                                     ).fit_sample(X, y)

                logging.info('Training distribution after resampling: %s.' % Counter(y))

                csv_path = output_path / ('%s.%d.%d.train.oversampled.%dv%d.csv' %
                                          ("".join(dataset), partition, fold, classes[i], classes[j]))

                pd.DataFrame(np.c_[X, y]).to_csv(csv_path, index=False, header=header)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, choices=datasets.names(), required=True)
    parser.add_argument('-partition', type=int, choices=[1, 2, 3, 4, 5], required=True)
    parser.add_argument('-fold', type=int, choices=[1, 2], required=True)
    parser.add_argument('-mode', type=str, choices=['OVE', 'OVO'], default='OVE')
    parser.add_argument('-output_path', type=str, default=OUTPUT_PATH)
    parser.add_argument('-k1', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], required=True)
    parser.add_argument('-k2', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], required=True)
    parser.add_argument('-cleaning_strategy', type=str, choices=['translate and remove', 'translate', 'remove'],
                        default='translate+remove')
    parser.add_argument('-selection_strategy', type=str, choices=['proportional', 'random'], default='proportional')
    parser.add_argument('-p_norm', type=float, default=2.0)
    parser.add_argument('-gamma', type=float, choices=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                        default=0.1)
    parser.add_argument('-method', type=str, choices=['sampling', 'complete'], default='sampling')

    args = parser.parse_args()

    prepare(**vars(args))
