import ast
import os.path as osp

import pandas as pd

CURRENT_PATH = osp.dirname(osp.realpath(__file__))
DATA_DIR = osp.join(CURRENT_PATH, '..', '..', 'data')


def prepare_data(input_file, meta_info_file, train_congress, test_congress):
    tweets = pd.read_csv(input_file,
                         converters={'tokens': ast.literal_eval},
                         lineterminator='\n',
                         parse_dates=['created_at'])

    tweets['doc'] = tweets['tokens'].str.join(' ')

    meta_info = pd.read_csv(meta_info_file)

    all_congress = set(train_congress + test_congress)
    for congress in all_congress:
        if congress 
