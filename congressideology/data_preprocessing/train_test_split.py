import os.path as osp

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from congressideology.config import config

CURRENT_PATH = osp.dirname(osp.realpath(__file__))
DATA_DIR = osp.join(CURRENT_PATH, '..', '..', 'data')

congress_meta = pd.read_csv(osp.join(DATA_DIR, 'raw', 'congress_meta.csv'))
congress_meta['party_bin'] = np.where(congress_meta.party == 'R', 1, 0)

for congress_num in [114, 115]:
    congress = congress_meta.loc[congress_meta.congress_num == congress_num].copy(
    )
    congress.dropna(subset=['twitter_account'], inplace=True)
    members_info = pd.read_csv(
        osp.join(DATA_DIR, 'raw', f'HS{congress_num}_members.csv'))
    congress = pd.merge(congress, members_info[[
                        'icpsr', 'nominate_dim1', 'nominate_dim2']], left_on='icpsr_id', right_on='icpsr')

    twitter_accounts = list(congress['twitter_account'].unique())
    train_handles, test_handles = train_test_split(
        twitter_accounts, test_size=config.test_split, random_state=config.random_state, shuffle=True)

    congress_train = congress.loc[congress['twitter_account'].isin(
        train_handles)].copy()
    congress_test = congress.loc[congress['twitter_account'].isin(
        test_handles)].copy()

    train_icpsr = list(congress_train['icpsr'].unique())
    test_icpsr = list(congress_test['icpsr'].unique())
