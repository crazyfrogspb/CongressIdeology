import ast
import os.path as osp
import random

import boto3
import botocore
import pandas as pd
import torch.utils.data

from congressideology.config import config
from congressideology.models.reddit.data_utils import (RedditDataset,
                                                       prepare_data,
                                                       text_collate_func)


def load_comments(directory, dataset_type):
    df_comments = pd.read_csv(
        osp.join(directory, f'reddit_{dataset_type}.csv'),
        dtype={'id': str, 'subreddit': str},
        converters={'tokens': ast.literal_eval},
        usecols=['subreddit', 'tokens', 'id'],
        lineterminator='\n')

    return list(df_comments['tokens']), list(df_comments['subreddit'])


def load_data(directory, min_count=5, subsample=1.0, batch_size=32):
    data = {}
    data_loaders = {}

    for dataset_type in ['train', 'val', 'test']:
        comments, subreddits = load_comments(directory, dataset_type)

        if subsample < 1.0 and dataset_type == 'train':
            # for testing
            sample_size = int(subsample * len(comments))
            tokens, subreddits = zip(
                *random.sample(list(zip(comments, subreddits)), sample_size))

        if dataset_type == 'train':
            data_dict = prepare_data(comments, subreddits, min_count)
            label_encoder = data_dict['label_encoder']
            data[dataset_type] = RedditDataset(
                data_dict['lang'], data_dict['pairs'])
            shuffle = True
        else:
            data_dict = prepare_data(
                comments, subreddits, min_count, label_encoder)
            data[dataset_type] = RedditDataset(
                data['train'].lang, data_dict['pairs'])
            shuffle = False

        data_loaders[dataset_type] = torch.utils.data.DataLoader(dataset=data[dataset_type],
                                                                 batch_size=batch_size,
                                                                 collate_fn=text_collate_func,
                                                                 shuffle=shuffle)
    return data, data_loaders, label_encoder


def download_model(run_uuid):
    if not osp.exists(osp.join(config.model_dir, 'reddit', f"checkpoint_{run_uuid}.pth")):
        s3 = boto3.resource('s3')
        try:
            s3.Bucket(config.bucket_name).download_file(
                f"congressideology/models/reddit/checkpoint_{run_uuid}.pth",
                osp.join(config.model_dir, 'reddit', f"checkpoint_{run_uuid}.pth"))
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise ValueError(f'The model with id {run_uuid} does not exist')
            else:
                raise
