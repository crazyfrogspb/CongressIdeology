import os.path as osp

import torch

CURRENT_PATH = osp.dirname(osp.realpath(__file__))


class Config():
    num_batches = 30
    w2v_iters = 20
    test_split = 0.2
    random_state = 24
    data_dir = osp.join(CURRENT_PATH, '..', 'data')
    model_dir = osp.join(CURRENT_PATH, '..', 'models')
    bucket_name = 'nikitinphd'


class RedditConfig():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PAD_token = 0
    UNK_token = 1
    max_length = 100
    decay_patience = 5
    decay_factor = 0.1
    logging_freq = 500


config = Config()
reddit_config = RedditConfig()
