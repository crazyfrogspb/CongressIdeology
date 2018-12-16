import argparse
import os.path as osp
import random

import mlflow
import numpy as np
from sklearn import metrics

from congressideology.config import config
from congressideology.models.reddit.load_data import load_comments
from congressideology.models.reddit.training_utils import train_model
from redditscore.models import fasttext_mod


def train_fasttext(directory, n_epochs, ngrams, embedding_size, learning_rate, random_state, subsample):
    training_parameters = locals()
    training_parameters['model_type'] = 'fasttext'

    tokens_train, subreddits_train = load_comments(directory, 'train')
    tokens_val, subreddits_val = load_comments(directory, 'val')

    if subsample < 1.0:
        # for testing
        sample_size = int(subsample * len(tokens_train))
        tokens_train, subreddits_train = zip(
            *random.sample(list(zip(tokens_train, subreddits_train)), sample_size))

    ft_model = fasttext_mod.FastTextModel(
        epoch=n_epochs, dim=embedding_size, lr=learning_rate,
        wordNgrams=ngrams, thread=6, random_state=random_state)

    ft_model.fit(tokens_train, subreddits_train)

    probs = ft_model.predict_proba(tokens_val)
    preds = probs.idxmax(axis=1)

    acc = metrics.accuracy_score(subreddits_val, preds)
    log_loss = metrics.log_loss(
        subreddits_val, probs, labels=sorted(np.unique(subreddits_val)))

    with mlflow.start_run():
        for par_name, par_value in training_parameters.items():
            mlflow.log_param(par_name, par_value)
        mlflow.log_metric('best_loss', log_loss)
        mlflow.log_metric('best_acc', acc)

        ft_model.save_model(osp.join(config.model_dir, 'reddit',
                                     f'ft_{mlflow.active_run()._info.run_uuid}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Reddit model')

    parser.add_argument('directory', type=str)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--min_count', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--clipping_value', type=float, default=7.5)
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--early_stopping', type=int, default=3)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--random_state', type=int, default=24)
    parser.add_argument('--ngrams', type=int, default=1)

    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict['model_type'] == 'fasttext':
        train_fasttext(args_dict['directory'], args_dict['n_epochs'], args_dict['wordNgrams'],
                       args_dict['embedding_size'], args_dict['learning_rate'],
                       args_dict['random_state'], args_dict['subsample'])
    else:
        train_model(**args_dict)
