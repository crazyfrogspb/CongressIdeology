import argparse
import glob
import os.path as osp

import pandas as pd
from redditscore.tokenizer import CrazyTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def tokenize_reddit(comments_directory, output_directory,
                    subsample=100000, val_size=0.1, test_size=0.1, random_state=24):
    csv_files = glob.glob(osp.join(comments_directory, '*.csv'))
    df_comments = pd.concat((pd.read_csv(csv_file, lineterminator='\n', usecols=[
        'id', 'body', 'subreddit', 'created_utc']) for csv_file in csv_files))

    df_comments.drop_duplicates('id', inplace=True)
    df_comments['created_utc'] = pd.to_datetime(
        df_comments['created_utc'], unit='s')

    df_comments = df_comments.sample(frac=1.0, random_state=random_state)
    df_comments = df_comments.groupby('subreddit').head(subsample)

    tokenizer = CrazyTokenizer(
        keepcaps=False,
        decontract=True,
        ignore_stopwords='english',
        subreddits='',
        reddit_usernames='',
        numbers='',
        emails='',
        urls='')

    tokens = []
    for i in tqdm(range(df_comments.shape[0])):
        current_tokens = tokenizer.tokenize(
            df_comments.iloc[i, df_comments.columns.get_loc('body')])
        tokens.append(current_tokens)
    df_comments['tokens'] = tokens
    del tokens

    df_train_val, df_test = train_test_split(
        df_comments, test_size=test_size, random_state=random_state, shuffle=True)
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_size, random_state=random_state, shuffle=True)

    df_train = df_train.loc[df_train.tokens.str.len() > 0]
    df_val = df_val.loc[df_val.tokens.str.len() > 0]
    df_test = df_test.loc[df_test.tokens.str.len() > 0]

    df_train.to_csv(osp.join(output_directory, 'reddit_train.csv'), index=False)
    df_val.to_csv(osp.join(output_directory, 'reddit_val.csv'), index=False)
    df_test.to_csv(osp.join(output_directory, 'reddit_test.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize tweets')

    parser.add_argument('comments_directory', type=str)
    parser.add_argument('output_directory', type=str)
    parser.add_argument('--subsample', type=int, default=100000)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--random_state', type=int, default=24)

    args = parser.parse_args()
    args_dict = vars(args)

    tokenize_reddit(**args_dict)
