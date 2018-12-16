import argparse

import pandas as pd
from redditscore.tokenizer import CrazyTokenizer
from tqdm import tqdm


def tokenize_tweets(tweets_file, output_file):
    tweets = pd.read_csv(tweets_file, parse_dates=[
                         'created_at'], lineterminator='\n')
    tweets['id'] = pd.to_numeric(tweets['id'])
    tweets.drop_duplicates('id', inplace=True)

    tokenizer = CrazyTokenizer(
        keepcaps=False,
        decontract=True,
        ignore_stopwords='english',
        twitter_handles='realname',
        hashtags='split',
        numbers='',
        emails='',
        urls='')

    tokens = []
    for i in tqdm(range(tweets.shape[0])):
        current_tokens = tokenizer.tokenize(
            tweets.iloc[i, tweets.columns.get_loc('text')])
        tokens.append(current_tokens)
    tweets['tokens'] = tokens

    tweets.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize tweets')

    parser.add_argument('tweets_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument

    args = parser.parse_args()
    args_dict = vars(args)

    tokenize_tweets(**args_dict)
