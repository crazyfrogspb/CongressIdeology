import argparse
import json
import os

from dotenv import find_dotenv, load_dotenv
from redditscore.get_twitter_data import collect_congress_tweets

load_dotenv(find_dotenv())


def collect_tweets(meta_info_file, tweets_file, twitter_creds_file, congress_nums):
    with open(twitter_creds_file) as f:
        twitter_creds_list = list(json.load(f).values())

    collect_congress_tweets(congress_nums, tweets_file,
                            meta_info_file, '2015-01-03', twitter_creds_list,
                            propublica_api_key=os.getenv('PROPUBLICA_API_KEY'),
                            append_frequency=1, shuffle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect tweets')

    parser.add_argument('meta_info_file', type=str)
    parser.add_argument('tweets_file', type=str)
    parser.add_argument('twitter_creds_file', type=str)
    parser.add_argument('--congress_nums', type=int, nargs='*', required=True)

    args = parser.parse_args()
    args_dict = vars(args)

    collect_tweets(**args_dict)
