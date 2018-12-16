import argparse

from redditscore import get_reddit_data as grd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect Reddit data')

    parser.add_argument('subreddits_list_file', type=str)
    parser.add_argument('output_directory', type=str)
    parser.add_argument('project_id', type=str)
    parser.add_argument('private_key_file', type=str)
    parser.add_argument('first_month', type=str)
    parser.add_argument('last_month', type=str)

    args = parser.parse_args()
    args_dict = vars(args)

    timerange = (args_dict['first_month'], args_dict['last_month'])

    with open(args_dict['subreddits_list_file']) as fin:
        subreddits = fin.read().split()

    grd.get_comments(timerange, args_dict['project_id'], args_dict['private_key_file'],
                     subreddits, score_limit=0, comments_per_month=5000, top_scores=True,
                     csv_directory=args_dict['output_directory'], verbose=True)
