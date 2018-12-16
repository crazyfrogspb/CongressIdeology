import argparse
import os
import os.path as osp
from time import sleep

import pandas as pd
from congress import Congress
from congress.utils import NotFound
from dotenv import find_dotenv, load_dotenv

from congressideology.config import config

load_dotenv(find_dotenv())

congress = Congress(os.getenv('PROPUBLICA_API_KEY'))

KEEP_FIELDS = ['congress', 'bill_type', 'title', 'sponsor', 'sponsor_id',
               'sponsor_party', 'introduced_date', 'cosponsors_by_party', 'summary']


def collect_bill_data(congress_nums, output_file, timeout=0.1):
    bills = []
    for congress_num in congress_nums:
        votes = pd.read_csv(
            osp.join(config.data_dir, 'raw', 'votes', f'HS{congress_num}_rollcalls.csv'))
        votes.dropna(subset=['bill_number'], axis=0, inplace=True)
        for bill_id in votes['bill_number'].unique():
            try:
                bill_info = congress.bills.get(
                    bill_id=bill_id, congress=congress_num)
            except NotFound:
                continue
            bill_info = dict((k, bill_info[k])
                             for k in KEEP_FIELDS if k in bill_info)
            bill_info['bill_id'] = bill_id
            bills.append(bill_info)
            sleep(timeout)

    df_bills = pd.DataFrame(bills)
    df_bills.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect Bill data')

    parser.add_argument('output_file', type=str)
    parser.add_argument('--congress_nums', type=int, nargs='*', required=True)

    args = parser.parse_args()
    args_dict = vars(args)

    collect_bill_data(**args_dict)
