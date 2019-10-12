import pandas as pd

from .common.logger import create_logger
from . import config
from .tvp import train, predict
from .helper_func import train_valid_split_v2


def save_split_data():
    df = pd.read_csv(config.TRAIN_PATH)
    print(df.head())
    df_trn, df_val = train_valid_split_v2(df, valid_ich_ratio=0.1)
    print(df_trn.head())
    print(df_val.head())

    df_trn.to_csv('df_trn.csv', index=False)
    df_val.to_csv('df_val.csv', index=False)


if __name__ == '__main__':
    create_logger('log/brain.log')

    train()
    # predict()
    # save_split_data()
